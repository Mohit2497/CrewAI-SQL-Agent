# Fix for SQLite version issue in deployment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
import time
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import tempfile
from typing import Optional, Dict, List
import threading
from functools import wraps
import requests

# Import CrewAI components
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool

# Fixed LangChain imports
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CrewAI SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rate Limiter Class (for GROQ)
class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self, calls_per_minute: int = 20):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()
        self.call_times = []
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    self.call_times = [t for t in self.call_times if now - t < 60]
            
            self.call_times.append(now)

# Global rate limiter for GROQ
rate_limiter = RateLimiter(calls_per_minute=15)

# Retry decorator
def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait_time = backoff_factor ** attempt
                        if attempt < max_retries - 1:
                            st.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                    raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .upload-section {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .uploadedFile {
        border: 2px dashed #1f77b4 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        background-color: #f0f8ff !important;
    }
    .complex-query-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .rate-limit-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .model-status {
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .ollama-status {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .groq-status {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'uploaded_tables' not in st.session_state:
    st.session_state.uploaded_tables = []
if 'csv_preview_data' not in st.session_state:
    st.session_state.csv_preview_data = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
if 'temp_db_path' not in st.session_state:
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    st.session_state.temp_db_path = temp_file.name
    temp_file.close()
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = None
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = None

# Check Ollama availability
def check_ollama_status():
    """Check if Ollama is running and has models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            codellama_models = [m for m in models if 'codellama' in m.get('name', '').lower()]
            if codellama_models:
                return True, "ready", codellama_models[0]['name']
            elif models:
                return True, "no_codellama", models[0]['name']
            else:
                return True, "no_models", None
        return False, "not_running", None
    except:
        return False, "not_running", None

# Rate-limited LLM wrapper for GROQ
class RateLimitedLLM:
    """Wrapper for GROQ LLM with rate limiting"""
    def __init__(self, llm, rate_limiter):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.max_retries = 3
    
    @retry_with_backoff(max_retries=3)
    def invoke(self, prompt: str, **kwargs):
        """Invoke LLM with rate limiting"""
        self.rate_limiter.wait()
        st.session_state.api_calls += 1
        st.session_state.last_api_call = datetime.now()
        
        try:
            # Truncate prompt if too long to save tokens
            if len(prompt) > 2000:
                prompt = prompt[:2000] + "..."
            
            response = self.llm.invoke(prompt, **kwargs)
            return response
        except Exception as e:
            if "rate_limit" in str(e).lower():
                st.error("Rate limit exceeded. Waiting before retry...")
                time.sleep(30)
                raise
            else:
                raise
    
    def __getattr__(self, name):
        """Proxy other attributes to the wrapped LLM"""
        return getattr(self.llm, name)

# Initialize LLM with Ollama primary and GROQ fallback
@st.cache_resource
def initialize_llm():
    """Initialize LLM with Ollama (primary) and GROQ (fallback)"""
    
    # Check Ollama first
    ollama_available, ollama_status, available_model = check_ollama_status()
    
    if ollama_available and ollama_status in ["ready", "no_codellama"]:
        try:
            # Try CodeLlama first, then any available model
            model_to_use = "codellama:7b-instruct" if ollama_status == "ready" else available_model
            
            st.info(f"🔄 Connecting to Ollama ({model_to_use})...")
            
            llm = Ollama(
                model=model_to_use,
                base_url="http://localhost:11434",
                temperature=0,
                num_ctx=4096,
                num_predict=1000,
                top_k=10,
                top_p=0.95,
                system="""You are an expert SQL assistant. You excel at:
                - Writing efficient SQL queries including JOINs, CTEs, and window functions
                - Understanding database schemas and relationships
                - Explaining query results clearly
                - Providing data analysis insights
                Be precise and concise."""
            )
            
            # Test the connection
            test_response = llm.invoke("SELECT 1")
            st.success(f"✅ Connected to {model_to_use} via Ollama!")
            st.success("🚀 **No API limits** - Run unlimited queries!")
            st.session_state.llm_type = "ollama"
            
            return llm
            
        except Exception as e:
            st.warning(f"⚠️ Ollama connection failed: {str(e)}")
    
    elif ollama_status == "no_models":
        st.info("""
        📦 **Ollama is running but no models found!**
        
        Run this in terminal:
        ```bash
        ollama pull codellama:7b-instruct
        ```
        """)
    else:
        st.info("""
        💡 **Want unlimited queries? Install Ollama:**
        1. Download from https://ollama.ai
        2. Run: `ollama serve`
        3. Pull model: `ollama pull codellama:7b-instruct`
        """)
    
    # Try GROQ as fallback
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.warning("⚡ Using GROQ API (rate limited - 15 calls/minute)")
        try:
            # Try different GROQ models
            models_to_try = [
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "llama3-8b-8192"
            ]
            
            llm = None
            for model in models_to_try:
                try:
                    llm = ChatGroq(
                        temperature=0,
                        model_name=model,
                        api_key=groq_key,
                        max_tokens=500,
                        timeout=30,
                        max_retries=2
                    )
                    # Test the model
                    test_response = llm.invoke("Say test")
                    st.success(f"✅ Connected to GROQ ({model})")
                    st.session_state.llm_type = "groq"
                    break
                except Exception as e:
                    continue
            
            if llm:
                return RateLimitedLLM(llm, rate_limiter)
            else:
                st.error("❌ All GROQ models failed")
        except Exception as e:
            st.error(f"❌ GROQ initialization failed: {str(e)}")
    
    # No LLM available
    st.error("""
    ❌ **No LLM available!**
    
    **Option 1 (Recommended - Free & Unlimited):**
    1. Install Ollama from https://ollama.ai
    2. Run: `ollama serve`
    3. Pull model: `ollama pull codellama:7b-instruct`
    
    **Option 2 (Rate Limited):**
    Add GROQ_API_KEY to your .env file
    """)
    
    return None

# Initialize Database
def get_database_connection():
    """Get current database connection"""
    try:
        db_path = st.session_state.temp_db_path
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        return db, db_path
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None, None

# CSV/Excel Processing Functions
def validate_csv_structure(df):
    """Comprehensive CSV validation with detailed feedback"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check 1: Empty DataFrame
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("The file is empty.")
        return validation_results
    
    # Check 2: No columns
    if len(df.columns) == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("No columns found in the file.")
        return validation_results
    
    # Check 3: No rows
    if len(df) == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("No data rows found in the file.")
        return validation_results
    
    # Check 4: All NaN values
    if df.isna().all().all():
        validation_results['is_valid'] = False
        validation_results['errors'].append("All values in the file are empty or NaN.")
        return validation_results
    
    # Check 5: Duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        validation_results['warnings'].append(f"Duplicate column names found: {duplicate_cols}. They will be renamed.")
    
    # Check 6: Too many unnamed columns
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
    if len(unnamed_cols) > 0:
        validation_results['warnings'].append(f"{len(unnamed_cols)} unnamed column(s) detected. Consider adding proper headers.")
    
    # Check 7: Special characters in column names
    special_char_cols = [col for col in df.columns if not str(col).replace('_', '').replace('-', '').replace('.', '').replace(' ', '').isalnum()]
    if special_char_cols and len(special_char_cols) <= 3:
        validation_results['info'].append(f"Special characters in column names will be cleaned: {special_char_cols}")
    elif special_char_cols:
        validation_results['info'].append(f"Special characters in {len(special_char_cols)} column names will be cleaned.")
    
    # Check 8: Data quality metrics
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    
    if missing_percentage > 50:
        validation_results['warnings'].append(f"High percentage of missing data: {missing_percentage:.1f}%")
    elif missing_percentage > 20:
        validation_results['info'].append(f"Moderate amount of missing data: {missing_percentage:.1f}%")
    
    # Check 9: Row count
    if len(df) < 10:
        validation_results['warnings'].append(f"Very few rows ({len(df)}). Analysis might be limited.")
    elif len(df) > 1000000:
        validation_results['warnings'].append(f"Large dataset ({len(df):,} rows). Processing might be slower.")
    
    # Check 10: Column data types
    text_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) == 0:
        validation_results['info'].append("No numeric columns detected. Statistical analysis might be limited.")
    
    if len(text_cols) == len(df.columns):
        validation_results['info'].append("All columns are text. Consider if any should be numeric.")
    
    return validation_results

def display_validation_results(validation_results):
    """Display validation results in a user-friendly way"""
    if not validation_results['is_valid']:
        for error in validation_results['errors']:
            st.error(f"❌ {error}")
        return False
    
    # Display warnings
    for warning in validation_results['warnings']:
        st.warning(f"⚠️ {warning}")
    
    # Display info messages
    for info in validation_results['info']:
        st.info(f"ℹ️ {info}")
    
    return True

def process_csv_upload(uploaded_file):
    """Process uploaded CSV file and return DataFrame with validation"""
    try:
        # Check if file is empty
        if uploaded_file.size == 0:
            st.error("❌ The uploaded file is empty (0 bytes). Please upload a CSV file with data.")
            return None
        
        # Check file extension
        if not uploaded_file.name.lower().endswith('.csv'):
            st.warning("⚠️ File doesn't have .csv extension. Attempting to read as CSV anyway...")
        
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        successful_encoding = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                # First, try to read just the first few rows to check validity
                df_preview = pd.read_csv(uploaded_file, encoding=encoding, nrows=5)
                
                # If preview works, read the full file
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                successful_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                st.error("❌ The CSV file appears to be empty or corrupted.")
                return None
            except pd.errors.ParserError as e:
                st.error(f"❌ Error parsing CSV: {str(e)}")
                return None
        
        if df is None:
            st.error("❌ Could not read the file. Please ensure it's a valid CSV file.")
            return None
        
        # Show encoding used if not UTF-8
        if successful_encoding and successful_encoding != 'utf-8':
            st.info(f"ℹ️ File read successfully using {successful_encoding} encoding.")
        
        # Validate the DataFrame
        validation_results = validate_csv_structure(df)
        
        # Display validation results
        if not display_validation_results(validation_results):
            return None
        
        # Clean column names
        original_columns = df.columns.tolist()
        df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        
        # Show column renaming info if any changes were made
        renamed_cols = [(orig, new) for orig, new in zip(original_columns, df.columns) if orig != new]
        if renamed_cols:
            with st.expander("Column names were cleaned"):
                for orig, new in renamed_cols[:10]:  # Show first 10
                    st.text(f"{orig} → {new}")
                if len(renamed_cols) > 10:
                    st.text(f"... and {len(renamed_cols) - 10} more")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.error("Please check if your file is a valid CSV format.")
        return None

def process_excel_upload(uploaded_file):
    """Process uploaded Excel file and return DataFrame with validation"""
    try:
        # Check if file is empty
        if uploaded_file.size == 0:
            st.error("❌ The uploaded file is empty (0 bytes). Please upload an Excel file with data.")
            return None, None
        
        # Read Excel file to get sheet names
        try:
            uploaded_file.seek(0)
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if not sheet_names:
                st.error("❌ No sheets found in the Excel file.")
                return None, None
            
            # If multiple sheets, let user select
            if len(sheet_names) > 1:
                st.info(f"📊 Found {len(sheet_names)} sheets in the Excel file")
                selected_sheet = st.selectbox(
                    "Select a sheet to import:",
                    sheet_names,
                    help="Choose which sheet you want to analyze"
                )
            else:
                selected_sheet = sheet_names[0]
                st.info(f"📊 Loading sheet: '{selected_sheet}'")
            
            # Read the selected sheet
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            
        except Exception as e:
            st.error(f"❌ Error reading Excel file: {str(e)}")
            return None, None
        
        # Validate the DataFrame
        validation_results = validate_csv_structure(df)  # Reuse the same validation
        
        # Display validation results
        if not display_validation_results(validation_results):
            return None, None
        
        # Clean column names
        original_columns = df.columns.tolist()
        df.columns = [str(col).strip().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        
        # Show column renaming info if any changes were made
        renamed_cols = [(orig, new) for orig, new in zip(original_columns, df.columns) if str(orig) != str(new)]
        if renamed_cols:
            with st.expander("Column names were cleaned"):
                for orig, new in renamed_cols[:10]:
                    st.text(f"{orig} → {new}")
                if len(renamed_cols) > 10:
                    st.text(f"... and {len(renamed_cols) - 10} more")
        
        # Return dataframe and selected sheet name
        return df, selected_sheet
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None, None

def get_excel_info(uploaded_file):
    """Get information about Excel file without fully loading it"""
    try:
        uploaded_file.seek(0)
        excel_file = pd.ExcelFile(uploaded_file)
        
        info = {
            'sheets': excel_file.sheet_names,
            'num_sheets': len(excel_file.sheet_names)
        }
        
        # Get basic info about each sheet
        sheet_info = {}
        for sheet in excel_file.sheet_names[:5]:  # Limit to first 5 sheets
            try:
                # Read only first few rows to get column info
                df_preview = pd.read_excel(uploaded_file, sheet_name=sheet, nrows=5)
                sheet_info[sheet] = {
                    'columns': len(df_preview.columns),
                    'preview_rows': len(df_preview)
                }
            except:
                sheet_info[sheet] = {'error': 'Could not read sheet'}
        
        info['sheet_details'] = sheet_info
        return info
        
    except Exception as e:
        return None

def suggest_table_name(filename):
    """Suggest a table name based on filename"""
    name = os.path.splitext(filename)[0]
    name = name.lower().replace(' ', '_').replace('-', '_')
    name = re.sub(r'[^a-z0-9_]', '', name)
    if name and not name[0].isalpha():
        name = 'table_' + name
    return name or 'uploaded_data'

def load_csv_to_database(df, table_name, db_path):
    """Load DataFrame into SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            st.info(f"Replaced existing table '{table_name}'")
        else:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            st.success(f"Created new table '{table_name}'")
        
        conn.commit()
        conn.close()
        
        # Add to uploaded tables list
        if table_name not in st.session_state.uploaded_tables:
            st.session_state.uploaded_tables.append(table_name)
        
        return table_name
    except Exception as e:
        st.error(f"Error loading data to database: {str(e)}")
        return None

def get_table_info(db_path):
    """Get detailed information about all tables"""
    tables_info = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get sample data to determine column types
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()
            
            if sample_data:
                df_sample = pd.DataFrame(sample_data, columns=[col[1] for col in columns])
                
                # Categorize columns
                numeric_cols = []
                text_cols = []
                date_cols = []
                
                for col in df_sample.columns:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df_sample[col])
                        numeric_cols.append(col)
                    except:
                        # Try to convert to datetime
                        try:
                            pd.to_datetime(df_sample[col])
                            date_cols.append(col)
                        except:
                            text_cols.append(col)
                
                tables_info[table_name] = {
                    'columns': [col[1] for col in columns],
                    'numeric_columns': numeric_cols,
                    'text_columns': text_cols,
                    'date_columns': date_cols,
                    'row_count': len(sample_data)
                }
        
        conn.close()
        return tables_info
        
    except Exception as e:
        return {}

# Generate simple questions that don't require AI
def generate_simple_questions(df, table_name):
    """Generate simple questions that can be answered without AI"""
    questions = []
    
    # Basic questions
    questions.append(f"Show all data from {table_name}")
    questions.append(f"How many records are in {table_name}?")
    questions.append(f"Show columns in {table_name}")
    
    # Get column names
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        questions.append(f"Show minimum and maximum {numeric_cols[0]} from {table_name}")
    
    if text_cols:
        questions.append(f"Show unique {text_cols[0]} values from {table_name}")
    
    return questions

# Optimized generate_csv_questions
def generate_csv_questions(df, table_name):
    """Generate questions - both simple and complex"""
    simple_questions = generate_simple_questions(df, table_name)
    
    # Add a few complex questions if we have the columns for it
    complex_questions = []
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols and text_cols:
        complex_questions.append(f"What is the average {numeric_cols[0]} by {text_cols[0]}?")
        complex_questions.append(f"Show top 5 {text_cols[0]} by {numeric_cols[0]}")
    
    if numeric_cols:
        complex_questions.append(f"Calculate running total of {numeric_cols[0]}")
    
    if len(st.session_state.uploaded_tables) > 1:
        other_table = [t for t in st.session_state.uploaded_tables if t != table_name][0]
        complex_questions.append(f"Compare {table_name} with {other_table}")
    
    return simple_questions + complex_questions

# Format numeric values
def format_numeric_value(value):
    """Format numeric values with appropriate decimal places"""
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return round(value, 2)
        else:
            return value
    return value

# Extract SQL queries from text
def extract_sql_queries(text):
    """Extract SQL queries from the analysis text"""
    patterns = [
        r"```sql\n(.*?)\n```",
        r"```\n(SELECT.*?)\n```",
        r"Query:\s*(SELECT.*?)(?:\n|$)",
        r"(SELECT\s+.*?(?:;|$))",
        r"(WITH\s+.*?SELECT\s+.*?(?:;|$))",
    ]
    
    queries = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        for match in matches:
            query = match.strip()
            if query and not any(query == q for q in queries):
                queries.append(query)
    
    return queries

# Create optimized agents based on LLM type
def create_agents(tools, llm):
    """Create agents optimized for the available LLM"""
    if llm is None:
        return None
    
    # Adjust prompts based on LLM type
    if st.session_state.llm_type == "ollama":
        # Optimized for CodeLlama
        sql_specialist = Agent(
            role="SQL Expert",
            goal="Write efficient SQL queries",
            backstory="""You are CodeLlama, specialized in SQL. Focus on:
            - Correct SQL syntax with JOINs, CTEs, window functions
            - Query optimization
            - Clear column aliases
            Always verify table and column names.""",
            verbose=False,
            allow_delegation=False,
            tools=tools,
            llm=llm,
            max_iter=3,
            memory=False
        )
    else:
        # Optimized for GROQ (shorter prompts)
        sql_specialist = Agent(
            role="SQL Expert",
            goal="Write SQL queries",
            backstory="SQL expert. Write efficient queries. Be concise.",
            verbose=False,
            allow_delegation=False,
            tools=tools,
            llm=llm,
            max_iter=2,
            memory=False
        )

    data_analyst = Agent(
        role="Data Analyst",
        goal="Analyze query results",
        backstory="Analyze data for patterns and insights. Be concise.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
        max_iter=1,
        memory=False
    )

    business_consultant = Agent(
        role="Business Consultant",
        goal="Create recommendations",
        backstory="Transform insights into business recommendations. Brief format.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
        max_iter=1,
        memory=False
    )
    
    return [sql_specialist, data_analyst, business_consultant]

# Define tool creation function
def create_tools(db, db_path):
    """Create tools with given database connection"""
    
    @tool("list_tables")
    def list_tables() -> str:
        """List all available tables in the database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            if not tables:
                return "No tables found. Please upload a CSV or Excel file first."
            
            table_names = [t[0] for t in tables]
            result = f"Available tables: {', '.join(table_names)}"
            
            if st.session_state.uploaded_tables:
                result += f"\n\nUploaded in this session: {', '.join(st.session_state.uploaded_tables)}"
            
            return result
        except Exception as e:
            return f"Error listing tables: {str(e)}"

    @tool("tables_schema")
    def tables_schema(tables: str) -> str:
        """Get schema and sample data for specified tables"""
        try:
            conn = sqlite3.connect(db_path)
            result = []
            
            for table in tables.split(','):
                table = table.strip()
                cursor = conn.cursor()
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                result.append(f"\nTable: {table}")
                result.append("Columns:")
                for col in columns:
                    result.append(f"  - {col[1]} ({col[2]})")
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                samples = cursor.fetchall()
                if samples:
                    result.append("Sample data:")
                    for row in samples:
                        result.append(f"  {row}")
            
            conn.close()
            return '\n'.join(result)
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute a SQL query and return results"""
        sql_query = sql_query.strip()
        if sql_query.startswith('"') and sql_query.endswith('"'):
            sql_query = sql_query[1:-1]
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Execute query
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Store results
            st.session_state.query_results = {
                'query': sql_query,
                'dataframe': df,
                'text_result': df.to_string() if len(df) < 20 else f"{df.head(20).to_string()}\n... ({len(df)} total rows)",
                'timestamp': datetime.now()
            }
            
            # Return formatted result
            if len(df) == 0:
                return "Query returned no results"
            elif len(df) == 1 and len(df.columns) == 1:
                return f"Result: {df.iloc[0, 0]}"
            elif len(df) <= 10:
                return df.to_string()
            else:
                return f"{df.head(10).to_string()}\n... showing 10 of {len(df)} rows"
                
        except Exception as e:
            return f"Error executing query: {str(e)}"

    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """Validate SQL query before execution"""
        sql_upper = sql_query.upper()
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return f"Warning: Query contains {keyword} statement"
        
        # Check if tables exist
        table_pattern = r'FROM\s+(\w+)'
        tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
        
        for table in tables:
            if table.lower() not in [t.lower() for t in st.session_state.uploaded_tables]:
                return f"Warning: Table '{table}' may not exist"
        
        return "Query validation passed"
    
    return [list_tables, tables_schema, execute_sql, check_sql]

# Simple query executor for basic queries
def execute_simple_query(question: str, table_name: str, db_path: str) -> Dict:
    """Execute simple queries without using LLM"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Pattern matching for simple queries
        question_lower = question.lower()
        
        if "show all" in question_lower or "show me all" in question_lower:
            query = f"SELECT * FROM {table_name} LIMIT 100"
        elif "count" in question_lower or "how many" in question_lower:
            query = f"SELECT COUNT(*) as total_count FROM {table_name}"
        elif "columns" in question_lower or "schema" in question_lower:
            query = f"PRAGMA table_info({table_name})"
        else:
            return None
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        result_text = f"Query executed: {query}\n\nResults:\n{df.to_string()}"
        
        # Store in session state
        st.session_state.query_results = {
            'query': query,
            'dataframe': df,
            'text_result': result_text,
            'timestamp': datetime.now()
        }
        
        return {
            "success": True,
            "question": question,
            "timestamp": datetime.now(),
            "result": result_text,
            "queries": [query]
        }
        
    except Exception as e:
        return None

# Optimized run_analysis function
def run_analysis(question, agents):
    """Run analysis with rate limiting and error handling"""
    st.session_state.executed_queries = []
    
    # Check if using GROQ and enforce rate limits
    if st.session_state.llm_type == "groq":
        if st.session_state.last_api_call:
            time_since_last = (datetime.now() - st.session_state.last_api_call).seconds
            if time_since_last < 4:
                wait_time = 4 - time_since_last
                with st.spinner(f"Rate limiting: waiting {wait_time} seconds..."):
                    time.sleep(wait_time)
    
    try:
        sql_specialist, data_analyst, business_consultant = agents
        
        # Adjust task descriptions based on LLM type
        if st.session_state.llm_type == "ollama":
            # More detailed for Ollama
            data_extraction = Task(
                description=f"""
                Extract data to answer: {question}
                
                Tables available: {', '.join(st.session_state.uploaded_tables)}
                Write and execute appropriate SQL query.
                Use JOINs, CTEs, or window functions if needed.
                """,
                expected_output="SQL query and results",
                agent=sql_specialist
            )
        else:
            # Concise for GROQ
            data_extraction = Task(
                description=f"Query: {question[:100]}\nTables: {', '.join(st.session_state.uploaded_tables[:5])}\nWrite SQL.",
                expected_output="SQL query",
                agent=sql_specialist
            )
        
        data_analysis = Task(
            description=f"Analyze results for: {question[:50]}",
            expected_output="Key insights",
            agent=data_analyst
        )
        
        business_report = Task(
            description="Create brief recommendations",
            expected_output="Summary and recommendations",
            agent=business_consultant
        )
        
        # Create crew
        crew = Crew(
            agents=[sql_specialist, data_analyst, business_consultant],
            tasks=[data_extraction, data_analysis, business_report],
            process=Process.sequential,
            verbose=0,
            memory=False,
            embedder={
                "provider": "literal",
                "config": {"api_key": "dummy"}
            }
        )
        
        # Show appropriate warnings
        if st.session_state.llm_type == "groq" and st.session_state.api_calls > 10:
            st.warning("⚠️ High API usage. Consider switching to Ollama for unlimited queries.")
        
        # Execute with progress tracking
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        try:
            status_placeholder.text("🚀 Starting analysis...")
            progress_bar.progress(0.2)
            time.sleep(0.5)
            
            status_placeholder.text("📝 Generating SQL query...")
            progress_bar.progress(0.4)
            
            result = crew.kickoff()
            
            status_placeholder.text("✅ Analysis complete!")
            progress_bar.progress(1.0)
            time.sleep(0.5)
            
        finally:
            progress_bar.empty()
            status_placeholder.empty()
        
        # Extract queries
        result_str = str(result)
        queries = extract_sql_queries(result_str)
        
        return {
            "success": True,
            "question": question,
            "timestamp": datetime.now(),
            "result": result_str,
            "queries": queries
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "rate_limit" in error_msg or "429" in error_msg:
            st.error("🚫 Rate limit exceeded!")
            st.info("""
            **Solutions:**
            1. Install Ollama for unlimited queries (recommended)
            2. Wait 60 seconds before trying again
            3. Use simpler questions
            """)
            
            # Show cooldown timer
            progress_bar = st.progress(0)
            for i in range(60):
                progress_bar.progress((i + 1) / 60)
                time.sleep(1)
            progress_bar.empty()
            
            st.success("✅ You can try again now!")
        else:
            st.error(f"❌ Analysis failed: {str(e)}")
            with st.expander("Error details"):
                st.code(str(e))
        
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "timestamp": datetime.now()
        }

# Main UI
def main():
    # Initialize system
    llm = initialize_llm()
    
    # Get database connection
    db, db_path = get_database_connection()
    
    if db is None or db_path is None:
        st.error("Failed to initialize database")
        st.stop()
        return
    
    # Store LLM in session state
    st.session_state.llm = llm
    
    # Header with model status
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🤖 CrewAI SQL Agent")
        st.markdown("### Upload CSV or Excel files and ask questions about your data")
    with col2:
        if st.session_state.llm_type == "ollama":
            st.markdown('<div class="model-status ollama-status">🟢 Ollama (Unlimited)</div>', unsafe_allow_html=True)
        elif st.session_state.llm_type == "groq":
            st.metric("API Calls", st.session_state.api_calls)
    with col3:
        if st.session_state.llm_type == "groq":
            calls_remaining = max(0, 15 - (st.session_state.api_calls % 15))
            st.metric("Calls Left", calls_remaining)
    
    # Show rate limit warning for GROQ
    if st.session_state.llm_type == "groq" and st.session_state.api_calls > 10:
        st.markdown("""
        <div class="rate-limit-warning">
        ⚠️ <strong>Rate Limit Warning:</strong> Consider installing Ollama for unlimited queries.
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Data Upload", 
        "🔍 Analysis", 
        "📊 Full Report", 
        "💾 Query & Results"
    ])
    
    with tab1:
        st.header("Upload and Preview Data Files")
        
        # Upload section with better styling
        with st.container():
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a CSV or Excel file to analyze. The file will be loaded into a temporary database for this session."
            )
        
        if uploaded_file is not None:
            # Show file info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 File Name", uploaded_file.name)
            with col2:
                file_size = uploaded_file.size
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                st.metric("📦 File Size", size_str)
            with col3:
                file_extension = uploaded_file.name.split('.')[-1].upper()
                st.metric("📋 File Type", file_extension)
            with col4:
                # For Excel files, show number of sheets
                if file_extension.lower() in ['xlsx', 'xls']:
                    excel_info = get_excel_info(uploaded_file)
                    if excel_info:
                        st.metric("📑 Sheets", excel_info['num_sheets'])
            
            # Add a file size warning for very large files
            if file_size > 50 * 1024 * 1024:  # 50MB
                st.warning("⚠️ Large file detected. Processing might take longer.")
            
            # Special handling for Excel files with multiple sheets
            sheet_name = None
            if file_extension.lower() in ['xlsx', 'xls']:
                excel_info = get_excel_info(uploaded_file)
                if excel_info and excel_info['num_sheets'] > 1:
                    st.markdown("### 📑 Excel File Details")
                    
                    # Show sheet information
                    with st.expander("View sheet information", expanded=True):
                        for sheet, details in excel_info['sheet_details'].items():
                            if 'error' not in details:
                                st.info(f"**{sheet}**: {details['columns']} columns")
                            else:
                                st.warning(f"**{sheet}**: Could not read")
            
            # Process the file
            if file_extension.lower() == 'csv':
                df = process_csv_upload(uploaded_file)
                sheet_name = None
            elif file_extension.lower() in ['xlsx', 'xls']:
                df, sheet_name = process_excel_upload(uploaded_file)
            else:
                st.error(f"❌ Unsupported file type: {file_extension}")
                df = None
            
            if df is not None:
                # Success message
                if sheet_name:
                    st.success(f"✅ Successfully loaded '{sheet_name}' from {uploaded_file.name}")
                else:
                    st.success(f"✅ Successfully loaded {uploaded_file.name}")
                
                # Preview section
                st.markdown("### 📊 Data Preview")
                
                # Basic info in a colored container
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    with col4:
                        null_count = df.isnull().sum().sum()
                        st.metric("Missing Values", f"{null_count:,}")
                
                # Create tabs for different views
                preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📋 Sample Data", "🔍 Data Types", "📊 Quick Stats"])
                
                with preview_tab1:
                    st.markdown("#### First 10 Rows")
                    st.dataframe(
                        df.head(10), 
                        use_container_width=True,
                        height=400
                    )
                
                with preview_tab2:
                    st.markdown("#### Column Information")
                    # Create detailed column info
                    col_info = []
                    for col in df.columns:
                        col_data = {
                            'Column Name': col,
                            'Data Type': str(df[col].dtype),
                            'Non-Null Count': df[col].count(),
                            'Null Count': df[col].isnull().sum(),
                            'Null %': f"{(df[col].isnull().sum() / len(df) * 100):.1f}%",
                            'Unique Values': df[col].nunique(),
                            'Unique %': f"{(df[col].nunique() / len(df) * 100):.1f}%"
                        }
                        
                        # Add sample values for each column
                        if df[col].dtype in ['int64', 'float64']:
                            col_data['Min'] = df[col].min()
                            col_data['Max'] = df[col].max()
                            col_data['Mean'] = round(df[col].mean(), 2) if df[col].dtype == 'float64' else df[col].mean()
                        else:
                            # For text columns, show most common values
                            top_values = df[col].value_counts().head(3)
                            if len(top_values) > 0:
                                col_data['Top Values'] = ', '.join([f"{val} ({count})" for val, count in top_values.items()])
                        
                        col_info.append(col_data)
                    
                    col_info_df = pd.DataFrame(col_info)
                    
                    # Display based on data type
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    text_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_cols:
                        st.markdown("##### 🔢 Numeric Columns")
                        numeric_info = col_info_df[col_info_df['Column Name'].isin(numeric_cols)]
                        numeric_display_cols = ['Column Name', 'Data Type', 'Non-Null Count', 'Null %', 'Min', 'Max', 'Mean']
                        st.dataframe(
                            numeric_info[numeric_display_cols], 
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    if text_cols:
                        st.markdown("##### 📝 Text Columns")
                        text_info = col_info_df[col_info_df['Column Name'].isin(text_cols)]
                        text_display_cols = ['Column Name', 'Data Type', 'Non-Null Count', 'Null %', 'Unique Values']
                        st.dataframe(
                            text_info[text_display_cols], 
                            use_container_width=True,
                            hide_index=True
                        )
                
                with preview_tab3:
                    st.markdown("#### Quick Statistics")
                    
                    # Numeric statistics
                    numeric_df = df.select_dtypes(include=['int64', 'float64'])
                    if not numeric_df.empty:
                        st.markdown("##### 🔢 Numeric Column Statistics")
                        st.dataframe(
                            numeric_df.describe().round(2),
                            use_container_width=True
                        )
                    
                    # Categorical statistics
                    text_df = df.select_dtypes(include=['object'])
                    if not text_df.empty and len(text_df.columns) > 0:
                        st.markdown("##### 📝 Text Column Statistics")
                        
                        # Show value counts for first few text columns
                        num_cols_to_show = min(3, len(text_df.columns))
                        cols = st.columns(num_cols_to_show)
                        
                        for i, col in enumerate(text_df.columns[:num_cols_to_show]):
                            with cols[i]:
                                st.markdown(f"**{col}**")
                                value_counts = df[col].value_counts().head(5)
                                for val, count in value_counts.items():
                                    st.text(f"{val}: {count}")
                                if len(df[col].unique()) > 5:
                                    st.text(f"... and {len(df[col].unique()) - 5} more")
                
                # Load to database section
                st.markdown("### 💾 Load to Database")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Suggest table name based on file and sheet
                    if sheet_name:
                        suggested_name = suggest_table_name(f"{uploaded_file.name.split('.')[0]}_{sheet_name}")
                    else:
                        suggested_name = suggest_table_name(uploaded_file.name)
                    
                    table_name = st.text_input(
                        "Table name:", 
                        value=suggested_name,
                        help="Choose a name for this table in the database"
                    )
                with col2:
                    st.write("") # Empty space for alignment
                    st.write("") # Empty space for alignment
                    load_button = st.button("Load to Database", type="primary", use_container_width=True)
                
                if load_button:
                    if table_name:
                        with st.spinner(f"Loading data to table '{table_name}'..."):
                            result = load_csv_to_database(df, table_name, db_path)
                            if result:
                                st.success(f"✅ Successfully loaded data to table '{result}'")
                                
                                # Store preview data
                                st.session_state.csv_preview_data = {
                                    'dataframe': df,
                                    'table_name': result,
                                    'source_file': uploaded_file.name,
                                    'sheet_name': sheet_name
                                }
                                
                                # Generate suggested questions
                                st.markdown("### 💡 Suggested Questions")
                                st.markdown("Click any question below to analyze it:")
                                
                                questions = generate_csv_questions(df, result)
                                
                                cols = st.columns(2)
                                for i, q in enumerate(questions):
                                    with cols[i % 2]:
                                        if st.button(q, key=f"csv_q_{i}", use_container_width=True):
                                            st.session_state.selected_question = q
                                            st.info("Switch to the Analysis tab to run this question!")
                    else:
                        st.error("Please provide a table name")
        
        # Show uploaded tables
        if st.session_state.uploaded_tables:
            st.markdown("### 📁 Uploaded Tables in This Session")
            for i, table in enumerate(st.session_state.uploaded_tables):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    # Show more info about the table
                    table_info = f"**{table}**"
                    if 'csv_preview_data' in st.session_state and st.session_state.csv_preview_data:
                        if st.session_state.csv_preview_data.get('table_name') == table:
                            source = st.session_state.csv_preview_data.get('source_file', 'Unknown')
                            sheet = st.session_state.csv_preview_data.get('sheet_name')
                            if sheet:
                                table_info += f" (from {source} - {sheet})"
                            else:
                                table_info += f" (from {source})"
                    st.info(table_info)
                with col2:
                    # Show row count
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        conn.close()
                        st.metric("Rows", f"{count:,}")
                    except:
                        pass
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{table}"):
                        try:
                            conn = sqlite3.connect(db_path)
                            conn.execute(f"DROP TABLE IF EXISTS {table}")
                            conn.commit()
                            conn.close()
                            st.session_state.uploaded_tables.remove(table)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing table: {str(e)}")
    
    with tab2:
        if not st.session_state.uploaded_tables:
            st.warning("⚠️ Please upload a CSV or Excel file first to start analyzing data.")
            st.stop()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("🤔 Ask Your Question")
            
            # Show LLM status
            if llm is None:
                st.error("❌ No LLM available. Please set up Ollama or add GROQ_API_KEY")
                st.stop()
            
            # Simple query options (only if no LLM or for basic queries)
            use_simple = st.checkbox("Use simple mode (no AI, instant results)", 
                                   help="For basic queries like 'show all data' or 'count rows'")
            
            if st.session_state.csv_preview_data:
                source_info = f"Analyzing table: **{st.session_state.csv_preview_data['table_name']}**"
                if st.session_state.csv_preview_data.get('sheet_name'):
                    source_info += f" (Sheet: {st.session_state.csv_preview_data['sheet_name']})"
                st.info(source_info)
            
            question = st.text_area(
                "Enter your question about the data:",
                value=st.session_state.get('selected_question', ''),
                height=100,
                placeholder="e.g., What are the top 10 products by revenue? Show me the trend over time."
            )
            
            col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
            with col1_1:
                analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
            with col1_2:
                if st.button("🧹 Clear", key="clear_question", use_container_width=True):
                    st.session_state.selected_question = ""
                    st.rerun()
            
            if analyze_button:
                if question:
                    if use_simple and st.session_state.uploaded_tables:
                        # Try simple query first
                        result = execute_simple_query(
                            question, 
                            st.session_state.uploaded_tables[0], 
                            db_path
                        )
                        if result:
                            st.session_state.current_analysis = result
                            st.success("✅ Query executed successfully!")
                        else:
                            st.warning("This question is too complex for simple mode. Using AI analysis...")
                            use_simple = False
                    
                    if not use_simple:
                        # Use CrewAI analysis
                        db, _ = get_database_connection()
                        if db and llm:
                            # Create tools and agents
                            tools = create_tools(db, db_path)
                            agents = create_agents(tools, llm)
                            
                            if agents:
                                with st.spinner("🤖 AI agents analyzing your data..."):
                                    result = run_analysis(question, agents)
                                    st.session_state.current_analysis = result
                                    
                                    if result['success']:
                                        st.success("✅ Analysis complete!")
                                    else:
                                        st.error("❌ Analysis failed. Check the error message below.")
                else:
                    st.warning("Please enter a question to analyze.")
        
        with col2:
            st.header("📊 Available Tables & Schema")
            
            if st.session_state.uploaded_tables:
                # Get detailed table info
                tables_info = get_table_info(db_path)
                
                for table in st.session_state.uploaded_tables:
                    with st.expander(f"📁 {table}", expanded=True):
                        if table in tables_info:
                            info = tables_info[table]
                            
                            # Show column categories
                            if info['numeric_columns']:
                                st.markdown("**🔢 Numeric columns:**")
                                st.text(", ".join(info['numeric_columns']))
                            
                            if info['text_columns']:
                                st.markdown("**📝 Text columns:**")
                                st.text(", ".join(info['text_columns']))
                            
                            if info['date_columns']:
                                st.markdown("**📅 Date columns:**")
                                st.text(", ".join(info['date_columns']))
                            
                            st.markdown(f"**Total columns:** {len(info['columns'])}")
                        else:
                            st.text("Schema information not available")
            else:
                st.info("No tables uploaded yet. Upload a CSV or Excel file to get started.")
        
        # Display analysis results
        if st.session_state.current_analysis:
            st.markdown("---")
            st.header("📈 Analysis Results")
            
            analysis = st.session_state.current_analysis
            
            # Show the question
            st.markdown(f"**Question:** {analysis['question']}")
            
            if analysis.get('success', False):
                # Show SQL queries if any
                if analysis.get('queries'):
                    with st.expander("🔍 SQL Queries Used", expanded=True):
                        for i, query in enumerate(analysis['queries']):
                            st.code(query, language='sql')
                            
                            # Add copy button
                            if st.button(f"📋 Copy Query {i+1}", key=f"copy_query_{i}"):
                                st.write("Query copied to clipboard!")
                                st.session_state.query_history.append({
                                    'query': query,
                                    'timestamp': datetime.now(),
                                    'question': analysis['question']
                                })
                
                # Show the analysis result
                with st.expander("📊 Analysis Report", expanded=True):
                    st.markdown(analysis.get('result', 'No result available'))
                
                # Show data results if available
                if st.session_state.query_results:
                    with st.expander("📋 Query Results Data", expanded=True):
                        df_result = st.session_state.query_results.get('dataframe')
                        if df_result is not None and not df_result.empty:
                            st.dataframe(df_result, use_container_width=True)
                            
                            # Download button for results
                            csv = df_result.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
            else:
                # Show error
                st.error(f"❌ Error: {analysis.get('error', 'Unknown error')}")
    
    with tab3:
        st.header("📊 Full Analysis Report")
        
        if st.session_state.current_analysis and st.session_state.current_analysis.get('success'):
            analysis = st.session_state.current_analysis
            
            # Report header
            st.markdown(f"### 📅 Report Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Question:** {analysis['question']}")
            
            # Executive Summary
            st.markdown("### 📋 Executive Summary")
            result_text = analysis.get('result', '')
            
            # Extract key findings (look for bullet points or numbered lists)
            lines = result_text.split('\n')
            summary_lines = []
            for line in lines:
                if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                    summary_lines.append(line.strip())
            
            if summary_lines:
                for line in summary_lines[:5]:  # Show first 5 key points
                    st.markdown(f"- {line.lstrip('•-*123. ')}")
            else:
                st.info("Run an analysis to see the full report here.")
            
            # Detailed Analysis
            st.markdown("### 🔍 Detailed Analysis")
            st.markdown(result_text)
            
            # Visualizations (if applicable)
            if st.session_state.query_results and st.session_state.query_results.get('dataframe') is not None:
                df_result = st.session_state.query_results['dataframe']
                
                if not df_result.empty:
                    st.markdown("### 📊 Data Visualizations")
                    
                    # Auto-generate appropriate visualizations
                    numeric_cols = df_result.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    text_cols = df_result.select_dtypes(include=['object']).columns.tolist()
                    
                    if len(df_result) <= 20 and numeric_cols:
                        # Bar chart for small datasets
                        if text_cols:
                            fig = px.bar(df_result, x=text_cols[0], y=numeric_cols[0], 
                                       title=f"{numeric_cols[0]} by {text_cols[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif len(numeric_cols) >= 2:
                        # Scatter plot for numeric relationships
                        fig = px.scatter(df_result, x=numeric_cols[0], y=numeric_cols[1],
                                       title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.markdown("### 📋 Result Data")
                    st.dataframe(df_result, use_container_width=True)
            
            # Export options
            st.markdown("### 💾 Export Report")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as text
                report_text = f"""
SQL Analysis Report
Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Question: {analysis['question']}

Analysis Results:
{analysis.get('result', '')}

SQL Queries Used:
{chr(10).join(analysis.get('queries', []))}
"""
                st.download_button(
                    label="📄 Download as Text",
                    data=report_text,
                    file_name=f"analysis_report_{analysis['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                    mime='text/plain'
                )
            
            with col2:
                # Export as JSON
                report_json = {
                    'timestamp': analysis['timestamp'].isoformat(),
                    'question': analysis['question'],
                    'result': analysis.get('result', ''),
                    'queries': analysis.get('queries', []),
                    'success': analysis.get('success', False)
                }
                st.download_button(
                    label="📊 Download as JSON",
                    data=json.dumps(report_json, indent=2),
                    file_name=f"analysis_report_{analysis['timestamp'].strftime('%Y%m%d_%H%M%S')}.json",
                    mime='application/json'
                )
        else:
            st.info("No analysis report available. Run an analysis in the Analysis tab to generate a report.")
    
    with tab4:
        st.header("💾 Query History & Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📜 Query History")
            
            if st.session_state.query_history:
                for i, item in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                    with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['timestamp'].strftime('%H:%M:%S')}"):
                        st.code(item['query'], language='sql')
                        if 'question' in item:
                            st.markdown(f"**Question:** {item['question']}")
                        
                        # Re-run button
                        if st.button(f"🔄 Re-run", key=f"rerun_{i}"):
                            try:
                                conn = sqlite3.connect(db_path)
                                df = pd.read_sql_query(item['query'], conn)
                                conn.close()
                                
                                st.session_state.query_results = {
                                    'query': item['query'],
                                    'dataframe': df,
                                    'text_result': df.to_string(),
                                    'timestamp': datetime.now()
                                }
                                st.success("Query re-executed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info("No queries executed yet.")
        
        with col2:
            st.subheader("📊 Current Results")
            
            if st.session_state.query_results:
                results = st.session_state.query_results
                
                st.markdown(f"**Last Updated:** {results['timestamp'].strftime('%H:%M:%S')}")
                st.code(results['query'], language='sql')
                
                df = results.get('dataframe')
                if df is not None and not df.empty:
                    st.markdown(f"**Result:** {len(df)} rows × {len(df.columns)} columns")
                    
                    # Quick stats
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numeric_cols) > 0:
                        st.markdown("**Quick Stats:**")
                        for col in numeric_cols[:3]:  # First 3 numeric columns
                            col_stats = f"- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}"
                            st.text(col_stats)
                    
                    # Show sample data
                    st.markdown("**Sample Data:**")
                    st.dataframe(df.head(10), use_container_width=True)
                else:
                    st.info("No data in results.")
            else:
                st.info("No query results to display.")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ Information")
        
        # Session info
        st.markdown("### 📊 Session Info")
        st.info(f"Session ID: {st.session_state.session_id}")
        st.info(f"Tables loaded: {len(st.session_state.uploaded_tables)}")
        
        if st.session_state.llm_type == "groq":
            st.warning(f"API calls made: {st.session_state.api_calls}")
        
        # Help section
        st.markdown("### 🤝 How to Use")
        st.markdown("""
        1. **Upload Data**: Upload CSV or Excel files
        2. **Load to Database**: Give your table a name
        3. **Ask Questions**: Use natural language
        4. **View Results**: Check analysis and SQL queries
        """)
        
        # Example questions
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "Show me the top 10 rows",
            "What is the average value by category?",
            "Find all records where amount > 1000",
            "Calculate the monthly trend",
            "Compare this year vs last year"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.selected_question = q
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            if st.button("🗑️ Clear All Data"):
                if st.button("⚠️ Confirm Clear All", key="confirm_clear"):
                    # Clear database
                    conn = sqlite3.connect(db_path)
                    for table in st.session_state.uploaded_tables:
                        conn.execute(f"DROP TABLE IF EXISTS {table}")
                    conn.commit()
                    conn.close()
                    
                    # Reset session state
                    st.session_state.uploaded_tables = []
                    st.session_state.current_analysis = None
                    st.session_state.query_results = None
                    st.session_state.csv_preview_data = None
                    st.session_state.query_history = []
                    st.session_state.analysis_history = []
                    
                    st.success("All data cleared!")
                    st.rerun()
            
            # Database download
            if st.session_state.uploaded_tables:
                with open(db_path, 'rb') as f:
                    st.download_button(
                        label="💾 Download Database",
                        data=f.read(),
                        file_name=f"session_{st.session_state.session_id}.db",
                        mime='application/octet-stream'
                    )
        
        # Footer
        st.markdown("---")
        st.markdown("Made with ❤️ using CrewAI & Streamlit")

# Run the main function
if __name__ == "__main__":
    main()