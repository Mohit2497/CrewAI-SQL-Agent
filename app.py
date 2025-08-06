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

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CrewAI SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Create a temporary database file for this session
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    st.session_state.temp_db_path = temp_file.name
    temp_file.close()

# SQL Query Builder Helpers
class AdvancedQueryBuilder:
    """Helper class for building complex SQL queries"""
    
    @staticmethod
    def detect_join_columns(df1, df2, table1_name, table2_name):
        """Detect potential join columns between two dataframes"""
        join_suggestions = []
        
        # Check for exact column name matches
        common_columns = set(df1.columns) & set(df2.columns)
        for col in common_columns:
            if col.lower().endswith('_id') or col.lower() == 'id':
                join_suggestions.append({
                    'confidence': 'high',
                    'column1': col,
                    'column2': col,
                    'suggested_query': f"JOIN {table2_name} ON {table1_name}.{col} = {table2_name}.{col}"
                })
        
        # Check for foreign key patterns
        for col1 in df1.columns:
            if col1.lower().endswith('_id'):
                base_name = col1[:-3]  # Remove '_id'
                if 'id' in df2.columns:
                    join_suggestions.append({
                        'confidence': 'medium',
                        'column1': col1,
                        'column2': 'id',
                        'suggested_query': f"JOIN {table2_name} ON {table1_name}.{col1} = {table2_name}.id"
                    })
        
        return join_suggestions
    
    @staticmethod
    def build_window_function(func_type, column, partition_by=None, order_by=None, window_frame=None):
        """Build window function syntax"""
        base_func = f"{func_type}({column})"
        over_clause_parts = []
        
        if partition_by:
            over_clause_parts.append(f"PARTITION BY {partition_by}")
        if order_by:
            over_clause_parts.append(f"ORDER BY {order_by}")
        if window_frame:
            over_clause_parts.append(window_frame)
        
        if over_clause_parts:
            return f"{base_func} OVER ({' '.join(over_clause_parts)})"
        else:
            return f"{base_func} OVER ()"
    
    @staticmethod
    def build_cte(cte_name, query):
        """Build Common Table Expression"""
        return f"WITH {cte_name} AS (\n  {query}\n)"

# Complex Query Examples
COMPLEX_QUERY_EXAMPLES = {
    "running_total": """-- Running total example
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM sales
ORDER BY date;""",
    
    "ranking": """-- Rank within groups
SELECT 
    category,
    product_name,
    sales_amount,
    RANK() OVER (PARTITION BY category ORDER BY sales_amount DESC) as rank_in_category
FROM products
ORDER BY category, rank_in_category;""",
    
    "moving_average": """-- 7-day moving average
SELECT 
    date,
    daily_sales,
    AVG(daily_sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7_days
FROM daily_metrics;""",
    
    "year_over_year": """-- Year-over-year comparison using CTEs
WITH current_year AS (
    SELECT 
        STRFTIME('%m', date) as month,
        SUM(amount) as cy_sales
    FROM sales
    WHERE STRFTIME('%Y', date) = '2024'
    GROUP BY month
),
previous_year AS (
    SELECT 
        STRFTIME('%m', date) as month,
        SUM(amount) as py_sales
    FROM sales
    WHERE STRFTIME('%Y', date) = '2023'
    GROUP BY month
)
SELECT 
    cy.month,
    cy.cy_sales,
    py.py_sales,
    ROUND((cy.cy_sales - py.py_sales) * 100.0 / py.py_sales, 2) as yoy_growth_pct
FROM current_year cy
LEFT JOIN previous_year py ON cy.month = py.month
ORDER BY cy.month;""",
    
    "top_n_per_group": """-- Top 3 products per category
WITH ranked_products AS (
    SELECT 
        category,
        product_name,
        total_sales,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY total_sales DESC) as rn
    FROM product_sales
)
SELECT 
    category,
    product_name,
    total_sales
FROM ranked_products
WHERE rn <= 3
ORDER BY category, rn;""",
    
    "multi_table_analysis": """-- Complex multi-table analysis with CTEs
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.region,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(o.total_amount) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name, c.region
),
regional_averages AS (
    SELECT 
        region,
        AVG(total_spent) as avg_regional_spent
    FROM customer_metrics
    GROUP BY region
)
SELECT 
    cm.customer_name,
    cm.region,
    cm.order_count,
    cm.total_spent,
    ra.avg_regional_spent,
    ROUND(cm.total_spent - ra.avg_regional_spent, 2) as diff_from_regional_avg
FROM customer_metrics cm
JOIN regional_averages ra ON cm.region = ra.region
ORDER BY cm.total_spent DESC;"""
}

# Initialize LLM (cached)
@st.cache_resource
def initialize_llm():
    """Initialize the language model"""
    try:
        # For Hugging Face Spaces, secrets are in environment variables
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            st.error("Please set GROQ_API_KEY in Space Settings → Repository secrets")
            return None
            
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=api_key
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

# Initialize Database
def get_database_connection():
    """Get current database connection"""
    try:
        # Use the session's temporary database
        db_path = st.session_state.temp_db_path
        
        # Create SQLDatabase object
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        return db, db_path
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None, None

# Enhanced CSV Processing Functions with Validation
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
    """Get detailed information about all tables for generating smart questions"""
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

def generate_csv_questions(df, table_name):
    """Generate relevant questions based on CSV data including advanced SQL operations"""
    questions = []
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Basic questions
    questions.append(f"Show me all data from {table_name}")
    questions.append(f"How many records are in {table_name}?")
    
    # Window function questions
    if numeric_cols:
        col = numeric_cols[0]
        questions.append(f"Show running total of {col} in {table_name}")
        questions.append(f"Calculate 7-day moving average of {col}")
        
        if text_cols:
            questions.append(f"Rank records by {col} within each {text_cols[0]}")
            questions.append(f"Show top 3 {text_cols[0]} by {col} using window functions")
    
    # Advanced aggregation questions
    if text_cols and numeric_cols:
        questions.append(f"Show {numeric_cols[0]} by {text_cols[0]} with subtotals and grand total")
        questions.append(f"Calculate percentage of total {numeric_cols[0]} for each {text_cols[0]}")
    
    # Date-based advanced questions
    if date_cols and numeric_cols:
        questions.append(f"Compare {numeric_cols[0]} month-over-month from {table_name}")
        questions.append(f"Show year-to-date cumulative {numeric_cols[0]}")
    
    # CTE questions
    if len(numeric_cols) > 1:
        questions.append(f"Identify outliers and analyze their characteristics in {table_name}")
    
    # Multi-table questions (if other tables exist)
    if st.session_state.uploaded_tables and len(st.session_state.uploaded_tables) > 1:
        other_tables = [t for t in st.session_state.uploaded_tables if t != table_name]
        questions.append(f"Join {table_name} with {other_tables[0]} to show combined insights")
    
    return questions

def create_advanced_sql_agent_instructions():
    """Create instructions for SQL agents to handle advanced queries"""
    return """
    You are an expert SQL developer who specializes in complex queries. You can handle:
    
    1. **JOIN Operations**:
       - INNER JOIN: When user wants matching records from both tables
       - LEFT JOIN: When user wants all records from first table
       - FULL OUTER JOIN: When user wants all records from both tables
       - Self JOIN: When comparing records within same table
       - Multiple JOINs: Connect 3 or more tables
    
    2. **Window Functions**:
       - ROW_NUMBER(): For ranking without ties
       - RANK(): For ranking with ties
       - DENSE_RANK(): For ranking with consecutive ranks
       - Running totals: SUM() OVER (ORDER BY ...)
       - Moving averages: AVG() OVER (ROWS BETWEEN n PRECEDING AND CURRENT ROW)
       - LAG/LEAD: Access previous/next row values
       - PERCENT_RANK(): Calculate percentile ranking
    
    3. **Advanced Aggregations**:
       - GROUP BY with ROLLUP: For subtotals and grand totals
       - GROUP BY with multiple columns: Multi-dimensional analysis
       - HAVING clause: Filter aggregated results
       - Conditional aggregation: SUM(CASE WHEN ... THEN ... END)
       - Statistical functions: STDDEV, VARIANCE
    
    4. **Common Table Expressions (CTEs)**:
       - Simple CTEs: WITH temp_table AS (SELECT ...)
       - Multiple CTEs: Chain multiple temporary results
       - Use CTEs to break down complex queries into readable steps
    
    When user asks questions like:
    - "Compare X across Y" → Use window functions
    - "Show relationship between tables" → Use appropriate JOIN
    - "Calculate running total" → Use SUM() OVER()
    - "Show top N per group" → Use ROW_NUMBER() with PARTITION BY
    - "Multi-step analysis" → Use CTEs
    
    Always:
    - Check available tables first
    - Verify column names exist
    - Use appropriate JOIN conditions
    - Consider performance (use LIMIT for testing)
    - Explain what the query does
    """

def monitor_query_complexity(query):
    """Check if query is complex and might need optimization"""
    complexity_indicators = {
        'joins': query.upper().count('JOIN'),
        'subqueries': query.upper().count('SELECT') - 1,
        'window_functions': query.upper().count('OVER'),
        'ctes': query.upper().count('WITH'),
        'aggregations': sum(query.upper().count(func) for func in ['GROUP BY', 'SUM', 'AVG', 'COUNT'])
    }
    
    total_complexity = sum(complexity_indicators.values())
    
    if total_complexity > 5:
        return "high", complexity_indicators
    elif total_complexity > 2:
        return "medium", complexity_indicators
    else:
        return "low", complexity_indicators

# Define tool creation function
def create_tools(db, db_path):
    """Create tools with given database connection"""
    
    @tool("list_tables")
    def list_tables() -> str:
        """List all available tables in the database"""
        try:
            tool = ListSQLDatabaseTool(db=db)
            result = tool.invoke("")
            
            if not result or "No tables" in result:
                return "No tables found. Please upload a CSV or Excel file first."
            
            if st.session_state.uploaded_tables:
                result += f"\n\nUploaded tables in this session: {', '.join(st.session_state.uploaded_tables)}"
            
            return result
        except Exception as e:
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
            except Exception as e2:
                return f"Error listing tables: {str(e2)}"

    @tool("tables_schema")
    def tables_schema(tables: str) -> str:
        """Get schema and sample data for specified tables"""
        try:
            tool = InfoSQLDatabaseTool(db=db)
            return tool.invoke(tables)
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute a SQL query and return results"""
        sql_query = sql_query.strip()
        if sql_query.startswith('"') and sql_query.endswith('"'):
            sql_query = sql_query[1:-1]
        
        try:
            # Monitor query complexity
            complexity, indicators = monitor_query_complexity(sql_query)
            if complexity == "high":
                st.warning(f"⚠️ Complex query detected: {indicators}")
            
            if 'executed_queries' not in st.session_state:
                st.session_state.executed_queries = []
            st.session_state.executed_queries.append(sql_query)
            
            tool = QuerySQLDataBaseTool(db=db)
            result = tool.invoke(sql_query)
            
            # Also get as dataframe
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                
                st.session_state.query_results = {
                    'query': sql_query,
                    'dataframe': df,
                    'text_result': result,
                    'timestamp': datetime.now(),
                    'complexity': complexity
                }
            except:
                st.session_state.query_results = {
                    'query': sql_query,
                    'dataframe': None,
                    'text_result': result,
                    'timestamp': datetime.now(),
                    'complexity': complexity
                }
            
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """Validate SQL query before execution"""
        try:
            llm = st.session_state.get('llm', None)
            if llm:
                tool = QuerySQLCheckerTool(db=db, llm=llm)
                return tool.invoke({"query": sql_query})
            else:
                return "Query validation passed"
        except:
            return "Query validation passed"
    
    return [list_tables, tables_schema, execute_sql, check_sql]

# Create agents function
def create_agents(tools, llm):
    """Create agents with given tools and LLM"""
    
    # Get the advanced SQL instructions
    advanced_instructions = create_advanced_sql_agent_instructions()
    
    sql_specialist = Agent(
        role="Senior Database Administrator & SQL Expert",
        goal="Extract accurate data using both simple and complex SQL queries including JOINs, window functions, CTEs, and advanced aggregations",
        backstory=f"""You are a database expert working with user-uploaded data files. 
        You excel at writing complex SQL queries.
        
        {advanced_instructions}
        
        Always check what tables are available first. Remember that all data comes from 
        CSV or Excel files uploaded by the user. When multiple tables exist, consider if a JOIN
        would provide better insights.""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
        max_iter=5  # Allow more iterations for complex queries
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze query results and extract meaningful insights from both simple and complex SQL query results",
        backstory="""You are a data analysis expert who specializes in interpreting results
        from complex SQL queries including window functions, CTEs, and multi-table JOINs.
        You can explain sophisticated analytical results in business terms.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


    return [sql_specialist, data_analyst]

# Format numeric value helper
def format_numeric_value(value):
    """Format numeric values with appropriate decimal places"""
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            # Round to 2 decimal places if it's a float
            return round(value, 2)
        else:
            return value
    return value

# Main UI
def main():
    # Initialize system
    llm = initialize_llm()
    
    # Check LLM initialization
    if llm is None:
        st.error("Failed to initialize LLM. Please check your GROQ_API_KEY")
        st.stop()
        return
    
    # Store LLM in session state
    st.session_state.llm = llm
    
    # Get database connection
    db, db_path = get_database_connection()
    
    if db is None or db_path is None:
        st.error("Failed to initialize database")
        st.stop()
        return
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🤖 CrewAI SQL Agent")
        st.markdown("### Upload CSV or Excel files and ask questions about your data")
    
    # Check if any data is uploaded
    if not st.session_state.uploaded_tables:
        st.info("👋 Welcome! Start by uploading a CSV or Excel file in the 'Data Upload' tab.")
    else:
        st.success(f"✅ Ready to analyze {len(st.session_state.uploaded_tables)} table(s)")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 Data Upload", 
        "🔍 Analysis",  
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
            
            # Advanced SQL Features Section
            with st.expander("🚀 Advanced SQL Features", expanded=False):
                st.markdown("""
                **Available Advanced Features:**
                - **JOINs**: Combine multiple tables
                - **Window Functions**: Rankings, running totals, moving averages
                - **CTEs**: Multi-step analysis with temporary results
                - **Advanced Aggregations**: Subtotals, rollups, conditional sums
                
                **Example Questions:**
                - "Show running total of sales by date"
                - "Rank customers by purchase amount within each region"
                - "Calculate 30-day moving average of orders"
                - "Compare this month's performance to last month"
                - "Join customer and order tables to show full picture"
                """)
                
                # Show available tables for JOIN operations
                if len(st.session_state.uploaded_tables) > 1:
                    st.markdown("**Available tables for JOIN operations:**")
                    for table in st.session_state.uploaded_tables:
                        st.write(f"- {table}")
                    
                    # Auto-detect JOIN possibilities
                    if st.button("🔍 Detect JOIN possibilities"):
                        tables_info = get_table_info(db_path)
                        if len(tables_info) >= 2:
                            st.markdown("**Possible JOINs detected:**")
                            # Simple detection based on column names
                            for t1 in tables_info:
                                for t2 in tables_info:
                                    if t1 != t2:
                                        common_cols = set(tables_info[t1]['columns']) & set(tables_info[t2]['columns'])
                                        if common_cols:
                                            st.info(f"Tables '{t1}' and '{t2}' share columns: {', '.join(common_cols)}")
                
                # Show query templates
                st.markdown("**📝 Query Templates:**")
                template_options = {
                    "Running Total": "running_total",
                    "Ranking": "ranking",
                    "Moving Average": "moving_average",
                    "Year-over-Year": "year_over_year",
                    "Top N per Group": "top_n_per_group",
                    "Multi-table Analysis": "multi_table_analysis"
                }
                
                selected_template = st.selectbox(
                    "Choose a template:",
                    options=["None"] + list(template_options.keys()),
                    help="Select a query template to see example SQL"
                )
                
                if selected_template != "None":
                    template_key = template_options[selected_template]
                    template_code = COMPLEX_QUERY_EXAMPLES.get(template_key, "")
                    st.code(template_code, language='sql')
                    
                    if st.button("Use this template", key="use_template"):
                        st.info("💡 Adapt this template to your data by updating table and column names in your question!")
            
            col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
            with col1_1:
                analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
            with col1_2:
                if st.button("🧹 Clear", key="clear_question", use_container_width=True):
                    st.session_state.selected_question = ""
                    st.rerun()
            
            if analyze_button:
                if question:
                    with st.spinner("Running analysis..."):
                        # Get fresh database connection
                        db, db_path = get_database_connection()
                        
                        if db and db_path:
                            # Create tools and agents
                            tools = create_tools(db, db_path)
                            agents = create_agents(tools, llm)
                            
                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            stages = [
                                (0.2, "🔌 Connecting to database..."),
                                (0.4, "🔍 Analyzing tables..."),
                                (0.6, "✍️ Writing SQL queries..."),
                                (0.8, "📊 Analyzing results..."),
                            ]
                            
                            for progress, status in stages:
                                progress_bar.progress(progress)
                                status_text.text(status)
                                time.sleep(0.5)
                            
                            # Run analysis
                            result = run_analysis(question, agents)
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Analysis complete!")
                            time.sleep(0.5)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.session_state.current_analysis = result
                            st.session_state.analysis_history.append(result)
                            
                            if 'selected_question' in st.session_state:
                                del st.session_state.selected_question
                        else:
                            st.error("Failed to connect to database")
                else:
                    st.warning("⚠️ Please enter a question to analyze.")
    
    with tab3:
        st.header("💾 Query & Results")
        
        # Current query and results FIRST (moved up)
        if st.session_state.current_analysis and st.session_state.current_analysis.get('success', False):
            result = st.session_state.current_analysis
            
            # Create a prominent section for current query
            with st.container():
                st.markdown("### 📝 Current Query")
                
                # Display query in a more visible way
                if result.get('queries'):
                    for i, query in enumerate(result['queries'], 1):
                        if len(result['queries']) > 1:
                            st.markdown(f"**Query {i}:**")
                        
                        # Check query complexity
                        complexity, indicators = monitor_query_complexity(query)
                        if complexity == "high":
                            st.markdown(f"<div class='complex-query-box'>⚡ Complex Query (JOINs: {indicators['joins']}, Window Functions: {indicators['window_functions']}, CTEs: {indicators['ctes']})</div>", unsafe_allow_html=True)
                        
                        # Display query in a code block for better visibility
                        st.code(query, language='sql')
                elif st.session_state.query_results:
                    # Display current query in code block
                    st.code(st.session_state.query_results['query'], language='sql')
                
                # Show results in a compact way
                st.markdown("### 📊 Results")
                if st.session_state.query_results and st.session_state.query_results['dataframe'] is not None:
                    df = st.session_state.query_results['dataframe']
                    
                    # Add to history (hidden logic)
                    current_query = st.session_state.query_results['query']
                    current_timestamp = st.session_state.query_results['timestamp']
                    
                    existing_index = None
                    for i, item in enumerate(st.session_state.query_history):
                        if item['query'] == current_query:
                            existing_index = i
                            break
                    
                    if existing_index is not None:
                        st.session_state.query_history[existing_index] = {
                            'timestamp': current_timestamp,
                            'query': current_query,
                            'dataframe': df.copy(),
                            'result_text': st.session_state.query_results.get('text_result', ''),
                            'complexity': st.session_state.query_results.get('complexity', 'low')
                        }
                    else:
                        st.session_state.query_history.append({
                            'timestamp': current_timestamp,
                            'query': current_query,
                            'dataframe': df.copy(),
                            'result_text': st.session_state.query_results.get('text_result', ''),
                            'complexity': st.session_state.query_results.get('complexity', 'low')
                        })
                    
                    # Display results based on size
                    if len(df) == 1 and len(df.columns) == 1:
                        # Single value - show as metric with formatting
                        value = df.iloc[0, 0]
                        formatted_value = format_numeric_value(value)
                        st.metric("Result", formatted_value)
                    elif len(df) <= 10:
                        # Small dataset - show full table with formatted numbers
                        # Format numeric columns
                        df_display = df.copy()
                        for col in df_display.select_dtypes(include=['float64']).columns:
                            df_display[col] = df_display[col].apply(format_numeric_value)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    else:
                        # Large dataset - show preview with info
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.info(f"Showing first 10 rows of {len(df)} total rows")
                        with col2:
                            st.metric("Total Rows", f"{len(df):,}")
                        
                        # Show preview with formatted numbers
                        df_display = df.head(10).copy()
                        for col in df_display.select_dtypes(include=['float64']).columns:
                            df_display[col] = df_display[col].apply(format_numeric_value)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("🧹 Clear", key="clear_current_query", use_container_width=True):
                            st.session_state.current_analysis = None
                            st.session_state.query_results = None
                            st.rerun()
                        
                elif st.session_state.query_results:
                    # Text results only
                    st.text_area("Results", st.session_state.query_results['text_result'], height=150, disabled=True)
            
            st.divider()
        
        # Query History section (moved down)
        with st.container():
            # History header with controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("### 📜 Query History")
            with col2:
                dedupe = st.checkbox("Deduplicate", value=True, help="Keep only the latest execution of each unique query")
            with col3:
                if st.session_state.query_history:
                    if st.button("🗑️ Clear History", use_container_width=True):
                        st.session_state.query_history = []
                        st.rerun()
            
            # Show query history
            if st.session_state.query_history:
                # Prepare history data
                if dedupe:
                    seen_queries = {}
                    for item in st.session_state.query_history:
                        query_key = item['query']
                        if query_key not in seen_queries or item['timestamp'] > seen_queries[query_key]['timestamp']:
                            seen_queries[query_key] = item
                    history_items = list(seen_queries.values())
                    history_items.sort(key=lambda x: x['timestamp'], reverse=True)
                else:
                    history_items = st.session_state.query_history[::-1]
                
                # Create history dataframe with full queries
                history_data = []
                for item in history_items:
                    result_str = "N/A"
                    if 'dataframe' in item and item['dataframe'] is not None:
                        df = item['dataframe']
                        if len(df) == 1 and len(df.columns) == 1:
                            # Format single value results
                            value = df.iloc[0, 0]
                            result_str = str(format_numeric_value(value))
                        elif len(df) <= 3:
                            result_str = df.to_string(index=False, max_cols=3)
                            if len(result_str) > 100:
                                result_str = result_str[:100] + "..."
                        else:
                            result_str = f"{len(df)} rows × {len(df.columns)} columns"
                    elif 'result_text' in item:
                        result_str = str(item['result_text'])[:100] + "..." if len(str(item['result_text'])) > 100 else str(item['result_text'])
                    
                    # Add complexity indicator
                    complexity = item.get('complexity', 'low')
                    complexity_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(complexity, "🟢")
                    
                    history_data.append({
                        'Time': item['timestamp'].strftime('%H:%M:%S'),
                        'Complexity': complexity_icon,
                        'Query': item['query'],  # Full query, no truncation
                        'Result': result_str
                    })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    
                    # Compact history display
                    with st.expander(f"📋 View History ({len(history_data)} queries)", expanded=True):
                        # Info about deduplication
                        if dedupe and len(history_items) < len(st.session_state.query_history):
                            st.caption(f"Showing {len(history_items)} unique queries from {len(st.session_state.query_history)} total executions")
                        
                        st.caption("🟢 Simple | 🟡 Medium | 🔴 Complex")
                        
                        # Show history table with full queries
                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            height=250,
                            column_config={
                                "Time": st.column_config.TextColumn("Time", width="small"),
                                "Complexity": st.column_config.TextColumn("", width="small"),
                                "Query": st.column_config.TextColumn("Query", width="large"),
                                "Result": st.column_config.TextColumn("Result", width="medium")
                            },
                            hide_index=True
                        )
                        
                        # Download option
                        if st.button("💾 Download Full History", key="download_history_btn"):
                            full_history = []
                            for item in st.session_state.query_history:
                                history_item = {
                                    'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                                    'Query': item['query'],
                                    'Complexity': item.get('complexity', 'low')
                                }
                                
                                if 'dataframe' in item and item['dataframe'] is not None:
                                    history_item['Result'] = item['dataframe'].to_dict('records')
                                elif 'result_text' in item:
                                    history_item['Result'] = item['result_text']
                                else:
                                    history_item['Result'] = 'N/A'
                                
                                full_history.append(history_item)
                            
                            history_json = json.dumps(full_history, indent=2, default=str)
                            st.download_button(
                                label="💾 Save as JSON",
                                data=history_json,
                                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            else:
                st.info("ℹ️ No queries executed yet. Run an analysis to see queries here.")
        
        # If no current analysis
        if not (st.session_state.current_analysis and st.session_state.current_analysis.get('success', False)):
            st.info("ℹ️ Run an analysis to see current query and results")

# Helper functions
def get_database_stats(db_path):
    """Get database statistics"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        stats = {
            'tables': [],
            'total_tables': len(tables),
            'total_rows': 0
        }
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            stats['tables'].append({'name': table_name, 'rows': count})
            stats['total_rows'] += count
        
        conn.close()
        return stats
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")
        return None

def extract_sql_queries(text):
    """Extract SQL queries from the analysis text"""
    patterns = [
        r"```sql\n(.*?)\n```",
        r"```\n(SELECT.*?)\n```",
        r"Query:\s*(SELECT.*?)(?:\n|$)",
        r"(SELECT\s+.*?(?:;|$))",
        r"(WITH\s+.*?SELECT\s+.*?(?:;|$))",  # Added pattern for CTEs
    ]
    
    queries = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        for match in matches:
            query = match.strip()
            if query and not any(query == q for q in queries):
                queries.append(query)
    
    return queries

def run_analysis(question, agents):
    """Run analysis with provided agents"""
    st.session_state.executed_queries = []
    
    try:
        sql_specialist, data_analyst = agents
        
        # Create tasks
        data_extraction = Task(
            description=f"""
            Extract data to answer: {question}
            
            IMPORTANT: 
            1. First list ALL available tables to see what's in the database
            2. The data comes from CSV or Excel files uploaded by the user
            3. Available tables: {', '.join(st.session_state.uploaded_tables)}
            4. Make sure to query the correct table that contains the data for this question
            5. Show the exact SQL query you're executing
            6. Use advanced SQL features when appropriate (JOINs, Window Functions, CTEs)
            7. For complex questions, break them down using CTEs
            """,
            expected_output="SQL query results with the query used",
            agent=sql_specialist
        )
        
        data_analysis = Task(
            description=f"""
            Analyze the data for: {question}
            
            Provide insights based on the actual data returned from the uploaded file.
            Look for patterns, trends, and interesting findings.
            If the query used advanced SQL features, explain the insights they revealed.
            """,
            expected_output="Detailed analysis with insights",
            agent=data_analyst
        )
        
        
        # Create and run crew
        crew = Crew(
            agents=[sql_specialist, data_analyst],
            tasks=[data_extraction, data_analysis],
            process=Process.sequential,
            verbose=0
        )
        
        result = crew.kickoff()
        queries = extract_sql_queries(result)
        
        return {
            "success": True,
            "question": question,
            "timestamp": datetime.now(),
            "result": result,
            "queries": queries
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "timestamp": datetime.now()
        }

# Sidebar
with st.sidebar:
    st.header("🤖 CrewAI SQL Agent")
    
    st.divider()
    
    # Session info
    st.subheader("📊 Session Info")
    st.info(f"Session: {st.session_state.session_id}")
    
    # Clear all data button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        try:
            # Get fresh connection
            db, db_path = get_database_connection()
            if db_path:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                # Drop each table
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
                
                conn.commit()
                conn.close()
            
            # Clear session state
            st.session_state.uploaded_tables = []
            st.session_state.csv_preview_data = None
            st.session_state.current_analysis = None
            st.session_state.query_results = None
            st.session_state.query_history = []
            st.success("✅ All data cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
    
    st.divider()
    
    # Show uploaded tables
    if st.session_state.uploaded_tables:
        st.subheader("📁 Uploaded Tables")
        for table in st.session_state.uploaded_tables[-5:]:
            st.success(f"📊 {table}")
        if len(st.session_state.uploaded_tables) > 5:
            st.info(f"...and {len(st.session_state.uploaded_tables) - 5} more")
    else:
        st.info("📭 No tables uploaded yet")
    
    st.divider()
    
    # SQL Features
    st.subheader("🚀 SQL Features")
    st.markdown("""
    **Supported Operations:**
    - ✅ Basic queries
    - ✅ JOINs (all types)
    - ✅ Window functions
    - ✅ CTEs
    - ✅ Advanced aggregations
    - ✅ Date/time operations
    """)
    
    st.divider()
    
    # Help section
    with st.expander("❓ Help", expanded=False):
        st.markdown("""
        **How to use:**
        
        1. **Upload File**: Go to Data Upload tab
        2. **Choose Format**: CSV or Excel files supported
        3. **Select Sheet**: For Excel files with multiple sheets
        4. **Preview**: Check your data before loading
        5. **Load**: Give your table a meaningful name
        6. **Ask Questions**: Use suggested questions or write your own
        7. **View Results**: Check the Query & Results tab
        
        **Advanced SQL Tips:**
        - Use "running total" for cumulative calculations
        - Use "rank by" for ordering within groups
        - Use "join" to combine multiple tables
        - Use "moving average" for trend analysis
        - Use "compare" for period-over-period analysis
        
        **Tips:**
        - Supports CSV and Excel (XLS, XLSX) files
        - Empty files are automatically rejected
        - Column names are cleaned automatically
        - All data is temporary (session-based)
        - Use descriptive table names
        - Try the suggested questions first
        - You can upload multiple files
        
        **Common Questions:**
        - Show trends over time
        - Find top/bottom performers
        - Calculate averages and totals
        - Group by categories
        - Compare different segments
        - Rank within groups
        - Calculate running totals
        - Join multiple tables
        """)
    
    st.divider()
    
    # Footer
    st.caption("Made with ❤️ using Streamlit & CrewAI")

# Clean up temporary database on app close
import atexit

def cleanup():
    """Clean up temporary database file"""
    if 'temp_db_path' in st.session_state:
        try:
            os.unlink(st.session_state.temp_db_path)
        except:
            pass

atexit.register(cleanup)

# Run the app
if __name__ == "__main__":
    main()