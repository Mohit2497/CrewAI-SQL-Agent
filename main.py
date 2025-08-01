# streamlit_app_clean.py
"""
Clean Streamlit interface for SQL Agent System
Works exclusively with uploaded CSV files
"""

# CRITICAL: SQLite fix for Streamlit Cloud - MUST be at the very top
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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
import concurrent.futures

# Import CrewAI components
from crewai import Agent, Crew, Process, Task
from langchain.tools import Tool

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
    page_icon="📊",
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

# Initialize LLM (cached)
@st.cache_resource
def initialize_llm():
    """Initialize the language model"""
    try:
        # Try Streamlit secrets first (for deployment)
        api_key = None
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            # Fall back to environment variable (for local development)
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            st.error("Please set GROQ_API_KEY in Streamlit secrets or .env file")
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

# CSV Processing Functions
def process_csv_upload(uploaded_file):
    """Process uploaded CSV file and return DataFrame"""
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("Could not read CSV file with any common encoding")
            return None
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
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

def generate_csv_questions(df, table_name):
    """Generate relevant questions based on CSV data"""
    questions = []
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Basic questions
    questions.append(f"Show me all data from {table_name}")
    questions.append(f"How many records are in {table_name}?")
    
    if numeric_cols:
        col = numeric_cols[0]
        questions.append(f"What is the average {col} in {table_name}?")
        questions.append(f"What are the minimum and maximum values of {col} in {table_name}?")
        if len(numeric_cols) > 1:
            questions.append(f"Show me the correlation between {numeric_cols[0]} and {numeric_cols[1]}")
        
    if text_cols:
        col = text_cols[0]
        questions.append(f"What are the unique values of {col} in {table_name}?")
        questions.append(f"What is the most common {col} in {table_name}?")
        
    if text_cols and numeric_cols:
        questions.append(f"What is the average {numeric_cols[0]} by {text_cols[0]} in {table_name}?")
        questions.append(f"Show me the top 10 {text_cols[0]} by {numeric_cols[0]} from {table_name}")
    
    if date_cols and numeric_cols:
        questions.append(f"Show me the trend of {numeric_cols[0]} over {date_cols[0]}")
    
    return questions

# Define tool creation function
def create_tools(db, db_path):
    """Create tools with given database connection"""
    
    # Create simple wrapper functions
    def list_tables_func(query: str = "") -> str:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            if not tables:
                return "No tables found. Please upload a CSV file first."
            
            table_names = [t[0] for t in tables]
            return f"Available tables: {', '.join(table_names)}"
        except Exception as e:
            return f"Error listing tables: {str(e)}"
    
    def get_table_schema_func(table_name: str) -> str:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_data = cursor.fetchall()
            
            conn.close()
            
            schema_info = f"Table: {table_name}\n"
            schema_info += "Columns:\n"
            for col in columns:
                schema_info += f"  - {col[1]} ({col[2]})\n"
            
            schema_info += "\nSample data:\n"
            for row in sample_data:
                schema_info += f"  {row}\n"
            
            return schema_info
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
    def execute_sql_func(sql_query: str) -> str:
        try:
            # Clean the query
            sql_query = sql_query.strip()
            if sql_query.startswith('"') and sql_query.endswith('"'):
                sql_query = sql_query[1:-1]
            
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Store in session state
            st.session_state.query_results = {
                'query': sql_query,
                'dataframe': df,
                'text_result': df.to_string(),
                'timestamp': datetime.now()
            }
            
            # Return string representation
            if len(df) == 0:
                return "Query returned no results."
            elif len(df) == 1 and len(df.columns) == 1:
                return f"Result: {df.iloc[0, 0]}"
            elif len(df) > 10:
                return f"Query returned {len(df)} rows. First 10 rows:\n{df.head(10).to_string()}"
            else:
                return df.to_string()
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    # Create tools using Tool class
    list_tables = Tool(
        name="list_tables",
        description="List all available tables in the database",
        func=list_tables_func
    )
    
    get_schema = Tool(
        name="get_table_schema",
        description="Get the schema and sample data for a specific table. Input should be the table name.",
        func=get_table_schema_func
    )
    
    execute_sql = Tool(
        name="execute_sql",
        description="Execute a SQL query and return results. Input should be a valid SQL query string.",
        func=execute_sql_func
    )
    
    return [list_tables, get_schema, execute_sql]

# Create agents function
def create_agents(tools, llm):
    """Create agents with given tools and LLM"""
    
    sql_specialist = Agent(
        role="Senior Database Administrator",
        goal="Extract accurate data from uploaded CSV files using optimized SQL queries",
        backstory="""You are a database expert working with user-uploaded CSV data. 
        Always check what tables are available first, then get the schema, then write SQL queries.
        Be precise with your SQL syntax.""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
        max_iter=3  # Limit iterations
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze query results and extract meaningful insights from CSV data",
        backstory="""You are a data analysis expert who specializes in finding patterns 
        and insights in user-uploaded data. Provide clear, concise analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=3  # Limit iterations
    )

    business_consultant = Agent(
        role="Business Strategy Consultant",
        goal="Transform data insights into actionable business recommendations",
        backstory="""You are a senior business consultant who helps users make 
        data-driven decisions based on their uploaded data. Keep recommendations practical.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=3  # Limit iterations
    )
    
    return [sql_specialist, data_analyst, business_consultant]

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
        st.title("CrewAI SQL Agent")
        st.markdown("### Upload CSV files and ask questions about your data")
    
    # Check if any data is uploaded
    if not st.session_state.uploaded_tables:
        st.info("Welcome! Start by uploading a CSV file in the 'CSV Upload' tab.")
    else:
        st.success(f"Ready to analyze {len(st.session_state.uploaded_tables)} table(s)")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "CSV Upload", 
        "Analysis", 
        "Full Report", 
        "Query & Results"
    ])
    
    with tab1:
        st.header("Upload and Preview CSV Files")
        
        # Upload section with better styling
        with st.container():
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file to analyze. The file will be loaded into a temporary database for this session."
            )
        
        if uploaded_file is not None:
            df = process_csv_upload(uploaded_file)
            
            if df is not None:
                st.success(f"Successfully loaded {uploaded_file.name}")
                
                # Preview section
                st.markdown("### Data Preview")
                
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
                preview_tab1, preview_tab2 = st.tabs(["Sample Data", "Column Details"])
                
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
                            col_data['Top Values'] = ', '.join([f"{val} ({count})" for val, count in top_values.items()])
                        
                        col_info.append(col_data)
                    
                    col_info_df = pd.DataFrame(col_info)
                    
                    # Display based on data type
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    text_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_cols:
                        st.markdown("##### Numeric Columns")
                        numeric_info = col_info_df[col_info_df['Column Name'].isin(numeric_cols)]
                        numeric_display_cols = ['Column Name', 'Data Type', 'Non-Null Count', 'Null %', 'Min', 'Max', 'Mean']
                        st.dataframe(
                            numeric_info[numeric_display_cols], 
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    if text_cols:
                        st.markdown("##### Text Columns")
                        text_info = col_info_df[col_info_df['Column Name'].isin(text_cols)]
                        text_display_cols = ['Column Name', 'Data Type', 'Non-Null Count', 'Null %', 'Unique Values']
                        st.dataframe(
                            text_info[text_display_cols], 
                            use_container_width=True,
                            hide_index=True
                        )
                
                # Load to database section
                st.markdown("### Load to Database")
                
                col1, col2 = st.columns([3, 1])
                with col1:
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
                                st.success(f"Successfully loaded data to table '{result}'")
                                
                                # Store preview data
                                st.session_state.csv_preview_data = {
                                    'dataframe': df,
                                    'table_name': result
                                }
                                
                                # Generate suggested questions
                                st.markdown("### Suggested Questions")
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
            st.markdown("### Uploaded Tables in This Session")
            for i, table in enumerate(st.session_state.uploaded_tables):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"**{table}**")
                with col2:
                    if st.button("Remove", key=f"remove_{table}"):
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
            st.warning("Please upload a CSV file first to start analyzing data.")
            st.stop()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("Ask Your Question")
            
            if st.session_state.csv_preview_data:
                st.info(f"Analyzing table: **{st.session_state.csv_preview_data['table_name']}**")
            
            question = st.text_area(
                "Enter your question about the data:",
                value=st.session_state.get('selected_question', ''),
                height=100,
                placeholder="e.g., What are the top 10 products by revenue? Show me the trend over time."
            )
            
            col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
            with col1_1:
                analyze_button = st.button("Analyze", type="primary", use_container_width=True)
            with col1_2:
                if st.button("Clear", key="clear_question", use_container_width=True):
                    st.session_state.selected_question = ""
                    st.rerun()
            
            if analyze_button:
                if question:
                    with st.spinner("Running analysis..."):
                        try:
                            # Get fresh database connection
                            db, db_path = get_database_connection()
                            
                            if db and db_path:
                                # Create tools and agents
                                tools = create_tools(db, db_path)
                                agents = create_agents(tools, llm)
                                
                                # Show progress
                                progress_text = st.empty()
                                progress_text.text("Starting analysis...")
                                
                                # Run analysis with timeout
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(run_analysis, question, agents)
                                    try:
                                        result = future.result(timeout=60)  # 60 second timeout
                                    except concurrent.futures.TimeoutError:
                                        st.error("Analysis timed out after 60 seconds. Please try a simpler question.")
                                        result = {
                                            "success": False,
                                            "error": "Timeout",
                                            "question": question,
                                            "timestamp": datetime.now()
                                        }
                                
                                progress_text.empty()
                                
                                if result.get('success'):
                                    st.session_state.current_analysis = result
                                    st.session_state.analysis_history.append(result)
                                    st.success("Analysis completed successfully!")
                                else:
                                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                                
                                if 'selected_question' in st.session_state:
                                    del st.session_state.selected_question
                            else:
                                st.error("Failed to connect to database")
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("Please enter a question to analyze.")
    
    with tab3:
        if st.session_state.current_analysis and st.session_state.current_analysis.get('success', False):
            result = st.session_state.current_analysis
            
            st.header("Analysis Report")
            st.markdown(f"**Question:** {result['question']}")
            st.markdown(f"**Generated:** {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("---")
            
            report_sections = result['result'].split('\n\n')
            for section in report_sections:
                if section.strip():
                    if any(header in section for header in ['Executive Summary', 'Key Findings', 'Recommendations', 'Next Steps']):
                        st.subheader(section.split('\n')[0])
                        remaining = '\n'.join(section.split('\n')[1:])
                        if remaining.strip():
                            st.markdown(remaining)
                    else:
                        st.markdown(section)
            
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="Download Full Report",
                    data=result['result'],
                    file_name=f"analysis_{result['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("Run an analysis to see the full report")
    
    with tab4:
        st.header("Query & Results")
        
        # Current query and results FIRST (moved up)
        if st.session_state.current_analysis and st.session_state.current_analysis.get('success', False):
            result = st.session_state.current_analysis
            
            # Create a prominent section for current query
            with st.container():
                st.markdown("### Current Query")
                
                # Display query in a more visible way
                if result.get('queries'):
                    for i, query in enumerate(result['queries'], 1):
                        if len(result['queries']) > 1:
                            st.markdown(f"**Query {i}:**")
                        
                        # Display query in a code block for better visibility
                        st.code(query, language='sql')
                elif st.session_state.query_results:
                    # Display current query in code block
                    st.code(st.session_state.query_results['query'], language='sql')
                
                # Show results in a compact way
                st.markdown("### Results")
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
                            'result_text': st.session_state.query_results.get('text_result', '')
                        }
                    else:
                        st.session_state.query_history.append({
                            'timestamp': current_timestamp,
                            'query': current_query,
                            'dataframe': df.copy(),
                            'result_text': st.session_state.query_results.get('text_result', '')
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
                            label="Download CSV",
                            data=csv,
                            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("Clear", key="clear_current_query", use_container_width=True):
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
                st.markdown("### Query History")
            with col2:
                dedupe = st.checkbox("Deduplicate", value=True, help="Keep only the latest execution of each unique query")
            with col3:
                if st.session_state.query_history:
                    if st.button("Clear History", use_container_width=True):
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
                    
                    history_data.append({
                        'Time': item['timestamp'].strftime('%H:%M:%S'),
                        'Query': item['query'],  # Full query, no truncation
                        'Result': result_str
                    })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    
                    # Compact history display
                    with st.expander(f"View History ({len(history_data)} queries)", expanded=True):
                        # Info about deduplication
                        if dedupe and len(history_items) < len(st.session_state.query_history):
                            st.caption(f"Showing {len(history_items)} unique queries from {len(st.session_state.query_history)} total executions")
                        
                        # Show history table with full queries
                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            height=250,
                            column_config={
                                "Time": st.column_config.TextColumn("Time", width="small"),
                                "Query": st.column_config.TextColumn("Query", width="large"),
                                "Result": st.column_config.TextColumn("Result", width="medium")
                            },
                            hide_index=True
                        )
                        
                        # Download option
                        if st.button("Download Full History", key="download_history_btn"):
                            full_history = []
                            for item in st.session_state.query_history:
                                history_item = {
                                    'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                                    'Query': item['query']
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
                                label="Save as JSON",
                                data=history_json,
                                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            else:
                st.info("No queries executed yet. Run an analysis to see queries here.")
        
        # If no current analysis
        if not (st.session_state.current_analysis and st.session_state.current_analysis.get('success', False)):
            st.info("Run an analysis to see current query and results")

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
        sql_specialist, data_analyst, business_consultant = agents
        
        # Create tasks
        data_extraction = Task(
            description=f"""
            Extract data to answer: {question}
            
            IMPORTANT: 
            1. First list ALL available tables to see what's in the database
            2. The data comes from CSV files uploaded by the user
            3. Available tables: {', '.join(st.session_state.uploaded_tables)}
            4. Make sure to query the correct table that contains the data for this question
            5. Show the exact SQL query you're executing
            """,
            expected_output="SQL query results with the query used",
            agent=sql_specialist
        )
        
        data_analysis = Task(
            description=f"""
            Analyze the data for: {question}
            
            Provide insights based on the actual data returned from the CSV file.
            Look for patterns, trends, and interesting findings.
            """,
            expected_output="Detailed analysis with insights",
            agent=data_analyst
        )
        
        business_report = Task(
            description=f"""
            Create business recommendations for: {question}
            
            Structure your response with clear sections:
            - Executive Summary
            - Key Findings
            - Recommendations
            - Next Steps
            """,
            expected_output="Structured business report",
            agent=business_consultant
        )
        
        # Create and run crew with fixes
        crew = Crew(
            agents=[sql_specialist, data_analyst, business_consultant],
            tasks=[data_extraction, data_analysis, business_report],
            process=Process.sequential,
            verbose=0,
            memory=False,  # Disable memory to avoid issues
            max_iter=5  # Limit total iterations
        )
        
        result = crew.kickoff()
        
        # Convert result to string if it isn't already
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
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "timestamp": datetime.now()
        }

# Sidebar
with st.sidebar:
    st.header("CrewAI SQL Agent")
    
    st.divider()
    
    # Session info
    st.subheader("Session Info")
    st.info(f"Session: {st.session_state.session_id}")
    
    # Clear all data button
    if st.button("Clear All Data", use_container_width=True):
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
            st.success("All data cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
    
    st.divider()
    
    # Show uploaded tables
    if st.session_state.uploaded_tables:
        st.subheader("Uploaded Tables")
        for table in st.session_state.uploaded_tables[-5:]:
            st.success(f"{table}")
        if len(st.session_state.uploaded_tables) > 5:
            st.info(f"...and {len(st.session_state.uploaded_tables) - 5} more")
    else:
        st.info("No tables uploaded yet")
    
    st.divider()
    
    # Help section
    with st.expander("Help", expanded=False):
        st.markdown("""
        **How to use:**
        
        1. **Upload CSV**: Go to CSV Upload tab
        2. **Preview**: Check your data before loading
        3. **Load**: Give your table a meaningful name
        4. **Ask Questions**: Use suggested questions or write your own
        5. **View Results**: Check the Query & Results tab
        
        **Tips:**
        - This app works exclusively with CSV files
        - All data is temporary (session-based)
        - Use descriptive table names
        - Try the suggested questions first
        - You can upload multiple CSV files
        
        **Common Questions:**
        - Show trends over time
        - Find top/bottom performers
        - Calculate averages and totals
        - Group by categories
        - Compare different segments
        """)
    
    st.divider()
    
    # Footer
    st.caption("Made with Streamlit & CrewAI")

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