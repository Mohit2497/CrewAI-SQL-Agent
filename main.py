# streamlit_app_clean.py
"""
Clean Streamlit interface for SQL Agent System
No emojis, simplified UI
"""

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
    page_title="SQL Agent - AI Database Analysis",
    page_icon="",
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
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Initialize LLM (cached)
@st.cache_resource
def initialize_llm():
    """Initialize the language model"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
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

# Initialize Database (NOT cached to allow refresh)
def get_database_connection():
    """Get current database connection"""
    try:
        db_path = os.getenv("DATABASE_PATH", "company_data.db")
        
        # Create database if it doesn't exist
        if not os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.close()
            st.info(f"Created new database: {db_path}")
        
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
        
        # Force system reinitialization
        st.session_state.system_initialized = False
        
        return table_name
    except Exception as e:
        st.error(f"Error loading data to database: {str(e)}")
        return None

def generate_csv_questions(df, table_name):
    """Generate relevant questions based on CSV data"""
    questions = []
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Basic questions
    questions.append(f"Show me all data from {table_name}")
    questions.append(f"How many records are in {table_name}?")
    
    if numeric_cols:
        questions.append(f"What is the average {numeric_cols[0]} in {table_name}?")
        questions.append(f"What are the minimum and maximum values of {numeric_cols[0]} in {table_name}?")
        
    if text_cols and numeric_cols:
        questions.append(f"What is the average {numeric_cols[0]} by {text_cols[0]} in {table_name}?")
        questions.append(f"Show me the top 10 {text_cols[0]} by {numeric_cols[0]} from {table_name}")
    
    return questions

# Define tool creation function
def create_tools(db, db_path):
    """Create tools with given database connection"""
    
    @tool("list_tables")
    def list_tables() -> str:
        """List all available tables in the database"""
        try:
            tool = ListSQLDatabaseTool(db=db)
            result = tool.invoke("")
            
            if st.session_state.uploaded_tables:
                result += f"\n\nRecently uploaded tables: {', '.join(st.session_state.uploaded_tables)}"
            
            return result
        except Exception as e:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                
                table_names = [t[0] for t in tables]
                result = f"Available tables: {', '.join(table_names)}"
                
                if st.session_state.uploaded_tables:
                    result += f"\n\nRecently uploaded: {', '.join(st.session_state.uploaded_tables)}"
                
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
                    'timestamp': datetime.now()
                }
            except:
                st.session_state.query_results = {
                    'query': sql_query,
                    'dataframe': None,
                    'text_result': result,
                    'timestamp': datetime.now()
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
    
    sql_specialist = Agent(
        role="Senior Database Administrator",
        goal="Extract accurate data from the database using optimized SQL queries",
        backstory="""You are a database expert. Always check what tables are available first,
        especially any recently uploaded tables. Be very careful to query the correct table.""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze query results and extract meaningful insights",
        backstory="""You are a data analysis expert who can work with any type of data.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    business_consultant = Agent(
        role="Business Strategy Consultant",
        goal="Transform data insights into actionable business recommendations",
        backstory="""You are a senior business consultant who helps make data-driven decisions.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return [sql_specialist, data_analyst, business_consultant]

# Main UI
def main():
    # Initialize system
    llm = initialize_llm()
    
    # Check LLM initialization
    if llm is None:
        st.error("Failed to initialize LLM. Please check your GROQ_API_KEY in .env file")
        return
    
    # Store LLM in session state
    st.session_state.llm = llm
    
    # Get database connection
    db, db_path = get_database_connection()
    
    if db is None or db_path is None:
        st.error("Failed to connect to database")
        return
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("CrewAI SQL Agent")
        st.markdown("### AI-Powered Database Analysis with CSV Upload")
    
    # Show status
    st.success("System Ready")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "CSV Upload", 
        "Analysis", 
        "Full Report", 
        "Query & Results"
    ])
    
    with tab1:
        st.header("Upload and Preview CSV Files")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze. The file will be loaded into the database."
        )
        
        if uploaded_file is not None:
            df = process_csv_upload(uploaded_file)
            
            if df is not None:
                st.success(f"Successfully loaded {uploaded_file.name}")
                
                # Preview section
                st.markdown("### Data Preview")
                
                # Basic info
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
                
                # Data preview
                st.markdown("#### First 10 Rows")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Load to database section
                st.markdown("### Load to Database")
                
                suggested_name = suggest_table_name(uploaded_file.name)
                table_name = st.text_input(
                    "Table name:", 
                    value=suggested_name,
                    help="Choose a name for this table in the database"
                )
                
                if st.button("Load to Database", type="primary"):
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
                                questions = generate_csv_questions(df, result)
                                
                                for q in questions:
                                    if st.button(q, key=f"csv_q_{q}"):
                                        st.session_state.selected_question = q
                                        st.info("Switch to the Analysis tab to run this question!")
                    else:
                        st.error("Please provide a table name")
        
        # Show uploaded tables
        if st.session_state.uploaded_tables:
            st.markdown("### Uploaded Tables")
            for table in st.session_state.uploaded_tables:
                st.success(f"{table}")
    
    with tab2:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("Ask Your Question")
            
            if st.session_state.csv_preview_data:
                st.info(f"Table '{st.session_state.csv_preview_data['table_name']}' is ready for queries!")
            
            question = st.text_area(
                "Enter your business question:",
                value=st.session_state.get('selected_question', ''),
                height=100,
                placeholder="e.g., What is the average value by category in my uploaded data?"
            )
            
            if st.button("Analyze", type="primary", use_container_width=True):
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
                                (0.2, "Connecting to database..."),
                                (0.4, "Analyzing tables..."),
                                (0.6, "Writing SQL queries..."),
                                (0.8, "Analyzing results..."),
                                (0.9, "Generating report...")
                            ]
                            
                            for progress, status in stages:
                                progress_bar.progress(progress)
                                status_text.text(status)
                                time.sleep(0.5)
                            
                            # Run analysis
                            result = run_analysis(question, agents)
                            
                            progress_bar.progress(1.0)
                            status_text.text("Analysis complete!")
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
                    st.warning("Please enter a question to analyze.")
        
        with col2:
            st.header("Database Overview")
            
            # Check if there's a recently uploaded CSV
            if st.session_state.csv_preview_data:
                table_name = st.session_state.csv_preview_data['table_name']
                df = st.session_state.csv_preview_data['dataframe']
                
                st.info(f"Showing info for uploaded table: **{table_name}**")
                
                # Show table stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                
                # Show column information
                st.markdown("#### Column Information")
                
                # Create column info dataframe
                col_info = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                
                # Display column info
                st.dataframe(col_info, use_container_width=True, height=300)
                
                # Show sample data
                with st.expander("Sample Data (first 5 rows)"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Button to switch focus to a different table
                if len(st.session_state.uploaded_tables) > 1:
                    st.markdown("#### Switch Table View")
                    selected_table = st.selectbox(
                        "Select a table to view:",
                        st.session_state.uploaded_tables,
                        index=st.session_state.uploaded_tables.index(table_name)
                    )
                    if st.button("View Selected Table"):
                        st.info(f"Viewing {selected_table}")
            else:
                # Show regular database overview if no CSV uploaded
                stats = get_database_stats(db_path)
                if stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Tables", stats['total_tables'])
                    with col2:
                        st.metric("Total Rows", f"{stats['total_rows']:,}")
                    
                    with st.expander("All Tables", expanded=True):
                        for table in stats['tables']:
                            if table['name'] in st.session_state.uploaded_tables:
                                st.success(f"{table['name']}: {table['rows']:,} rows (uploaded)")
                            else:
                                st.info(f"{table['name']}: {table['rows']:,} rows")
    
    with tab3:
        if st.session_state.current_analysis and st.session_state.current_analysis.get('success', False):
            result = st.session_state.current_analysis
            
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
        st.markdown("### Query & Results")
        
        # Add query history section with options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("#### Query History")
        with col2:
            # Deduplication toggle
            dedupe = st.checkbox("Deduplicate", value=True, help="Keep only the latest execution of each unique query")
        with col3:
            if st.session_state.query_history:
                if st.button("Clear History", use_container_width=True):
                    st.session_state.query_history = []
                    st.rerun()
        
        # Show query history
        if st.session_state.query_history:
            # Prepare history data based on deduplication setting
            if dedupe:
                # Create a deduplicated view
                seen_queries = {}
                for item in st.session_state.query_history:
                    # Keep the most recent execution of each query
                    query_key = item['query']
                    if query_key not in seen_queries or item['timestamp'] > seen_queries[query_key]['timestamp']:
                        seen_queries[query_key] = item
                history_items = list(seen_queries.values())
                # Sort by timestamp descending
                history_items.sort(key=lambda x: x['timestamp'], reverse=True)
            else:
                # Show all entries
                history_items = st.session_state.query_history[::-1]  # Most recent first
            
            # Create history dataframe
            history_data = []
            for item in history_items:
                # Format the result for display
                result_str = "N/A"
                if 'dataframe' in item and item['dataframe'] is not None:
                    df = item['dataframe']
                    if len(df) == 1 and len(df.columns) == 1:
                        # Single value result
                        result_str = str(df.iloc[0, 0])
                    elif len(df) <= 3:
                        # Small result set - show as string
                        result_str = df.to_string(index=False, max_cols=3)
                        if len(result_str) > 100:
                            result_str = result_str[:100] + "..."
                    else:
                        # Large result set - show summary
                        result_str = f"{len(df)} rows × {len(df.columns)} columns"
                elif 'result_text' in item:
                    result_str = str(item['result_text'])[:100] + "..." if len(str(item['result_text'])) > 100 else str(item['result_text'])
                
                history_data.append({
                    'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Query': item['query'],
                    'Result': result_str
                })
            
            history_df = pd.DataFrame(history_data)
            
            # Show history in a compact table
            with st.expander("View Query History", expanded=False):
                # Show count info
                if dedupe and len(history_items) < len(st.session_state.query_history):
                    st.info(f"Showing {len(history_items)} unique queries (deduplicated from {len(st.session_state.query_history)} total executions)")
                
                # View options
                view_mode = st.radio(
                    "View mode:",
                    ["Table View", "Detailed View"],
                    horizontal=True,
                    key="history_view_mode"
                )
                
                if view_mode == "Table View":
                    # Use st.dataframe with custom column config for better display
                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        height=300,
                        column_config={
                            "Timestamp": st.column_config.TextColumn(
                                "Timestamp",
                                width="small",
                            ),
                            "Query": st.column_config.TextColumn(
                                "Query",
                                width="large",
                                help="Full SQL query"
                            ),
                            "Result": st.column_config.TextColumn(
                                "Result",
                                width="medium",
                            )
                        }
                    )
                else:
                    # Detailed view - show each query in full
                    for idx, row in history_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text(f"Time: {row['Timestamp']}")
                            with col2:
                                st.text(f"Result: {row['Result']}")
                            
                            # Show full query in a text area
                            st.text_area(
                                "Query:",
                                value=row['Query'],
                                height=80,
                                disabled=True,
                                key=f"history_query_{idx}"
                            )
                            st.markdown("---")
                
                # Download full history
                full_history = []
                for item in st.session_state.query_history:
                    history_item = {
                        'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'Query': item['query']
                    }
                    
                    # Add result based on what's available
                    if 'dataframe' in item and item['dataframe'] is not None:
                        history_item['Result'] = item['dataframe'].to_dict('records')
                    elif 'result_text' in item:
                        history_item['Result'] = item['result_text']
                    else:
                        history_item['Result'] = 'N/A'
                    
                    full_history.append(history_item)
                
                if full_history:
                    history_json = json.dumps(full_history, indent=2)
                    st.download_button(
                        label="Download Query History",
                        data=history_json,
                        file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        st.markdown("---")
        
        # Current query and results
        if st.session_state.current_analysis and st.session_state.current_analysis.get('success', False):
            result = st.session_state.current_analysis
            
            # Show current query
            st.markdown("#### Current Query")
            if result.get('queries'):
                for i, query in enumerate(result['queries'], 1):
                    if len(result['queries']) > 1:
                        st.markdown(f"**Query {i}:**")
                    # Use text_area for full query display with auto-height
                    st.text_area("", value=query, height=None, disabled=True, key=f"query_{i}")
            elif st.session_state.query_results:
                # Use text_area for full query display
                st.text_area("", value=st.session_state.query_results['query'], height=None, disabled=True, key="current_query")
            
            # Show results in a smaller box
            st.markdown("#### Results")
            if st.session_state.query_results and st.session_state.query_results['dataframe'] is not None:
                df = st.session_state.query_results['dataframe']
                
                # Add to history with deduplication
                current_query = st.session_state.query_results['query']
                current_timestamp = st.session_state.query_results['timestamp']
                
                # Check if this query already exists (regardless of timestamp)
                existing_index = None
                for i, item in enumerate(st.session_state.query_history):
                    if item['query'] == current_query:
                        existing_index = i
                        break
                
                if existing_index is not None:
                    # Update existing entry with new timestamp and result
                    st.session_state.query_history[existing_index] = {
                        'timestamp': current_timestamp,
                        'query': current_query,
                        'dataframe': df.copy(),
                        'result_text': st.session_state.query_results.get('text_result', '')
                    }
                else:
                    # Add new entry
                    st.session_state.query_history.append({
                        'timestamp': current_timestamp,
                        'query': current_query,
                        'dataframe': df.copy(),
                        'result_text': st.session_state.query_results.get('text_result', '')
                    })
                
                # Show dataframe in a smaller container
                # Use container with custom height
                st.dataframe(df, use_container_width=True, height=200)
                
                # Download button for current results
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with col2:
                    if st.button("Clear Current Query"):
                        st.session_state.current_analysis = None
                        st.session_state.query_results = None
                        st.rerun()
                        
            elif st.session_state.query_results:
                # If no dataframe, show text results in a smaller container
                st.text_area("Results", st.session_state.query_results['text_result'], height=200, disabled=True)
        else:
            st.info("Run an analysis to see query and results")

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
            2. Pay special attention to any recently uploaded tables: {', '.join(st.session_state.uploaded_tables)}
            3. Make sure to query the correct table that contains the data for this question
            4. Show the exact SQL query you're executing
            """,
            expected_output="SQL query results with the query used",
            agent=sql_specialist
        )
        
        data_analysis = Task(
            description=f"""
            Analyze the data for: {question}
            
            Provide insights based on the actual data returned.
            """,
            expected_output="Detailed analysis with insights",
            agent=data_analyst
        )
        
        business_report = Task(
            description=f"""
            Create business recommendations for: {question}
            
            Structure your response with clear sections.
            """,
            expected_output="Structured business report",
            agent=business_consultant
        )
        
        # Create and run crew
        crew = Crew(
            agents=[sql_specialist, data_analyst, business_consultant],
            tasks=[data_extraction, data_analysis, business_report],
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
    st.header("CrewAI SQL Agent")
    
    st.divider()
    
    # Show uploaded tables
    if st.session_state.uploaded_tables:
        st.subheader("Uploaded Tables")
        for table in st.session_state.uploaded_tables[-5:]:
            st.info(table)
    
    st.divider()
    
    with st.expander("Help"):
        st.markdown("""
        **CSV Upload Tips:**
        
        1. Upload CSV → Preview → Load to Database
        2. New tables are immediately queryable
        3. Use suggested questions or write your own
        4. The system will query your uploaded data
        """)

# Run the app
if __name__ == "__main__":
    main()