# CrewAI SQL Agent

An AI-powered SQL agent system that enables natural language database queries using CrewAI, LangChain, and Streamlit. Upload CSV files and ask questions in plain English to get SQL queries, data analysis, and business insights.

## Features

- **Natural Language to SQL**: Ask questions in plain English, get SQL queries automatically
- **CSV Upload & Analysis**: Upload CSV files directly through the web interface
- **Intelligent Query Generation**: AI agents write optimized SQL queries
- **Business Insights**: Get data analysis and strategic recommendations
- **Query History**: Track all queries with deduplication and export options
- **Interactive Results**: View query results in tables with export to CSV
- **Multi-Agent System**: Specialized agents for SQL, analysis, and business strategy

## Prerequisites

- Python 3.8 or higher
- Groq API key (for LLaMA 3 access)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mohit2497/CrewAI-SQL-Agent.git
cd CrewAI-SQL-Agent
```

2. **Create a virtual environment**
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_PATH=company_data.db
```

To get a Groq API key:
- Visit [console.groq.com](https://console.groq.com)
- Sign up for a free account
- Generate an API key

## Usage

1. **Start the application**
```bash
streamlit run main.py
```

2. **Open your browser**
Navigate to `http://localhost:8501`

3. **Upload your data**
- Go to the "CSV Upload" tab
- Upload one or more CSV files
- Preview the data and load it into the database

4. **Ask questions**
- Switch to the "Analysis" tab
- Type your question in natural language
- Examples:
  - "What is the average salary by department?"
  - "Show me the top 10 customers by revenue."
  - "What are the monthly sales trends?"

5. **View results**
- See the generated SQL query
- View results in a table format
- Download results as CSV
- Read AI-generated analysis and recommendations

## Project Structure

```
CrewAI-SQL-Agent/
│
├── streamlit_app_clean.py    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (create this)
├── .env.example             # Environment variables template
├── README.md                # This file
├── .gitignore              # Git ignore configuration
└── setup_database.py        # (Optional) Pre-populate database with sample data
```

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for LLaMA 3 access
- `DATABASE_PATH`: Path to SQLite database (default: `company_data.db`)

### Model Selection

The system uses LLaMA 3 70B by default.

## How It Works

1. **Multi-Agent Architecture**
   - **SQL Specialist**: Converts natural language to SQL queries
   - **Data Analyst**: Analyzes query results and finds patterns
   - **Business Consultant**: Provides strategic recommendations

2. **Workflow**
   - Upload CSV → Ask question → SQL generation → Query execution → Analysis → Business insights

3. **Query Processing**
   - Questions are processed by specialized AI agents
   - SQL queries are validated before execution
   - Results are analyzed for patterns and insights
   - Business recommendations are generated

## Troubleshooting

### Common Issues

1. **"LLM initialization failed"**
   - Check your `GROQ_API_KEY` in the `.env` file
   - Ensure you have an active internet connection

2. **CSV upload errors**
   - Ensure CSV files are properly formatted
   - Check for special characters in column names
   - Try different encoding if you see errors

3. **Query errors**
   - Verify table names match uploaded CSV filenames
   - Check column names in your questions
   - Ensure data types are compatible for operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [LangChain](https://github.com/langchain-ai/langchain) for LLM tooling
- [Streamlit](https://streamlit.io/) for the web interface
- [Groq](https://groq.com/) for LLaMA 3 API access

## Contact

For questions or support, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and demonstration purposes. Always verify AI-generated SQL queries before running them on production databases.
