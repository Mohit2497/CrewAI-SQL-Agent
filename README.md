# CrewAI SQL Agent

An AI-powered SQL agent system that enables natural language database queries using CrewAI, LangChain, and Streamlit. Upload CSV files and ask questions in plain English to get SQL queries.

## Features

- **Natural Language to SQL**: Ask questions in plain English, get SQL queries automatically
- **CSV Upload & Analysis**: Upload CSV files directly through the web interface
- **Intelligent Query Generation**: AI agents write optimized SQL queries
- **Query History**: Track all queries with deduplication and export options
- **Intelligent Data Validation**: Comprehensive file validation with warnings and suggestions
- **Interactive Results**: View query results in tables with export to CSV
- **Multi-Agent System**: Specialized agents for SQL, analysis, and business strategy

<img width="752" height="532" alt="image" src="https://github.com/user-attachments/assets/f623a040-75e9-466f-aa6b-e6badd828ac8" />


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

3. **Run on Hugging Face**
Try it live: [CrewAI SQL Agent](https://huggingface.co/spaces/gLiTcH9724/CrewAI-SQL-Agent)
Or deploy your own:

- Go to Hugging Face Spaces
- Create new Space → Choose Streamlit
- Upload the app.py file
- Add your API key in Settings → Repository secrets as GROQ_API_KEY

4. **Upload your data**
- Go to the "CSV Upload" tab
- Upload one or more CSV files
- Preview the data and load it into the database

5. **Ask questions**
- Switch to the "Analysis" tab
- Type your question in natural language
- Examples:

6. **View results**
- See the generated SQL query
- View results in a table format
- Download results as CSV

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

2. **Workflow**
   - Upload CSV → Ask question → SQL generation → Query execution

3. **Query Processing**
   - Specialized AI agents process questions
   - SQL queries are validated before execution

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
3. Commit your changes (`git commit -m 'Add some AmazingFeature')
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [LangChain](https://github.com/langchain-ai/langchain) for LLM tooling
- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace](https://huggingface.co/spaces) for hosting the application
- [Groq](https://groq.com/) for LLaMA 3 API access

## Contact

For questions or support, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and demonstration purposes. Always verify AI-generated SQL queries before running them on production databases.
