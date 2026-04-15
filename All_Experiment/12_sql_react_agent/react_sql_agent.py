import sqlite3
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Load environment variables (ensure OPENAI_API_KEY is set in your .env)
load_dotenv("../../.env")

DB_PATH = "sales.db"

# 1. Provide the Tool with Error Catching Logic
@tool
def execute_sql_query(query: str) -> str:
    """Executes a SQL query against the sales database and returns the results.
    If the query is invalid, it returns the error so you can try again."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        # If no results but it's an INSERT/UPDATE etc, return success message
        if not results:
            conn.commit()
            conn.close()
            return "Query executed successfully, no data returned."
        
        # Get column names to format the output nicely
        col_names = [description[0] for description in cursor.description]
        formatted_results = [dict(zip(col_names, row)) for row in results]
        
        conn.close()
        return str(formatted_results)
    
    except Exception as e:
        # Crucial for the ReAct pattern: return the error message text!
        return f"SQL Error: {str(e)}\nPlease rewrite your query and try again."

# 2. Extract database schema dynamically to inject into the LLM context
def get_db_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    
    schema_details = "\n".join([table[0] for table in tables if table[0]])
    return schema_details

# 3. Create the Agent Configuration
def setup_agent():
    # Model - using Groq as requested
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # Tools
    tools = [execute_sql_query]
    
    # System Prompt with the schema
    db_schema = get_db_schema(DB_PATH)
    system_prompt = f"""You are a Data Analyst Agent. You answer questions by actively querying an SQLite database.
    
    Here is the exact schema of the database you are working with:
    {db_schema}
    
    INSTRUCTIONS:
    1. ALWAYS use the `execute_sql_query` tool to fetch data. Do not guess.
    2. If the tool returns a 'SQL Error', analyze the error message.
    3. Look at the schema above, correct your SQL query, and call the tool AGAIN.
    4. Only provide your final answer once you have successfully retrieved data.
    """
    
    # Compile the ReAct agent
    agent_executor = create_agent(llm, tools=tools, system_prompt=system_prompt)
    return agent_executor

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("Database not found! Please run 'python setup_db.py' first.")
        exit(1)
        
    print("=== Starting SQL ReAct Agent ===")
    agent = setup_agent()
    
    while True:
        user_input = input("\nAsk a question about the sales data (or 'quit'): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        print("\nAgent is thinking and querying...\n")
        
        # Stream the output so we can see the ReAct loop in real-time
        events = agent.stream(
            {"messages": [("user", user_input)]}, 
            stream_mode="values"
        )
        
        for event in events:
            message = event["messages"][-1]
            if message.type == "ai" and message.tool_calls:
                # LLM decides to take an action
                tool_call = message.tool_calls[0]
                print(f"[THINKING/ACTION] Generating SQL: {tool_call['args'].get('query')}")
            elif message.type == "tool":
                # The tool returns data or an error
                content = message.content
                if "SQL Error" in content:
                    print(f"[RETRY ERROR] {content}")
                else:
                    print(f"[OBSERVATION] Database returned data: {content}")
            elif message.type == "ai" and not message.tool_calls:
                # The final answer
                print(f"\n[FINAL ANSWER]:\n{message.content}")
