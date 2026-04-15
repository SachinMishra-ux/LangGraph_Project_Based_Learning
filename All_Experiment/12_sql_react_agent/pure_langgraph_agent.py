import sqlite3
import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Imports for pure LangGraph implementation
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv("../../.env")
DB_PATH = "sales.db"

# 1. Define Valid Tools (Same as before)
@tool
def execute_sql_query(query: str) -> str:
    """Executes a SQL query against the sales database and returns the results.
    If the query is invalid, it returns the error so you can try again."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            conn.commit()
            conn.close()
            return "Query executed successfully, no data returned."
        
        col_names = [description[0] for description in cursor.description]
        formatted_results = [dict(zip(col_names, row)) for row in results]
        
        conn.close()
        return str(formatted_results)
    
    except Exception as e:
        return f"SQL Error: {str(e)}\nPlease rewrite your query and try again."

def get_db_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return "\n".join([table[0] for table in tables if table[0]])


# 2. Build the ReAct Pattern using purely LangGraph Nodes
def setup_pure_graph():
    # Setup tools and bind them to the LLM
    tools = [execute_sql_query]
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # IMPORTANT: We explicitly bind the tools to the LLM so it knows it can output JSON to call them!
    llm_with_tools = llm.bind_tools(tools)
    
    # Define our System Instructions
    db_schema = get_db_schema(DB_PATH)
    system_prompt = SystemMessage(content=f"""You are a Data Analyst Agent. You answer questions by actively querying an SQLite database.
    
    Here is the exact schema of the database you are working with:
    {db_schema}
    
    INSTRUCTIONS:
    1. ALWAYS use the `execute_sql_query` tool to fetch data. Do not guess.
    2. If the tool returns a 'SQL Error', analyze the error message.
    3. Look at the schema above, correct your SQL query, and call the tool AGAIN.
    4. Only provide your final answer once you have successfully retrieved data.
    """)

    # ==========================
    # DEFINING THE GRAPH NODES
    # ==========================
    
    # Node 1: The Reasoner
    def call_model(state: MessagesState):
        """This node invokes the LLM. It injects the system prompt and passes the conversation history."""
        messages = state['messages']
        
        # Inject the system prompt if it's not already at the front
        if messages and not isinstance(messages[0], SystemMessage):
            messages = [system_prompt] + messages
            
        # Call the LLM (which has tools bound to it)
        response = llm_with_tools.invoke(messages)
        
        # Return a dictionary. LangGraph's MessagesState will automatically append this to the history.
        return {"messages": [response]}

    # Node 2: The Actor
    # LangGraph provides a ToolNode pre-built that automatically loops through tool_calls from an AI message and executes them
    tool_node = ToolNode(tools)

    # ==========================
    # WIRING THE GRAPH EDGES
    # ==========================
    
    # MessagesState handles the list[BaseMessage] 'messages' key array appending automatically
    workflow = StateGraph(MessagesState)
    
    # Register our nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set Entry Point -> Automatically trigger the LLM first
    workflow.add_edge(START, "agent")
    
    # CONDITIONAL ROUTING LOGIC:
    # After the 'agent' node runs, where do we go?
    # `tools_condition` looks at the LLM's response. 
    # If the LLM requested a tool, it routes to our "tools" node. 
    # If the LLM gave a final text answer, it routes to END.
    workflow.add_conditional_edges(
        "agent", 
        tools_condition,
    )
    
    # Simple explicit routing: After we run a tool, ALWAYS go back to the agent so it can observe what happened.
    workflow.add_edge("tools", "agent")

    # Compile into an executable application
    return workflow.compile()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("Database not found! Please run 'python setup_db.py' first.")
        exit(1)
        
    print("=== Starting Pure LangGraph ReAct Agent ===")
    app = setup_pure_graph()
    
    while True:
        user_input = input("\nAsk a question about the sales data (or 'quit'): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        print("\nAgent is thinking and querying...\n")
        
        # Stream the output so we can intercept the steps in the graph
        events = app.stream(
            {"messages": [("user", user_input)]}, 
            stream_mode="values"
        )
        
        for event in events:
            message = event["messages"][-1]
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                tool_call = message.tool_calls[0]
                print(f"[NODE: agent] Decided to act! Generating SQL: {tool_call['args'].get('query')}")
            elif message.type == "tool":
                content = message.content
                if "SQL Error" in content:
                    print(f"[NODE: tools] Database threw error: {content}")
                else:
                    print(f"[NODE: tools] Database returned data: {content}")
            elif message.type == "ai" and not hasattr(message, "tool_calls") or not getattr(message, "tool_calls", None):
                print(f"\n[FINAL ANSWER]:\n{message.content}")

