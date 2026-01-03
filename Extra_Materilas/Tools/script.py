# script.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os

# =========================
# Environment
# =========================
load_dotenv()

# =========================
# LLM
# =========================
def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )

llm = get_groq_llm()

# =========================
# Tools
# =========================
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Basic arithmetic calculator."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation"}

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price from Alpha Vantage."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    return requests.get(url).json()

tools = [calculator, get_stock_price, search_tool]

# Bind tools
llm_with_tools = llm.bind_tools(tools)

# =========================
# State
# =========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =========================
# Nodes
# =========================
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# =========================
# Graph
# =========================
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

chatbot = graph.compile()

# =========================
# Public function for UI
# =========================
def run_chat(messages: list[BaseMessage]):
    """Called by Streamlit UI"""
    result = chatbot.invoke({"messages": messages})
    return result["messages"]
