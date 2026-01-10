from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
import os
import sqlite3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from long_term_memory_helper import (
    extract_and_store_memory,
    read_user_memory
)

load_dotenv(override=True)

# -----------------------------
# LLM
# -----------------------------
def get_groq_llm():
    return ChatOpenAI(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )

llm = get_groq_llm()

# -----------------------------
# Short-term memory (chat)
# -----------------------------
sqlite_conn = sqlite3.connect(
    "whatsapp_bot_memory.sqlite",
    check_same_thread=False
)
memory = SqliteSaver(sqlite_conn)

# -----------------------------
# State
# -----------------------------
class WhatsAppState(TypedDict):
    messages: Annotated[list, add_messages]
    image_url: Optional[str]

# -----------------------------
# Node 1: Remember
# -----------------------------
def remember_node(state: WhatsAppState, config):
    user_id = config["configurable"]["thread_id"]
    last_msg = state["messages"][-1].content

    extract_and_store_memory(user_id, last_msg)
    return {}

# -----------------------------
# Node 2: Chat Agent
# -----------------------------
SYSTEM_PROMPT = """
You are a personal WhatsApp AI assistant.

Known user context:
{memory}

Use this information naturally.
Avoid generic responses.
Be concise and practical.
"""

def chat_agent(state: WhatsAppState, config):
    user_id = config["configurable"]["thread_id"]

    memories = read_user_memory(user_id, limit=5)
    memory_text = "\n".join(m.value["data"] for m in memories) if memories else "None"

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(memory=memory_text)
    )

    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

# -----------------------------
# Node 3: Vision Agent
# -----------------------------
def vision_agent(state: WhatsAppState, config):
    last_user_message = state["messages"][-1].content

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": state["image_url"]}
                },
                {
                    "type": "text",
                    "text": last_user_message or "Describe this image"
                }
            ]
        }
    ])

    return {"messages": [response]}

# -----------------------------
# Supervisor
# -----------------------------
def supervisor(state: WhatsAppState):
    if state.get("image_url"):
        return "vision_agent"
    return "remember"

# -----------------------------
# Build Graph
# -----------------------------
def build_langgraph_app():
    graph = StateGraph(WhatsAppState)

    graph.add_node("remember", remember_node)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("vision_agent", vision_agent)

    graph.set_conditional_entry_point(
        supervisor,
        {
            "remember": "remember",
            "vision_agent": "vision_agent"
        }
    )

    graph.add_edge("remember", "chat_agent")
    graph.add_edge("chat_agent", END)
    graph.add_edge("vision_agent", END)

    return graph.compile(checkpointer=memory)
