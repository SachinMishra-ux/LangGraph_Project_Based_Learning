from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
import os
import sqlite3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)

def get_groq_llm():
    return ChatOpenAI(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )

llm = get_groq_llm()

sqlite_conn = sqlite3.connect(
    "whatsapp_bot_memory.sqlite",
    check_same_thread=False
)

memory = SqliteSaver(sqlite_conn)


class WhatsAppState(TypedDict):
    messages: Annotated[list, add_messages]
    image_url: Optional[str]


## Chat Agent (Text + Memory)
def chat_agent(state: WhatsAppState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

## 🖼️ Vision Agent (Image Understanding)
def vision_agent(state: WhatsAppState):
    last_user_message = state["messages"][-1].content

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": state["image_url"]
                    }
                },
                {
                    "type": "text",
                    "text": last_user_message or "Describe this image"
                }
            ]
        }
    ])

    return {
        "messages": [response]
    }


# Supervisor (Routing Logic)

def supervisor(state: WhatsAppState):
    if state.get("image_url"):
        return "vision_agent"
    return "chat_agent"

# Build LangGraph App (with Memory)

def build_langgraph_app():
    graph = StateGraph(WhatsAppState)

    graph.add_node("chat_agent", chat_agent)
    graph.add_node("vision_agent", vision_agent)

    graph.set_conditional_entry_point(
        supervisor,
        {
            "chat_agent": "chat_agent",
            "vision_agent": "vision_agent"
        }
    )

    graph.add_edge("chat_agent", END)
    graph.add_edge("vision_agent", END)

    langgraph_app = graph.compile(checkpointer=memory)
    return langgraph_app
