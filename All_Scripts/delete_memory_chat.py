# delete_memory_chat.py

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage

load_dotenv(override=True)

def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7, max_tokens=2000
    )

model = get_groq_llm()

def chat(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def cleanup(state: MessagesState):
    messages = state["messages"]

    if len(messages) > 6:
        to_remove = messages[:4]  # delete oldest messages
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

    return {}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("cleanup", cleanup)

builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup")
builder.add_edge("cleanup", "__end__")

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "delete-demo"}}

# ---- Run ----
graph.invoke({"messages": [{"role": "user", "content": "Hi, I'm Nitish"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "I am learning LangGraph"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Explain short term memory"}]}, config)
result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)

print(result["messages"][-1].content)
