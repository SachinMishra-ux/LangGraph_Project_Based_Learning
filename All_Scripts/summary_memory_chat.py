# summary_memory_chat.py

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, RemoveMessage

load_dotenv(override=True)

def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7, max_tokens=2000
    )

model = get_groq_llm()

class ChatState(MessagesState):
    summary: str

def chat(state: ChatState):
    messages = []

    if state["summary"]:
        messages.append({
            "role": "system",
            "content": f"Conversation summary:\n{state['summary']}"
        })

    messages.extend(state["messages"])
    response = model.invoke(messages)
    return {"messages": [response]}

def summarize(state: ChatState):
    prompt = "Summarize the conversation so far."

    summary_input = state["messages"] + [HumanMessage(content=prompt)]
    response = model.invoke(summary_input)

    to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in to_delete],
    }

def should_summarize(state: ChatState):
    return len(state["messages"]) > 6

builder = StateGraph(ChatState)
builder.add_node("chat", chat)
builder.add_node("summarize", summarize)

builder.add_edge(START, "chat")
builder.add_conditional_edges(
    "chat",
    should_summarize,
    {True: "summarize", False: "__end__"}
)
builder.add_edge("summarize", "__end__")

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "summary-demo"}}

# ---- Run ----
graph.invoke({"messages": [HumanMessage(content="Hi, I'm Nitish")], "summary": ""}, config)
graph.invoke({"messages": [HumanMessage(content="I am learning LangGraph")], "summary": ""}, config)
graph.invoke({"messages": [HumanMessage(content="Explain short term memory")], "summary": ""}, config)
result = graph.invoke({"messages": [HumanMessage(content="What is my name?")], "summary": ""}, config)

print(result["messages"][-1].content)
