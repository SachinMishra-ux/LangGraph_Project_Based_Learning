# %%
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# %%


# %%
from langgraph.graph.message import add_messages

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

# %%
load_dotenv()
import os

# %%
def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7, max_tokens=2000
    )

llm = get_groq_llm()




# %%
def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # response store state
    return {'messages': [response]}

# %%
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile()

# %%
chatbot

# %%
initial_state = {
    'messages': [HumanMessage(content='What is the capital of india')]
}

chatbot.invoke(initial_state)['messages'][-1].content

# %%
initial_state = {
    'messages': [HumanMessage(content='HI there my name is sachin, How are you doing today?')]
}

chatbot.invoke(initial_state)['messages'][-1].content

# %%
initial_state = {
    'messages': [HumanMessage(content='Hey0 do you remember my name?')]
}

chatbot.invoke(initial_state)['messages'][-1].content

# %% [markdown]
# ## Let's add memory

# %%
from langgraph.checkpoint.memory import InMemorySaver

# %%
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

# %%
chatbot

# %%
config = {"configurable": {"thread_id": "1"}}

# %%
initial_state = {
    'messages': [HumanMessage(content='HI there my name is sachin')]
}

chatbot.invoke(initial_state,config=config)['messages'][-1].content

# %%
initial_state = {
    'messages': [HumanMessage(content='Do you remember my name?')]
}

chatbot.invoke(initial_state,config=config)['messages'][-1].content

# %%
config2 = {"configurable": {"thread_id": "2"}}

# %%
initial_state = {
    'messages': [HumanMessage(content='Address my name & give me a greeting message & surprise me')]
}

chatbot.invoke(initial_state,config=config2)['messages'][-1].content

# %%
initial_state = {
    'messages': [HumanMessage(content='Hey My Name is Robin, How are you doing today?')]
}

chatbot.invoke(initial_state,config=config2)['messages'][-1].content

# %%
initial_state = {
    'messages': [HumanMessage(content='Do you remember my name?')]
}

chatbot.invoke(initial_state,config=config2)['messages'][-1].content
