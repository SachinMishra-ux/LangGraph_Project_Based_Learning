# %%
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

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
class JokeState(TypedDict):

    topic: str
    joke: str
    explanation: str

# %%
def generate_joke(state: JokeState):

    prompt = f'generate a joke on the topic {state["topic"]}'
    response = llm.invoke(prompt).content

    return {'joke': response}

# %%
def generate_explanation(state: JokeState):

    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = llm.invoke(prompt).content

    return {'explanation': response}

# %%
graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

# %%
workflow

# %%
config1 = {"configurable": {"thread_id": "1"}}
workflow.invoke({'topic':'pizza'}, config=config1)

# %%
workflow.get_state(config1)

# %% [markdown]
# ## Intermediate States

# %%
list(workflow.get_state_history(config1))

# %%
config2 = {"configurable": {"thread_id": "2"}}
workflow.invoke({'topic':'pasta'}, config=config2)

# %%
workflow.get_state(config2)

# %%
list(workflow.get_state_history(config2))

# %% [markdown]
# ### Time Travel
# 
# ## we can go back to any previous node to do the execution again for debugging & changing the input as well

# %%
## so let's go to the pizza example where we got the topic & we want to go back to joke generation step & further steps.
## In order to do that we need to grab the checkpoint id where we got the topic. We can get it from the state history.

# %%
list(workflow.get_state_history(config1))

# %%
workflow.get_state({"configurable": {"thread_id": "1", "checkpoint_id": "1f0cb798-2a07-6a58-8000-c3993120bd58"}}) ## checkpoint id from pizza run where we haver the topic but we need to generate the joke

# %%
workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": "1f0cb798-2a07-6a58-8000-c3993120bd58"}})

# %% [markdown]
# ## now we can see more Snapshots due to the re-execution

# %%
list(workflow.get_state_history(config1))

# %% [markdown]
# #### Updating State

# %%
workflow.update_state({"configurable": {"thread_id": "1", "checkpoint_id": "1f0cb798-2a07-6a58-8000-c3993120bd58", "checkpoint_ns": ""}}, {'topic':'Burger'})

# %% [markdown]
# ## now we have updated the state with topic as samosa, one more snapshot will be created

# %%
list(workflow.get_state_history(config1))

# %% [markdown]
# ## and to execute the workflow from the updated state (samosa) we need to pass the checkpoint id of samosa snapshot

# %%
workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": "1f0cb7b9-a95e-6a80-8001-9aa3bded3643"}})

# %%
list(workflow.get_state_history(config1))

# %% [markdown]
# ### Fault Tolerance kind of similar as human in the loop (In HITL we do it perpupose fully using langraph interrupt)

# %%
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import time

# %%
# 1. Define the state
class CrashState(TypedDict):
    input: str
    step1: str
    step2: str
    step3: str

# %%
# 2. Define steps
def step_1(state: CrashState) -> CrashState:
    print("✅ Step 1 executed")
    return {"step1": "done", "input": state["input"]}

def step_2(state: CrashState) -> CrashState:
    print("⏳ Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)")
    time.sleep(10)  # Simulate long-running hang
    return {"step2": "done"}

def step_3(state: CrashState) -> CrashState:
    print("✅ Step 3 executed")
    return {"step3": "done"}

# %%
# 3. Build the graph
builder = StateGraph(CrashState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.set_entry_point("step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# %%
graph

# %%
try:
    print("▶️ Running graph: Please manually interrupt during Step 2...")
    graph.invoke({"input": "start"}, config={"configurable": {"thread_id": 'thread-1'}})
except KeyboardInterrupt:
    print("❌ Kernel manually interrupted (crash simulated).")

# %%
graph.get_state({"configurable": {"thread_id": 'thread-1'}})

# %%
# 6. Re-run to show fault-tolerant resume
print("\n🔁 Re-running the graph to demonstrate fault tolerance...")
final_state = graph.invoke(None, config={"configurable": {"thread_id": 'thread-1'}})
print("\n✅ Final State:", final_state)

# %%
list(graph.get_state_history({"configurable": {"thread_id": 'thread-1'}}))


