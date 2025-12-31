from typing import TypedDict, List, Dict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import InMemorySaver

# ---------------- STATE ----------------
class AggregateState(TypedDict, total=False):
    docs: List[str]
    partials: Annotated[List[Dict[str, int]], add]
    final_counts: Dict[str, int]

class WorkerState(TypedDict):
    doc: str

# ---------------- ORCHESTRATOR ----------------
def orchestrator(state: AggregateState) -> Command:
    sends = [Send("map_doc", {"doc": d}) for d in state["docs"]]
    return Command(update={}, goto=sends)

# ---------------- MAP ----------------
def map_doc(state: WorkerState):
    counts = {}
    for word in state["doc"].split():
        w = word.lower().strip(".,!?:;\"'()")
        if w:
            counts[w] = counts.get(w, 0) + 1
    return {"partials": [counts]}

# ---------------- REDUCE ----------------
def reducer(state: AggregateState):
    final = {}
    for part in state["partials"]:
        for word, count in part.items():
            final[word] = final.get(word, 0) + count
    return {"final_counts": final}

# ---------------- BUILD GRAPH ----------------
def build_graph():
    builder = StateGraph(AggregateState)

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("map_doc", map_doc)
    builder.add_node("reduce", reducer)

    builder.add_edge(START, "orchestrator")
    builder.add_edge("map_doc", "reduce")
    builder.add_edge("reduce", END)

    return builder.compile(checkpointer=InMemorySaver())
