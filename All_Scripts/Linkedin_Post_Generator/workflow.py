# linkedin_langgraph_workflow.py
import os
from typing import TypedDict, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import requests

# LLM / LangChain imports (matches your environment)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

# -------------------------
# 1) LLM setup (Groq)
# -------------------------
def get_groq_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )

llm = get_groq_llm()

# -------------------------
# 2) Environment (LinkedIn)
# -------------------------
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_AUTHOR_URN = os.getenv("LINKEDIN_AUTHOR_URN")  # e.g., urn:li:person:yyotj6wNOM
# Optional: allow passing token/urn at runtime from config if needed.

# -------------------------
# 3) Workflow state schema
# -------------------------
class PostState(TypedDict, total=False):
    info: dict
    draft: str
    human_feedback: Optional[str]
    approved: bool
    final_post: Optional[str]
    post_id: Optional[str]
    post_error: Optional[str]

# -------------------------
# 4) Nodes: generate / revise / feedback / decide / post
# -------------------------
def generate_draft(state: PostState) -> dict:
    info = state["info"]
    prompt = f"""You are writing a LinkedIn post. Here is the input:
Topic: {info.get('topic')}
Key points: {info.get('key_points')}
Tone: {info.get('tone')}
Audience: {info.get('audience')}

Write a LinkedIn-style post (1-2 short paragraphs) in plain human voice suitable for posting on LinkedIn. Do NOT mention brands unless instructed."""
    response = llm.invoke([HumanMessage(content=prompt)])
    draft = response.content.strip()
    print("\n=== AI Generated Draft ===\n", draft)
    return {"draft": draft, "approved": False}

def ask_for_feedback(state: PostState) -> dict:
    print("\n--- PAUSING FOR HUMAN REVIEW ---")
    print("Draft:\n", state.get("draft", ""))
    # interrupt returns human input when resumed via Command(resume={...})
    feedback = interrupt("Please review the draft. Provide feedback (or type 'approve' if OK):")
    approved = (feedback.strip().lower() == "approve")
    return {"human_feedback": feedback, "approved": approved}

def decide_next(state: PostState) -> str:
    # return routing key
    if state.get("approved", False):
        return "approved"
    return "revise"

def revise_draft(state: PostState) -> dict:
    feedback = state.get("human_feedback") or ""
    old_draft = state.get("draft", "")
    prompt = f"""You are rewriting a LinkedIn post. Original draft:
{old_draft}

Reviewer feedback:
{feedback}

Revise the draft accordingly (keeping same tone and audience). Output only the revised post text."""
    response = llm.invoke([HumanMessage(content=prompt)])
    new_draft = response.content.strip()
    print("\n=== AI Revised Draft ===\n", new_draft)
    return {"draft": new_draft, "approved": False}

def post_to_linkedin_real(state: PostState) -> Dict[str, Any]:
    """
    Posts final draft to LinkedIn using env vars LINKEDIN_ACCESS_TOKEN and LINKEDIN_AUTHOR_URN.
    Returns dict updates for the state: final_post, post_id on success OR post_error on failure.
    """
    token = LINKEDIN_ACCESS_TOKEN
    author_urn = LINKEDIN_AUTHOR_URN
    draft = state.get("draft", "")

    if not token:
        err = "Missing LINKEDIN_ACCESS_TOKEN in environment"
        print("ERROR:", err)
        return {"post_error": err}

    if not author_urn:
        err = "Missing LINKEDIN_AUTHOR_URN in environment"
        print("ERROR:", err)
        return {"post_error": err}

    url = "https://api.linkedin.com/rest/posts"
    headers = {
        "Authorization": f"Bearer {token}",
        "LinkedIn-Version": "202511",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }
    body = {
        "author": author_urn,
        "commentary": draft,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": []
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=20)
    except Exception as e:
        err = f"HTTP request error: {e}"
        print("ERROR:", err)
        return {"post_error": err}

    print("LinkedIn response status:", resp.status_code)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    print("LinkedIn response body:", data)

    if resp.status_code >= 200 and resp.status_code < 300:
        # success — try to extract post id
        post_id = data.get("id") or data.get("URN") or resp.headers.get("x-restli-id") or None
        return {"final_post": draft, "post_id": post_id}
    else:
        # include more error info in state
        error_message = data if isinstance(data, (str, dict)) else str(data)
        return {"post_error": f"Status {resp.status_code}: {error_message}"}

# -------------------------
# 5) Build StateGraph
# -------------------------
def build_workflow_graph() -> StateGraph:
    builder = StateGraph(PostState)

    builder.add_node("generate_draft", generate_draft)
    builder.add_node("ask_for_feedback", ask_for_feedback)
    builder.add_node("revise_draft", revise_draft)
    builder.add_node("post_to_linkedin", post_to_linkedin_real)

    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "ask_for_feedback")

    # conditional edges: decide_next returns "approved" or "revise"
    builder.add_conditional_edges(
        "ask_for_feedback",
        decide_next,
        {"approved": "post_to_linkedin", "revise": "revise_draft"}
    )

    builder.add_edge("revise_draft", "ask_for_feedback")
    builder.add_edge("post_to_linkedin", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    return graph

# -------------------------
# 6) Automatic run (CLI) — loops on interrupts until done
# -------------------------
def run_workflow_auto(initial_info: dict, thread_id: str = "linkedin_post_thread_1"):
    initial_state: PostState = {
        "info": initial_info,
        "draft": "",
        "human_feedback": None,
        "approved": False,
        "final_post": None,
        "post_id": None,
        "post_error": None
    }

    graph = build_workflow_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Kick off graph execution (it will run until first interrupt)
    graph.invoke(initial_state, config=config)

    # Loop until no interrupts (workflow ended)
    while True:
        state = graph.get_state(config)
        interrupts = getattr(state, "interrupts", None) or []
        if not interrupts:
            break

        # There may be one interrupt (ask_for_feedback) — prompt user and resume
        for intr in interrupts:
            prompt = intr.value if hasattr(intr, "value") else str(intr)
            print(f"\n[HUMAN REVIEW PROMPT] {prompt}")
            # Collect human input from CLI
            answer = input("Enter feedback (or 'approve'): ").strip()
            # resume with mapping
            graph.invoke(Command(resume={intr.id: answer}), config=config)

    # Final state after workflow completion
    final_state = graph.get_state(config)
    print("\n=== Workflow finished. Final state ===")
    print(final_state)
    return final_state

# -------------------------
# 7) Example usage
# -------------------------
if __name__ == "__main__":
    example_info = {
        "topic": "Data Science Workflow Automation with LangGraph",
        "key_points": "human-in-the-loop, agentic workflow, LinkedIn posts, brand voice",
        "tone": "professional yet friendly",
        "audience": "data scientists and AI practitioners on LinkedIn"
    }

    run_workflow_auto(example_info)
