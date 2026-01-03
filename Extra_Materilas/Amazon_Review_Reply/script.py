# script.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

# =========================
# Environment & LLM
# =========================
load_dotenv()

def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.6,
        max_tokens=1500
    )

model = get_groq_llm()

# =========================
# Schemas
# =========================
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

sentiment_model = model.with_structured_output(SentimentSchema)
diagnosis_model = model.with_structured_output(DiagnosisSchema)

# =========================
# LangGraph State
# =========================
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# =========================
# Graph Nodes
# =========================
def find_sentiment(state: ReviewState):
    sentiment = sentiment_model.invoke(
        f"Determine the sentiment of this product review:\n{state['review']}"
    ).sentiment
    return {"sentiment": sentiment}

def route_sentiment(state: ReviewState):
    return "positive_response" if state["sentiment"] == "positive" else "run_diagnosis"

def positive_response(state: ReviewState):
    prompt = f"""
You are an Amazon seller support assistant.

Write a warm, appreciative response to this positive customer review:
"{state['review']}"

Encourage the customer to continue shopping on Amazon.
"""
    return {"response": model.invoke(prompt).content}

def run_diagnosis(state: ReviewState):
    diagnosis = diagnosis_model.invoke(
        f"Analyze the following negative Amazon product review:\n{state['review']}"
    )
    return {"diagnosis": diagnosis.model_dump()}

def negative_response(state: ReviewState):
    d = state["diagnosis"]
    prompt = f"""
You are an Amazon Customer Experience Specialist.

The customer reported a {d['issue_type']} issue,
tone: {d['tone']},
urgency: {d['urgency']}.

Write a calm, empathetic, professional response offering help and resolution.
"""
    return {"response": model.invoke(prompt).content}

# =========================
# Build Graph
# =========================
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", route_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

# =========================
# Helper for UI
# =========================
def analyze_review(review: str) -> ReviewState:
    return workflow.invoke({"review": review})
