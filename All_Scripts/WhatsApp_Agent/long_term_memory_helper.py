import os
import uuid
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from embedding_setup import get_gemini_embedding_model

# -----------------------------
# Config
# -----------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "whatsapp_bot"
COLLECTION_NAME = "long_term_memory"

# -----------------------------
# Vector index config
# -----------------------------
index_config = create_vector_index_config(
    embed=get_gemini_embedding_model(),
    dims=1024,
    fields=["data"]
)

# -----------------------------
# Memory extraction LLM
# -----------------------------
def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7, max_tokens=2000
    )

memory_llm = get_groq_llm()

class MemoryItem(BaseModel):
    text: str
    is_new: bool

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """
You manage long-term memory for a WhatsApp assistant.

EXISTING MEMORY:
{existing}

TASK:
- Extract stable user facts only (name, profession, projects, preferences)
- No assumptions
- Atomic, factual sentences
- Mark is_new=true only if new
"""

# -----------------------------
# Internal helper
# -----------------------------
def _with_store():
    return MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        index_config=index_config
    )

# -----------------------------
# Public API
# -----------------------------
def read_user_memory(user_id: str, query: str | None = None, limit: int = 5):
    ns = ("user", user_id, "profile")

    with _with_store() as store:
        if query:
            return store.search(ns, query=query, limit=limit)
        return store.search(ns, limit=limit)


def write_user_memory(user_id: str, text: str):
    ns = ("user", user_id, "profile")

    with _with_store() as store:
        store.put(
            namespace=ns,
            key=str(uuid.uuid4()),
            value={"data": text}
        )


def extract_and_store_memory(user_id: str, user_message: str):
    ns = ("user", user_id, "profile")

    with _with_store() as store:
        items = store.search(ns)
        existing = "\n".join(i.value["data"] for i in items) if items else "(empty)"

    decision: MemoryDecision = memory_extractor.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(existing=existing)),
        {"role": "user", "content": user_message}
    ])

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                write_user_memory(user_id, mem.text)
