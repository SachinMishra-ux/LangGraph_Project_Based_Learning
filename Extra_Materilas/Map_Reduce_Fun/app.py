import streamlit as st
from langgraph_mapreduce import build_graph

st.set_page_config(page_title="LangGraph Map-Reduce Fun", layout="wide")

st.title("🧠 Learn Map-Reduce with LangGraph (Fun Way!)")
st.caption("Explained like you're 10 years old 🎈")

# ---------------- INTRO ----------------
st.markdown("""
### 🧩 What is Map-Reduce?

👧 **MAP** → Many kids count words in different notebooks  
👨‍🏫 **REDUCE** → Teacher adds all counts together  

LangGraph helps us **organize this teamwork**!
""")

st.divider()

# ---------------- INPUT DOCS ----------------
st.subheader("📄 Step 1: Enter Documents (Teacher gives notebooks)")

docs = st.text_area(
    "Enter multiple sentences (each line = one document)",
    value="Hello world hello\nWorld of LangGraph\nHello Map Reduce",
    height=120
).split("\n")

st.divider()

# ---------------- VISUAL FLOW ----------------
st.subheader("🗺️ Step 2: How the Graph Thinks")

col1, col2, col3, col4, col5 = st.columns(5)

col1.markdown("### 🚦 START")
col2.markdown("### 👨‍🏫 Orchestrator")
col3.markdown("### 👧👦 Map Workers")
col4.markdown("### 🧮 Reduce")
col5.markdown("### 🏁 END")

st.markdown("""
➡️ **Important:**  
Map workers are created **dynamically**, so arrows are invisible in the graph picture!
""")

st.divider()

# ---------------- RUN BUTTON ----------------
if st.button("🚀 Run Map-Reduce"):
    st.subheader("🔄 Step 3: Map Phase (Fan-Out)")

    for i, d in enumerate(docs, 1):
        st.markdown(f"👧 Worker {i} received:")
        st.code(d)

    graph = build_graph()

    initial_state = {
        "docs": docs,
        "partials": [],
        "final_counts": {}
    }

    config = {"configurable": {"thread_id": "demo"}}
    graph.invoke(initial_state, config=config)

    snapshot = graph.get_state(config)

    st.divider()

    st.subheader("📦 Step 4: Reduce Phase (Fan-In)")
    st.markdown("👨‍🏫 Teacher combines all word counts:")

    st.json(snapshot.values["final_counts"])

    st.balloons()

# ---------------- TEACHING TIP ----------------
st.divider()
st.info("""
🎯 **Teaching Tip**

If the graph picture looks disconnected:
- That's NORMAL ✅
- `Send()` creates **runtime connections**
- Execution ≠ Visualization
""")
