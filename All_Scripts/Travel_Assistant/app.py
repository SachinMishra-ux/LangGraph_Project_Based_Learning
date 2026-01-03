import streamlit as st
from langgraph.types import Command
from workflow import build_workflow
import uuid

st.set_page_config(page_title="AI Travel Planner", layout="centered")
st.title("✈️ AI Travel Planner")

# -----------------------------
# Session init
# -----------------------------
if "graph" not in st.session_state:
    st.session_state.graph = build_workflow()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "result" not in st.session_state:
    st.session_state.result = st.session_state.graph.invoke(
        {},
        config={"configurable": {"thread_id": st.session_state.thread_id}}
    )

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# -----------------------------
# Render loop
# -----------------------------
result = st.session_state.result

if "__interrupt__" in result:
    intr = result["__interrupt__"][0]
    st.write(intr.value)

    user_input = st.text_input("Your response", key="input")

    if st.button("Submit"):
        st.session_state.result = st.session_state.graph.invoke(
            Command(resume=user_input),
            config
        )
        st.rerun()

elif result.get("done"):
    st.success("🎉 Trip planning completed!")
    st.json(result)

else:
    st.write("Processing...")
