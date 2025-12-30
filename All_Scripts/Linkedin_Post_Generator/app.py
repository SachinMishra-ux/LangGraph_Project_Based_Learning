import streamlit as st
from workflow import (
    build_workflow_graph,
    run_workflow_auto,
    PostState,
    llm,
    LINKEDIN_ACCESS_TOKEN,
    LINKEDIN_AUTHOR_URN,
)
from langgraph.types import Command, interrupt
import time

# ---------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_workflow_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "ui_thread_1"

if "state" not in st.session_state:
    st.session_state.state = None

if "draft_ready" not in st.session_state:
    st.session_state.draft_ready = False

if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False

if "approved" not in st.session_state:
    st.session_state.approved = False

graph = st.session_state.graph
config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("📢 LinkedIn Post Generator (LangGraph + Human In The Loop)")

st.caption("Designed for LangGraph agentic workflow with HITL review.")

# ---------------------------------------------------
# STEP 1: USER PROVIDES INPUT
# ---------------------------------------------------

with st.expander("📝 Enter Post Info", expanded=True):
    topic = st.text_input("Topic", "")
    key_points = st.text_area("Key Points", "")
    tone = st.selectbox("Tone", ["professional", "friendly", "casual", "formal"])
    audience = st.text_input("Audience", "Data Scientists")

    generate_clicked = st.button("Generate Draft")

# ---------------------------------------------------
# STEP 2: GENERATE DRAFT
# ---------------------------------------------------

if generate_clicked:
    if not topic or not key_points:
        st.error("Please fill Topic and Key Points!")
    else:
        initial_state: PostState = {
            "info": {
                "topic": topic,
                "key_points": key_points,
                "tone": tone,
                "audience": audience
            },
            "draft": "",
            "human_feedback": None,
            "approved": False,
            "final_post": None
        }
        # Trigger workflow until it pauses for feedback
        graph.invoke(initial_state, config=config)
        st.session_state.state = graph.get_state(config)
        st.session_state.draft_ready = True
        st.session_state.awaiting_feedback = True
        st.rerun()

# ---------------------------------------------------
# STEP 3: SHOW DRAFT & ASK FOR FEEDBACK
# ---------------------------------------------------

if st.session_state.draft_ready:
    st.subheader("📝 Generated Draft")

    current_draft = st.session_state.state.values.get("draft", "")
    st.text_area("Generated Draft:", current_draft, height=200)

    st.write("### ⏳ Review the post")
    
    col1, col2 = st.columns(2)
    approve_btn = col1.button("✅ Approve")
    give_feedback_btn = col2.button("✏️ Provide Feedback")

    if approve_btn:
        # Resume workflow with approval
        interrupts = st.session_state.state.interrupts
        if interrupts:
            intr = interrupts[0]
            graph.invoke(Command(resume={intr.id: "approve"}), config=config)

        st.session_state.state = graph.get_state(config)
        st.session_state.approved = True
        st.session_state.awaiting_feedback = False
        st.rerun()

    if give_feedback_btn:
        st.session_state.show_feedback_box = True
        st.rerun()

# ---------------------------------------------------
# STEP 4: FEEDBACK ENTRY → REGENERATE
# ---------------------------------------------------

if st.session_state.get("show_feedback_box", False):
    st.subheader("💬 Provide Feedback")
    fb_text = st.text_area("Your Feedback", "")
    submit_fb = st.button("Submit Feedback")

    if submit_fb:
        interrupts = st.session_state.state.interrupts
        if interrupts:
            intr = interrupts[0]
            graph.invoke(Command(resume={intr.id: fb_text}), config=config)

        # After resume → draft regenerates → new interrupt
        st.session_state.state = graph.get_state(config)
        st.session_state.show_feedback_box = False
        st.rerun()

# ---------------------------------------------------
# STEP 5: APPROVED → POST TO LINKEDIN
# ---------------------------------------------------

if st.session_state.approved:
    st.success("Draft approved by human reviewer!")

    post_btn = st.button("🚀 Post to LinkedIn")

    if post_btn:
        st.info("Posting to LinkedIn ... Please wait.")
        time.sleep(0.5)

        # Resume workflow – posting happens here
        # No interrupt needed, workflow already in post_to_linkedin
        graph.invoke({}, config=config)
        st.session_state.state = graph.get_state(config)

        # Show results
        final = st.session_state.state.values
        if final.get("post_id"):
            st.success(f"🎉 Successfully posted to LinkedIn! Post ID: {final['post_id']}")
        else:
            st.error(f"Failed to post: {final.get('post_error')}")

        st.stop()
