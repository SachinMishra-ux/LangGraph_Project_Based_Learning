# app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from script import run_chat

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.chat-container {
    max-width: 750px;
    margin: auto;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
}
.ai-msg {
    background-color: #F1F0F0;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Header
# =========================
st.title("🤖 LangGraph Tool-Powered Assistant")
st.caption("Search • Calculator • Stock Prices • Multi-tool reasoning")

# =========================
# Chat History
# =========================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f'<div class="user-msg">{msg.content}</div>', unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f'<div class="ai-msg">{msg.content}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Input Box
# =========================
user_input = st.chat_input("Ask anything...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        updated_messages = run_chat(st.session_state.messages)

    st.session_state.messages = updated_messages
    st.rerun()
