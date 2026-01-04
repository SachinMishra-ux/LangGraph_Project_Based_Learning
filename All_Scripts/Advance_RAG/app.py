# import uuid
# import streamlit as st
# from langchain_core.messages import HumanMessage, AIMessage
# from workflow import build_graph

# # ------------------------------------------------
# # STEP LABELS (NODE → UI TEXT)
# # ------------------------------------------------
# STEP_LABELS = {
#     "question_rewriter": "🧠 Understanding your question",
#     "question_classifier": "🔎 Checking relevance",
#     "retrieve": "📂 Searching financial documents",
#     "retrieval_grader": "✅ Validating retrieved documents",
#     "refine_question": "✏️ Improving the question",
#     "generate_answer": "🧾 Generating final answer",
#     "search_internet": "🌐 Searching the internet",
#     "off_topic_response": "🚫 Handling off-topic query",
# }

# # ------------------------------------------------
# # PAGE CONFIG
# # ------------------------------------------------
# st.set_page_config(
#     page_title="Finance RAG Bot",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # ------------------------------------------------
# # SESSION STATE SETUP
# # ------------------------------------------------
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())

# if "graph" not in st.session_state:
#     st.session_state.graph = build_graph()

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# graph = st.session_state.graph
# config = {"configurable": {"thread_id": st.session_state.thread_id}}

# # ------------------------------------------------
# # SIDEBAR (COLLAPSIBLE)
# # ------------------------------------------------
# with st.sidebar:
#     st.image(
#         "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
#         use_column_width=True
#     )

#     st.markdown("## 📂 Document Filters")

#     selected_roles = st.multiselect(
#         "Select document roles",
#         options=["analyst", "scientist", "financial"],
#         default=["analyst"]
#     )

#     st.caption(
#         "Only documents matching selected roles will be used during retrieval."
#     )

#     st.divider()

#     st.markdown("### ℹ️ About")
#     st.caption(
#         """
#         • Finance-focused RAG  
#         • LangGraph agent workflow  
#         • Role-based document filtering  
#         • Source attribution  
#         • SQLite memory
#         """
#     )

# # ------------------------------------------------
# # MAIN CHAT UI
# # ------------------------------------------------
# st.title("📊 Financial Assistant")
# st.caption("Finance-focused RAG powered by LangGraph")

# # Render chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

#         if msg["role"] == "assistant" and msg.get("sources"):
#             with st.expander("📚 Sources used"):
#                 for i, src in enumerate(msg["sources"], 1):
#                     st.markdown(
#                         f"""
#                         **{i}. {src['file_name']}**  
#                         Role: `{src['role']}`  
#                         Page: `{src.get('page', 'N/A')}`
#                         """
#                     )

# # ------------------------------------------------
# # CHAT INPUT
# # ------------------------------------------------
# user_input = st.chat_input("Ask a finance-related question...")

# if user_input:
#     # ----------------------------
#     # Store user message
#     # ----------------------------
#     st.session_state.messages.append(
#         {"role": "user", "content": user_input}
#     )

#     # Convert history to LangChain messages
#     lc_messages = []
#     for m in st.session_state.messages:
#         if m["role"] == "user":
#             lc_messages.append(HumanMessage(content=m["content"]))
#         else:
#             lc_messages.append(AIMessage(content=m["content"]))

#     input_state = {
#         "question": HumanMessage(content=user_input),
#         "messages": lc_messages,
#         "roles": selected_roles,
#     }

#     # ------------------------------------------------
#     # LIVE NODE EXECUTION TRACKING (THE KEY FIX)
#     # ------------------------------------------------
#     status_box = st.status("🚀 Starting workflow...", expanded=True)

#     for event in graph.stream(input=input_state, config=config):
#         event_type = event.get("event")
#         node_name = event.get("name")

#         if event_type == "on_node_start" and node_name:
#             label = STEP_LABELS.get(node_name, node_name)
#             status_box.update(
#                 label=label,
#                 state="running",
#                 expanded=True
#             )

#     status_box.update(
#         label="✅ Answer ready",
#         state="complete",
#         expanded=False
#     )

#     # ------------------------------------------------
#     # FETCH FINAL STATE
#     # ------------------------------------------------
#     state = graph.get_state(config)

#     final_answer = state.values["messages"][-1].content
#     sources = state.values.get("sources", [])

#     assistant_msg = {
#         "role": "assistant",
#         "content": final_answer,
#     }

#     if sources:
#         assistant_msg["sources"] = sources

#     st.session_state.messages.append(assistant_msg)

#     # Render assistant response immediately
#     with st.chat_message("assistant"):
#         st.markdown(final_answer)

#         if sources:
#             with st.expander("📚 Sources used"):
#                 for i, src in enumerate(sources, 1):
#                     st.markdown(
#                         f"""
#                         **{i}. {src['file_name']}**  
#                         Role: `{src['role']}`  
#                         Page: `{src.get('page', 'N/A')}`
#                         """
#                     )


########################################


### Debug: the nodoes ###


# import uuid
# import streamlit as st
# from langchain_core.messages import HumanMessage, AIMessage
# from workflow import build_graph

# # ------------------------------------------------
# # NODE → UI LABELS
# # ------------------------------------------------
# STEP_LABELS = {
#     "question_rewriter": "🧠 Understanding your question",
#     "question_classifier": "🔎 Checking relevance",
#     "retrieve": "📂 Searching financial documents",
#     "retrieval_grader": "✅ Validating retrieved documents",
#     "refine_question": "✏️ Improving the question",
#     "generate_answer": "🧾 Generating final answer",
#     "search_internet": "🌐 Searching the internet",
#     "off_topic_response": "🚫 Handling off-topic query",
# }

# # ------------------------------------------------
# # PAGE CONFIG
# # ------------------------------------------------
# st.set_page_config(
#     page_title="Finance RAG Bot",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # ------------------------------------------------
# # SESSION STATE
# # ------------------------------------------------
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())

# if "graph" not in st.session_state:
#     st.session_state.graph = build_graph()

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# graph = st.session_state.graph
# config = {"configurable": {"thread_id": st.session_state.thread_id}}

# # ------------------------------------------------
# # SIDEBAR
# # ------------------------------------------------
# with st.sidebar:
#     st.markdown("## 📂 Document Filters")

#     selected_roles = st.multiselect(
#         "Select document roles",
#         ["analyst", "scientist", "financial"],
#         default=["analyst"]
#     )

#     st.caption("Only selected roles are used for retrieval.")

# # ------------------------------------------------
# # MAIN UI
# # ------------------------------------------------
# st.title("📊 Financial Assistant")
# st.caption("Finance-focused RAG powered by LangGraph")

# # Render chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

#         if msg["role"] == "assistant" and msg.get("sources"):
#             with st.expander("📚 Sources used"):
#                 for i, src in enumerate(msg["sources"], 1):
#                     st.markdown(
#                         f"""
#                         **{i}. {src['file_name']}**  
#                         Role: `{src['role']}`  
#                         Page: `{src.get('page', 'N/A')}`
#                         """
#                     )

# # ------------------------------------------------
# # INPUT
# # ------------------------------------------------
# user_input = st.chat_input("Ask a finance-related question...")

# if user_input:
#     # Store user message
#     st.session_state.messages.append(
#         {"role": "user", "content": user_input}
#     )

#     # Convert history
#     lc_messages = [
#         HumanMessage(content=m["content"]) if m["role"] == "user"
#         else AIMessage(content=m["content"])
#         for m in st.session_state.messages
#     ]

#     input_state = {
#         "question": HumanMessage(content=user_input),
#         "messages": lc_messages,
#         "roles": selected_roles,
#     }

#     # ------------------------------------------------
#     # LIVE WORKFLOW TRACKING (CORRECT WAY)
#     # ------------------------------------------------
#     status_box = st.status("🚀 Starting workflow...", expanded=True)

#     debug_events = []
#     last_step = None

#     for event in graph.stream(input=input_state, config=config):
#         debug_events.append(event)

#         # LangGraph emits step/node names here
#         step = (
#             event.get("step")
#             or event.get("node")
#             or event.get("name")
#         )

#         if step and step != last_step:
#             label = STEP_LABELS.get(step, f"🔄 {step}")
#             status_box.update(
#                 label=label,
#                 state="running",
#                 expanded=True
#             )
#             last_step = step

#     status_box.update(
#         label="✅ Answer ready",
#         state="complete",
#         expanded=False
#     )

#     # ------------------------------------------------
#     # FETCH FINAL STATE
#     # ------------------------------------------------
#     state = graph.get_state(config)

#     final_answer = state.values["messages"][-1].content
#     sources = state.values.get("sources", [])

#     assistant_msg = {
#         "role": "assistant",
#         "content": final_answer,
#     }

#     if sources:
#         assistant_msg["sources"] = sources

#     st.session_state.messages.append(assistant_msg)

#     # Render assistant reply
#     with st.chat_message("assistant"):
#         st.markdown(final_answer)

#         if sources:
#             with st.expander("📚 Sources used"):
#                 for i, src in enumerate(sources, 1):
#                     st.markdown(
#                         f"""
#                         **{i}. {src['file_name']}**  
#                         Role: `{src['role']}`  
#                         Page: `{src.get('page', 'N/A')}`
#                         """
#                     )

#     # ------------------------------------------------
#     # DEBUG PANEL (OPTIONAL – REMOVE LATER)
#     # ------------------------------------------------
#     with st.expander("🛠 Debug: LangGraph raw events"):
#         st.write(debug_events)

###########################################

import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from workflow import build_graph

# ------------------------------------------------
# STEP LABELS (NODE → UI TEXT)
# ------------------------------------------------
STEP_LABELS = {
    "question_rewriter": "🧠 Understanding your question",
    "question_classifier": "🔎 Checking relevance",
    "retrieve": "📂 Searching financial documents",
    "retrieval_grader": "✅ Validating retrieved documents",
    "refine_question": "✏️ Improving the question",
    "generate_answer": "🧾 Generating final answer",
    "search_internet": "🌐 Searching the internet",
    "off_topic_response": "🚫 Handling off-topic query",
}

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Finance RAG Bot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------
# SESSION STATE SETUP
# ------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

graph = st.session_state.graph
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ------------------------------------------------
# SIDEBAR (COLLAPSIBLE)
# ------------------------------------------------
with st.sidebar:
    st.image(
        "/Users/sachinmishra/Desktop/LangGraph_Project_Based_Learning/All_Scripts/Advance_RAG/Workflow.png",
        use_column_width=True
    )

    st.markdown("## 📂 Document Filters")

    selected_roles = st.multiselect(
        "Select document roles",
        options=["analyst", "scientist", "financial"],
        default=["analyst"]
    )

    st.caption("Only documents matching selected roles will be used during retrieval.")

    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        """
        • Finance-focused RAG  
        • LangGraph agent workflow  
        • Role-based document filtering  
        • Source attribution  
        • Handles GB's of data
        • SQLite memory
        
        """
    )

# ------------------------------------------------
# MAIN CHAT UI
# ------------------------------------------------
st.title("📊 Financial Assistant")
st.caption("Finance-focused RAG powered by LangGraph")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""
                        **{i}. {src['file_name']}**  
                        Role: `{src['role']}`  
                        Page: `{src.get('page', 'N/A')}`
                        """
                    )

# ------------------------------------------------
# CHAT INPUT
# ------------------------------------------------
user_input = st.chat_input("Ask a finance-related question...")

if user_input:
    # ----------------------------
    # Store user message
    # ----------------------------
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Convert history to LangChain messages
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    input_state = {
        "question": HumanMessage(content=user_input),
        "messages": lc_messages,
        "roles": selected_roles,
    }

    # ------------------------------------------------
    # 🔴 REAL-TIME NODE EXECUTION TRACKING (FIXED)
    # ------------------------------------------------
    status_box = st.status("🚀 Starting workflow...", expanded=True)

    last_step = None
    debug_events = []  # optional (for debugging)

    for event in graph.stream(input=input_state, config=config):
        debug_events.append(event)

        # 🔑 LangGraph node name = top-level key
        node_name = list(event.keys())[0]

        if node_name != last_step:
            label = STEP_LABELS.get(node_name, f"🔄 {node_name}")

            status_box.update(
                label=label,
                state="running",
                expanded=True
            )

            last_step = node_name

    status_box.update(
        label="✅ Answer ready",
        state="complete",
        expanded=False
    )

    # ------------------------------------------------
    # FETCH FINAL STATE
    # ------------------------------------------------
    state = graph.get_state(config)

    final_answer = state.values["messages"][-1].content
    sources = state.values.get("sources", [])

    assistant_msg = {
        "role": "assistant",
        "content": final_answer,
    }

    if sources:
        assistant_msg["sources"] = sources

    st.session_state.messages.append(assistant_msg)

    # ------------------------------------------------
    # RENDER ASSISTANT RESPONSE
    # ------------------------------------------------
    with st.chat_message("assistant"):
        st.markdown(final_answer)

        if sources:
            with st.expander("📚 Sources used"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"""
                        **{i}. {src['file_name']}**  
                        Role: `{src['role']}`  
                        Page: `{src.get('page', 'N/A')}`
                        """
                    )

    # ------------------------------------------------
    # OPTIONAL: RAW DEBUG (UNCOMMENT IF NEEDED)
    # ------------------------------------------------
    # with st.expander("🛠 Debug: LangGraph raw events"):
    #     st.json(debug_events)
