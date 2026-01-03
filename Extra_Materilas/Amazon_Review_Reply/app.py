# app.py
import streamlit as st
from script import analyze_review

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="🛒",
    layout="centered"
)

# =========================
# Styling
# =========================
st.markdown("""
<style>
.review-box {
    background-color: #f3f3f3;
    padding: 16px;
    border-radius: 8px;
    border-left: 5px solid #ff9900;
}
.response-box {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 8px;
    border-left: 5px solid #007185;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: bold;
    color: white;
}
.positive { background-color: #2ecc71; }
.negative { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("## 🛒 Amazon Review Intelligence System")
st.caption("AI-powered sentiment analysis & automated seller response")

# =========================
# Input
# =========================
review = st.text_area(
    "Customer Review",
    placeholder="Paste an Amazon product review here...",
    height=140
)

analyze = st.button("Analyze Review")

# =========================
# Processing
# =========================
if analyze and review.strip():
    with st.spinner("Analyzing customer feedback..."):
        result = analyze_review(review)

    # --- Review ---
    st.markdown("### 📝 Customer Review")
    st.markdown(f"<div class='review-box'>{review}</div>", unsafe_allow_html=True)

    # --- Sentiment ---
    sentiment = result["sentiment"]
    badge_class = "positive" if sentiment == "positive" else "negative"
    st.markdown(
        f"### 📊 Sentiment "
        f"<span class='badge {badge_class}'>{sentiment.upper()}</span>",
        unsafe_allow_html=True
    )

    # --- Diagnosis ---
    if sentiment == "negative":
        d = result["diagnosis"]
        st.markdown("### 🧠 Issue Diagnosis")
        st.json(d)

    # --- Response ---
    st.markdown("### 💬 Suggested Seller Response")
    st.markdown(
        f"<div class='response-box'>{result['response']}</div>",
        unsafe_allow_html=True
    )

elif analyze:
    st.warning("Please enter a customer review.")
