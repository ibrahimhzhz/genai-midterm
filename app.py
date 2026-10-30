import os
os.environ.setdefault("HF_HOME", "/tmp/hf")   # writable model cache on cloud hosts
import streamlit as st
from transformers import pipeline

NAME = "Ibrahim Hasan Zubairi"
SID = "B040050iz"
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title=f"Sentiment Analyzer - {SID}", page_icon="💬")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model=MODEL)

st.title(f"Sentiment Analyzer — {NAME} ({SID})")
st.caption(f"Model: {MODEL}")

txt = st.text_area("Type a sentence or review:", height=120)

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Please type some text first.")
    else:
        with st.spinner("Analyzing..."):
            r = load_model()(txt, truncation=True)[0]
        label, score = r["label"], float(r["score"])
        if label == "POSITIVE":
            st.success(f"Sentiment: {label} 😊")
        else:
            st.error(f"Sentiment: {label} ☹️")
        st.metric("Confidence", f"{score:.1%}")
        st.progress(score, text=f"Confidence: {score:.1%}")
        if score < 0.60:
            st.info("Low confidence — the model is unsure, treat this result with care.")
