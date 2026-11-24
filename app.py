import html
import pickle
import re
from pathlib import Path

import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.pkl"


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰")
st.title("📰 News Sentiment Analyzer")

try:
    artifact = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

pipeline = artifact["pipeline"]

title = st.text_input("News title", placeholder="Example: Company reports strong growth")
description = st.text_area("News description", height=190)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    combined = f"{clean_text(title)} . {clean_text(description)}".strip()
    if not combined.strip(" ."):
        st.warning("Please enter a title, a description, or both.")
    else:
        prediction = str(pipeline.predict([combined])[0])
        st.success(f"Predicted sentiment: **{prediction.title()}**")