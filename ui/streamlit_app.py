import streamlit as st
import requests

st.title(
    "Review Sentiment Analyzer (IndoBERT)"
)

st.subheader(
    "Sentiment Prediction"
)

review = st.text_area(
    "Masukkan Review:"
)

if st.button("Analyze Sentiment"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={
            "review": review
        }
    )

    result = response.json()

    st.success(
        f"Sentiment: {result['sentiment']}"
    )