import streamlit as st
import tensorflow as tf
import numpy as np
import json
import re
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from nltk.corpus import stopwords

# -----------------------------------------
# LOAD STOPWORDS
# -----------------------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# -----------------------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json") as f:
        return tokenizer_from_json(json.load(f))

model = load_model()
tokenizer = load_tokenizer()

MAX_LEN = 25

# -----------------------------------------
# CLEANING + ENCODING
# -----------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

def encode_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return padded

# -----------------------------------------
# SENTIMENT PREDICTION
# -----------------------------------------
def predict_sentiment(text):
    padded = encode_text(text)
    probs = model.predict(padded)[0]

    negative = float(probs[0] * 100)
    positive = float(probs[1] * 100)
    neutral  = float(probs[2] * 100)

    return {
        "positive": round(positive, 2),
        "negative": round(negative, 2),
        "neutral": round(neutral, 2)
    }

# -----------------------------------------
# UI
# -----------------------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🌟 Advanced Sentiment Analyzer (LSTM + Custom Data)")

st.write("Supports: **positivity**, **negativity**, **neutrality**, **negation**, **sarcasm**, and **emoji sentiment**.")

text = st.text_area("Enter your text here:", height=150)

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        scores = predict_sentiment(text)

        st.subheader("📊 Results")
        st.write(f"**Positive:** {scores['positive']}%")
        st.write(f"**Negative:** {scores['negative']}%")
        st.write(f"**Neutral:** {scores['neutral']}%")

        st.write("### 🔍 Probability Bars")
        st.progress(int(scores["positive"]))
        st.progress(int(scores["negative"]))
        st.progress(int(scores["neutral"]))

        # Final verdict
        label = max(scores, key=scores.get)

        if label == "positive":
            st.success("🙂 The sentiment is **POSITIVE**.")
        elif label == "negative":
            st.error("☹️ The sentiment is **NEGATIVE**.")
        else:
            st.info("😐 The sentiment is **NEUTRAL**.")
