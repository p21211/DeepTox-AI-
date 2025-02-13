import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_tfidf():
    return pickle.load(open("tf_idf.pkt", "rb"))


def load_model():
    return pickle.load(open("toxicity_model.pkt", "rb"))


def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name


st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .input-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .text-box {
            width: 60%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .submit-btn {
            background-color: #008CBA;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #005f73;
        }
        .result-toxic {
            font-size: 22px;
            font-weight: bold;
            color: red;
            text-align: center;
        }
        .result-non-toxic {
            font-size: 22px;
            font-weight: bold;
            color: green;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<p class="title"> DeepTox AI App</p>', unsafe_allow_html=True)


st.markdown('<div class="input-container">', unsafe_allow_html=True)
text_input = st.text_area("Enter your text here:", height=100)
st.markdown("</div>", unsafe_allow_html=True)


submit = st.button("🔍 Submit")


if submit and text_input.strip():
    with st.spinner("Analyzing... Please wait"):
        result = toxicity_prediction(text_input)

    if result == "Toxic":
        st.markdown(f'<p class="result-toxic">🚨 Result: {result}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="result-non-toxic">✅ Result: {result}</p>', unsafe_allow_html=True)
