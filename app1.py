# app_final.py
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, string
from nltk.corpus import stopwords

# ----------------------------
# 1️⃣ Page config and theme toggle
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analysis: NB vs LSTM",
    layout="wide"
)

st.sidebar.title("Settings")
theme = st.sidebar.radio("Choose Theme:", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        .main {background-color: #0E1117; color: #FAFAFA;}
        .sidebar .sidebar-content {background-color: #1E1E1E;}
        </style>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# 2️⃣ Load saved models and tokenizer
# ----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model_lstm = load_model("lstm_model.keras")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# ----------------------------
# 3️⃣ Preprocessing function
# ----------------------------
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words - {"not", "no", "never"}

def clean_text_fast(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in custom_stopwords]
    return " ".join(tokens)

# ----------------------------
# 4️⃣ Prediction functions
# ----------------------------
def predict_lstm(review, max_len=100):
    cleaned = clean_text_fast(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq_pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = (model_lstm.predict(seq_pad) > 0.5).astype(int).flatten()[0]
    return "Positive" if pred == 1 else "Negative"

def predict_nb(review):
    cleaned = clean_text_fast(review)
    vect = tfidf.transform([cleaned])
    pred = nb_model.predict(vect)[0]
    return pred.capitalize()

# ----------------------------
# 5️⃣ Streamlit interface
# ----------------------------
st.title("Sentiment Analysis: Naive Bayes vs LSTM")
st.write("Enter a product review and see predictions from both models:")

user_input = st.text_area("Type your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        lstm_sentiment = predict_lstm(user_input)
        nb_sentiment = predict_nb(user_input)
        st.success(f"LSTM Prediction: {lstm_sentiment}")
        st.success(f"Naive Bayes Prediction: {nb_sentiment}")





