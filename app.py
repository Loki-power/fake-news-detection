import streamlit as st
import joblib
from src.preprocessing import clean_text

model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")

st.title("📰 Fake News Detection")

text = st.text_area("Enter News Article")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()

        if prediction == 0:
            st.error(f"❌ Fake News (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {confidence:.2f})")
