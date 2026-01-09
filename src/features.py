import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features():
    df = pd.read_csv("data/processed/clean_data.csv")

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    joblib.dump(vectorizer, "tfidf.pkl")
    return X_tfidf, y

if __name__ == "__main__":
    extract_features()
