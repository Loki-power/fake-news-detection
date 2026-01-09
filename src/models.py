import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "clean_data.csv"))

    # 🔥 IMPORTANT FIX
    df["text"] = df["text"].fillna("")

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "tfidf.pkl"))

    print("✅ Model and TF-IDF saved successfully")

if __name__ == "__main__":
    train_model()
