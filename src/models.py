import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model():
    df = pd.read_csv("data/processed/clean_data.csv")

    X = df["text"]
    y = df["label"]

    vectorizer = joblib.load("tfidf.pkl")
    X_tfidf = vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    return X_test, y_test

if __name__ == "__main__":
    train_model()
