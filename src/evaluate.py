import joblib
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate_model():
    df = pd.read_csv("data/processed/clean_data.csv")

    X = df["text"]
    y = df["label"]

    vectorizer = joblib.load("tfidf.pkl")
    model = joblib.load("model.pkl")

    X_tfidf = vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
