import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

nltk.download("stopwords")

def clean_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]

    return " ".join(words)


def preprocess_and_save():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    fake = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "Fake.csv"))
    true = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "True.csv"))

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    df["text"] = df["text"].apply(clean_text)

    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df[["text", "label"]].to_csv(
        os.path.join(processed_dir, "clean_data.csv"),
        index=False
    )

    print("✅ clean_data.csv created successfully")


if __name__ == "__main__":
    preprocess_and_save()
