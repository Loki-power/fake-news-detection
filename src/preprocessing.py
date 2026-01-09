import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def preprocess_and_save():
    fake = pd.read_csv("data/raw/Fake.csv")
    true = pd.read_csv("data/raw/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)

    df = df[["text", "label"]]
    df.to_csv("data/processed/clean_data.csv", index=False)

if __name__ == "__main__":
    preprocess_and_save()
