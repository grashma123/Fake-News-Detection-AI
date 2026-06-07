import pandas as pd
import re
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -------------------------
# Load dataset
# -------------------------
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0)

# shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# -------------------------
# Clean text (FIXED VERSION)
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)


# -------------------------
# Features & labels
# -------------------------
X = data["content"]
y = data["label"]


# -------------------------
# TF-IDF (IMPROVED)
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

print("TF-IDF Shape:", X.shape)


# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Model (stable baseline)
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------
# Save model
# -------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel saved successfully!")