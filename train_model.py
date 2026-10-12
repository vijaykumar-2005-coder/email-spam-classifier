
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Find dataset
dataset_files = list(DATA_DIR.rglob("SMSSpamCollection"))

if not dataset_files:
    raise FileNotFoundError("Dataset not found!")

# Load dataset
df = pd.read_csv(
    dataset_files[0],
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Clean data
df.dropna(subset=["label", "message"], inplace=True)
df.drop_duplicates(inplace=True)
df = df[df["label"].isin(["ham", "spam"])].copy()

# Separate features and labels
X = df["message"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Text vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Linear SVM
model = LinearSVC(random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate model
predictions = model.predict(X_test_tfidf)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, pos_label="spam"
    ),
    "recall": recall_score(
        y_test, predictions, pos_label="spam"
    ),
    "f1": f1_score(
        y_test, predictions, pos_label="spam"
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ).tolist(),
    "total_messages": len(df),
    "spam_count": int((df["label"] == "spam").sum()),
    "ham_count": int((df["label"] == "ham").sum())
}

# Save model, vectorizer and metrics
joblib.dump(model, MODEL_DIR / "spam_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
joblib.dump(metrics, MODEL_DIR / "metrics.pkl")

print("=" * 50)
print("EMAIL SPAM CLASSIFIER - MODEL TRAINING")
print("=" * 50)

print("\nModel: Linear SVM")
print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Spam Precision: {metrics['precision'] * 100:.2f}%")
print(f"Spam Recall: {metrics['recall'] * 100:.2f}%")
print(f"Spam F1-score: {metrics['f1'] * 100:.2f}%")

print("\nModel files saved successfully!")
print("Location:", MODEL_DIR)
