
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Find the dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
dataset_files = list(DATA_DIR.rglob("SMSSpamCollection"))

if not dataset_files:
    raise FileNotFoundError(
        "SMSSpamCollection not found inside the data folder."
    )

# 2. Load and clean data
df = pd.read_csv(
    dataset_files[0],
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

df.dropna(subset=["label", "message"], inplace=True)
df.drop_duplicates(inplace=True)
df = df[df["label"].isin(["ham", "spam"])].copy()

X = df["message"]
y = df["label"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Define the models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Linear SVM": LinearSVC(random_state=42)
}

results = []
trained_models = {}

# 6. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "Spam Precision (%)": round(
            precision_score(
                y_test, predictions,
                pos_label="spam", zero_division=0
            ) * 100, 2
        ),
        "Spam Recall (%)": round(
            recall_score(
                y_test, predictions,
                pos_label="spam", zero_division=0
            ) * 100, 2
        ),
        "Spam F1 (%)": round(
            f1_score(
                y_test, predictions,
                pos_label="spam", zero_division=0
            ) * 100, 2
        )
    })

    trained_models[name] = model

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        zero_division=0
    ))

# 7. Compare the models
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Select the model with the highest spam F1-score
best_name = results_df.sort_values(
    by="Spam F1 (%)",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_name]

print(f"\nBest model by spam F1-score: {best_name}")

# 8. Test a custom message
message = input("\nEnter a message to classify: ")
message_tfidf = vectorizer.transform([message])
prediction = best_model.predict(message_tfidf)[0]

print(f"\nPrediction: {prediction.upper()}")
