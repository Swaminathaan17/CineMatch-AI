"""
Trains the overall-sentiment classifier: TF-IDF + Logistic Regression.

Run with the sample dataset (works offline, proves the pipeline is correct):
    python ml_training/train_sentiment.py

Run with the full IMDB 50k dataset once you've downloaded it in Replit
(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):
    python ml_training/train_sentiment.py --data data/imdb_50k.csv --review-col review --label-col sentiment

Saves:
    ml_engine/artifacts/sentiment_model.pkl
    ml_engine/artifacts/tfidf_vectorizer.pkl
    ml_training/sentiment_eval_report.txt   (accuracy/F1/confusion matrix - cite this in your report)
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "backend" / "ml_engine" / "artifacts"


def train(data_path: str, review_col: str, label_col: str, out_report: str, max_features: int):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[review_col, label_col])

    # Cross-validation instead of relying on a single train/test split - far
    # more honest when the dataset is small, since one split can get lucky
    # or unlucky. On the full 50k dataset this still runs fine, just slower.
    vectorizer_cv = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_cv = vectorizer_cv.fit_transform(df[review_col])
    cv_model = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(cv_model, X_cv, df[label_col], cv=5, scoring="accuracy")

    X_train, X_test, y_train, y_test = train_test_split(
        df[review_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACTS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    is_small_dataset = len(df) < 1000
    disclaimer = (
        f"\nNOTE: with only {len(df)} labeled examples, these numbers are a pipeline\n"
        f"sanity check, not a real accuracy claim. Retrain on the full IMDB 50k\n"
        f"dataset before citing numbers in your report.\n\n"
        if is_small_dataset
        else "\n"
    )

    report_text = (
        f"Dataset: {data_path} ({len(df)} rows, {X_train.shape[0]} train / {X_test.shape[0]} test)\n"
        f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n"
        f"Held-out test accuracy: {acc:.4f}\n"
        f"Held-out test weighted F1: {f1:.4f}\n"
        f"{disclaimer}"
        f"Classification report:\n{report}\n"
        f"Confusion matrix:\n{cm}\n"
    )
    Path(out_report).write_text(report_text)
    print(report_text)
    print(f"Model saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_reviews_labeled.csv")
    parser.add_argument("--review-col", default="review")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--report-out", default="ml_training/sentiment_eval_report.txt")
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()
    # small sample dataset needs far fewer features than the full 50k dataset
    # to avoid pure overfitting - auto-scale down when the dataset is tiny
    default_small = args.max_features if args.max_features != 5000 else None
    max_features = default_small
    if max_features is None:
        df_len = len(pd.read_csv(args.data))
        max_features = 5000 if df_len > 1000 else 100
    train(args.data, args.review_col, args.label_col, args.report_out, max_features)
