"""
Loads the trained TF-IDF + Logistic Regression artifacts and exposes simple
predict functions. If artifacts don't exist yet (train_sentiment.py hasn't
been run), raises a clear error rather than failing silently or faking output.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass
class SentimentResult:
    label: str          # "positive" | "negative"
    confidence: float    # probability of the predicted class, 0-1


class SentimentModelNotTrainedError(Exception):
    pass


class SentimentModel:
    def __init__(self):
        self._model = None
        self._vectorizer = None

    def _load(self):
        if self._model is not None:
            return
        model_path = ARTIFACTS_DIR / "sentiment_model.pkl"
        vec_path = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
        if not model_path.exists() or not vec_path.exists():
            raise SentimentModelNotTrainedError(
                "Sentiment model artifacts not found. Run "
                "'python ml_training/train_sentiment.py' first."
            )
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(vec_path, "rb") as f:
            self._vectorizer = pickle.load(f)

    def predict(self, text: str) -> SentimentResult:
        self._load()
        vec = self._vectorizer.transform([text])
        label = self._model.predict(vec)[0]
        proba = self._model.predict_proba(vec)[0]
        class_index = list(self._model.classes_).index(label)
        confidence = float(proba[class_index])
        return SentimentResult(label=label, confidence=round(confidence, 4))

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        self._load()
        if not texts:
            return []
        vecs = self._vectorizer.transform(texts)
        labels = self._model.predict(vecs)
        probas = self._model.predict_proba(vecs)
        results = []
        for label, proba in zip(labels, probas):
            class_index = list(self._model.classes_).index(label)
            results.append(
                SentimentResult(label=label, confidence=round(float(proba[class_index]), 4))
            )
        return results


sentiment_model = SentimentModel()
