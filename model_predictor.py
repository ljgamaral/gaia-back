from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
SENTIMENT_MODEL_PATH = BACKEND_ROOT / "sentiment_model.pkl"
TOPIC_MODEL_PATH = BACKEND_ROOT / "topic_model.pkl"


@lru_cache(maxsize=1)
def load_sentiment_model():
    with SENTIMENT_MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@lru_cache(maxsize=1)
def load_topic_model():
    with TOPIC_MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_with_model(model, text: str) -> tuple[str, float]:
    label = str(model.predict([text])[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = round(float(probabilities.max()), 3)

    return label, confidence


def predict_sentiment(text: str) -> tuple[str, float]:
    return predict_with_model(load_sentiment_model(), text)


def predict_topic(text: str) -> tuple[str, float]:
    return predict_with_model(load_topic_model(), text)
