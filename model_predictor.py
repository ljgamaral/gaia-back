from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_ROOT / "sentiment_model.pkl"


@lru_cache(maxsize=1)
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_sentiment(text: str) -> tuple[str, float]:
    model = load_model()
    label = str(model.predict([text])[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = round(float(probabilities.max()), 3)

    return label, confidence
