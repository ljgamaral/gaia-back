from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BACKEND_ROOT = Path(__file__).resolve().parent
VALID_LABELS = {"positivo", "negativo", "neutro"}

POSITIVE_TERMS = {
    "apreende": 2,
    "apreensao": 2,
    "combate": 2,
    "conservacao": 2,
    "fiscalizacao": 2,
    "preserva": 3,
    "preservacao": 3,
    "protege": 2,
    "protecao": 2,
    "recupera": 3,
    "recuperacao": 3,
    "reduz": 2,
    "reduzida": 2,
    "reduzido": 2,
    "reflorestamento": 3,
    "regenera": 3,
    "resgata": 2,
    "resgate": 2,
    "restauracao": 3,
    "sustentavel": 2,
    "sustentabilidade": 2,
}

NEGATIVE_TERMS = {
    "alagamento": 2,
    "ameaca": 2,
    "contamina": 3,
    "contaminacao": 3,
    "crime": 2,
    "desastre": 3,
    "desmatamento": 3,
    "destroi": 3,
    "destruicao": 3,
    "enchente": 3,
    "extincao": 3,
    "garimpo": 3,
    "ilegal": 2,
    "incendio": 3,
    "inundacao": 3,
    "morte": 2,
    "mortes": 2,
    "poluicao": 3,
    "queimada": 3,
    "queimadas": 3,
    "risco": 2,
    "seca": 3,
    "temporal": 2,
}

STOPWORDS = [
    "a",
    "as",
    "ao",
    "aos",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "que",
    "um",
    "uma",
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    backend_path = BACKEND_ROOT / path
    if backend_path.exists():
        return backend_path
    return BACKEND_ROOT.parent / path


def tokenize_for_heuristic(text: str) -> list[str]:
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return [
        token
        for token in re.findall(r"[a-z0-9]{3,}", normalized)
        if token not in STOPWORDS
    ]


def text_for_article(article: dict) -> str:
    return f"{article.get('title', '')}\n{article.get('content', '')}".strip()


def heuristic_label(text: str) -> str:
    counts = Counter(tokenize_for_heuristic(text))
    positive = sum(counts[term] * weight for term, weight in POSITIVE_TERMS.items())
    negative = sum(counts[term] * weight for term, weight in NEGATIVE_TERMS.items())

    if positive == 0 and negative == 0:
        return "neutro"
    if positive >= negative + 2:
        return "positivo"
    if negative >= positive + 2:
        return "negativo"
    return "neutro"


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b[a-z0-9]{3,}\b",
                    stop_words=STOPWORDS,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]
    )


def load_articles(path: Path) -> list[dict]:
    articles = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(articles, dict):
        return [articles]
    if isinstance(articles, list):
        return articles
    raise ValueError(f"Formato invalido em {path}")


def load_human_labels(path: str) -> dict[str, str]:
    if not path:
        return {}

    labels: dict[str, str] = {}
    with resolve_path(path).open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            url = (row.get("url") or "").strip()
            label = (row.get("human_label") or "").strip()
            if url and label in VALID_LABELS:
                labels[url] = label
    return labels


def make_report(labels: list[str], predictions: list[str]) -> dict:
    report = classification_report(
        labels,
        predictions,
        labels=sorted(VALID_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(accuracy_score(labels, predictions), 3),
        "test_size": len(labels),
        "labels": {
            label: {
                "precision": round(metrics["precision"], 3),
                "recall": round(metrics["recall"], 3),
                "f1": round(metrics["f1-score"], 3),
                "support": int(metrics["support"]),
            }
            for label, metrics in report.items()
            if label in VALID_LABELS
        },
    }


def prediction_confidences(model: Pipeline, texts: list[str]) -> list[float]:
    if not hasattr(model, "predict_proba"):
        return [0.0 for _ in texts]
    probabilities = model.predict_proba(texts)
    return [round(float(row.max()), 3) for row in probabilities]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="extracted_urls.json")
    parser.add_argument("--labels-csv", default="")
    parser.add_argument("--output-json", default="classified_articles.json")
    parser.add_argument("--output-csv", default="sentiment_dataset.csv")
    parser.add_argument("--model-out", default="sentiment_model.pkl")
    parser.add_argument("--report-out", default="sentiment_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    articles = load_articles(input_path)
    usable = [article for article in articles if not article.get("error") and text_for_article(article)]
    texts = [text_for_article(article) for article in usable]
    initial_labels = [heuristic_label(text) for text in texts]
    human_labels = load_human_labels(args.labels_csv)
    training_labels = [
        human_labels.get(article.get("url", ""), initial_label)
        for article, initial_label in zip(usable, initial_labels)
    ]
    label_source = f"human_label from {args.labels_csv}" if human_labels else "heuristic"

    stratify = training_labels if min(Counter(training_labels).values()) > 1 else None
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        training_labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )

    validation_model = build_model()
    validation_model.fit(train_texts, train_labels)
    test_predictions = validation_model.predict(test_texts).tolist()

    model = build_model()
    model.fit(texts, training_labels)
    predictions = model.predict(texts).tolist()
    confidences = prediction_confidences(model, texts)

    classified = []
    for article, initial_label, training_label, predicted_label, confidence in zip(
        usable,
        initial_labels,
        training_labels,
        predictions,
        confidences,
    ):
        row = dict(article)
        row["heuristic_label"] = initial_label
        row["training_label"] = training_label
        row["sentiment"] = predicted_label
        row["sentiment_confidence"] = confidence
        classified.append(row)

    report = {
        "source": str(input_path),
        "algorithm": "scikit-learn Pipeline(CountVectorizer, MultinomialNB)",
        "label_source": label_source,
        "total_input": len(articles),
        "total_usable": len(usable),
        "label_distribution": dict(Counter(predictions)),
        "training_note": (
            "O modelo usa a coluna human_label quando um CSV rotulado e informado. "
            "Caso contrario, usa rotulos automaticos de polaridade ambiental."
        ),
        "validation_against_training_labels": make_report(test_labels, test_predictions),
    }

    with resolve_path(args.model_out).open("wb") as model_file:
        pickle.dump(model, model_file)

    resolve_path(args.output_json).write_text(
        json.dumps(classified, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolve_path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with resolve_path(args.output_csv).open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "url",
            "title",
            "published_at",
            "heuristic_label",
            "training_label",
            "sentiment",
            "sentiment_confidence",
            "text",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for article in classified:
            writer.writerow(
                {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "published_at": article.get("published_at", ""),
                    "heuristic_label": article.get("heuristic_label", ""),
                    "training_label": article.get("training_label", ""),
                    "sentiment": article.get("sentiment", ""),
                    "sentiment_confidence": article.get("sentiment_confidence", ""),
                    "text": text_for_article(article),
                }
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
