from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

STOPWORDS = {
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
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    return [
        token
        for token in re.findall(r"[a-z0-9]{3,}", normalized)
        if token not in STOPWORDS
    ]


def text_for_article(article: dict) -> str:
    return f"{article.get('title', '')}\n{article.get('content', '')}".strip()


def heuristic_label(text: str) -> str:
    tokens = tokenize(text)
    counts = Counter(tokens)
    positive = sum(counts[term] * weight for term, weight in POSITIVE_TERMS.items())
    negative = sum(counts[term] * weight for term, weight in NEGATIVE_TERMS.items())

    if positive == 0 and negative == 0:
        return "neutro"
    if positive >= negative + 2:
        return "positivo"
    if negative >= positive + 2:
        return "negativo"
    return "neutro"


class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.class_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for token in tokenize(text):
                self.token_counts[label][token] += 1
                self.total_tokens[label] += 1
                self.vocabulary.add(token)

    def predict_one(self, text: str) -> tuple[str, float]:
        tokens = tokenize(text)
        total_docs = sum(self.class_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        scores: dict[str, float] = {}

        for label, class_count in self.class_counts.items():
            score = math.log(class_count / total_docs)
            denominator = self.total_tokens[label] + vocab_size
            for token in tokens:
                score += math.log((self.token_counts[label][token] + 1) / denominator)
            scores[label] = score

        best_label = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = 1 / (1 + math.exp(-min(margin, 20)))
        return best_label, confidence

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict_one(text) for text in texts]

    def to_dict(self) -> dict:
        return {
            "algorithm": "Multinomial Naive Bayes",
            "class_counts": dict(self.class_counts),
            "token_counts": {
                label: dict(counter)
                for label, counter in self.token_counts.items()
            },
            "total_tokens": dict(self.total_tokens),
            "vocabulary": sorted(self.vocabulary),
        }


def train_test_report(texts: list[str], labels: list[str], seed: int) -> dict:
    indexes = list(range(len(texts)))
    random.Random(seed).shuffle(indexes)
    split = max(1, int(len(indexes) * 0.8))
    train_indexes = indexes[:split]
    test_indexes = indexes[split:] or indexes[:]

    model = NaiveBayesClassifier()
    model.fit([texts[i] for i in train_indexes], [labels[i] for i in train_indexes])
    predictions = [model.predict_one(texts[i])[0] for i in test_indexes]

    labels_seen = sorted(set(labels))
    per_label = {}
    correct = 0
    for label in labels_seen:
        tp = sum(1 for i, pred in zip(test_indexes, predictions) if labels[i] == label and pred == label)
        fp = sum(1 for i, pred in zip(test_indexes, predictions) if labels[i] != label and pred == label)
        fn = sum(1 for i, pred in zip(test_indexes, predictions) if labels[i] == label and pred != label)
        support = sum(1 for i in test_indexes if labels[i] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }
    correct = sum(1 for i, pred in zip(test_indexes, predictions) if labels[i] == pred)
    return {
        "accuracy": round(correct / len(test_indexes), 3),
        "test_size": len(test_indexes),
        "labels": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="extracted_urls.json")
    parser.add_argument("--labels-csv", default="")
    parser.add_argument("--output-json", default="classified_articles.json")
    parser.add_argument("--output-csv", default="sentiment_dataset.csv")
    parser.add_argument("--model-out", default="sentiment_model.json")
    parser.add_argument("--report-out", default="sentiment_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    articles = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(articles, dict):
        articles = [articles]

    usable = [article for article in articles if not article.get("error") and text_for_article(article)]
    texts = [text_for_article(article) for article in usable]
    initial_labels = [heuristic_label(text) for text in texts]
    training_labels = initial_labels
    label_source = "heuristic"

    if args.labels_csv:
        label_rows = {}
        with resolve_path(args.labels_csv).open(encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                label = (row.get("human_label") or "").strip()
                url = (row.get("url") or "").strip()
                if url and label in {"positivo", "negativo", "neutro"}:
                    label_rows[url] = label
        training_labels = [
            label_rows.get(article.get("url", ""), initial_label)
            for article, initial_label in zip(usable, initial_labels)
        ]
        label_source = f"human_label from {args.labels_csv}"

    model = NaiveBayesClassifier()
    model.fit(texts, training_labels)
    predictions = model.predict(texts)

    classified = []
    for article, initial_label, training_label, (predicted_label, confidence) in zip(
        usable,
        initial_labels,
        training_labels,
        predictions,
    ):
        row = dict(article)
        row["heuristic_label"] = initial_label
        row["training_label"] = training_label
        row["sentiment"] = predicted_label
        row["sentiment_confidence"] = round(confidence, 3)
        classified.append(row)

    report = {
        "source": str(input_path),
        "label_source": label_source,
        "total_input": len(articles),
        "total_usable": len(usable),
        "label_distribution": dict(Counter(item["sentiment"] for item in classified)),
        "training_note": (
            "O modelo usa a coluna human_label quando um CSV rotulado e informado. "
            "Caso contrario, usa rotulos automaticos de polaridade ambiental."
        ),
        "validation_against_training_labels": train_test_report(texts, training_labels, args.seed),
    }

    resolve_path(args.output_json).write_text(
        json.dumps(classified, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolve_path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolve_path(args.model_out).write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2),
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
