from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {"positivo", "negativo", "neutro"}

NEGATIVE_PATTERNS = [
    r"\bdesmat",
    r"\bdevast",
    r"\bgarimpo\b",
    r"\bgarimpo ilegal\b",
    r"\bmercurio\b",
    r"\bpoluic",
    r"\bcontamin",
    r"\besgoto\b",
    r"\bcrime ambiental\b",
    r"\bincendio\b",
    r"\bqueimada",
    r"\bfumaca\b",
    r"\bseca\b",
    r"\bestiagem\b",
    r"\benchente\b",
    r"\balagamento",
    r"\binundac",
    r"\bcheia\b",
    r"\btemporal\b",
    r"\bciclone\b",
    r"\bchuva forte\b",
    r"\bdeslizamento\b",
    r"\bmorte de (animal|peixe|ave|onca|boto)",
    r"\banimais mortos\b",
    r"\brisco\b",
    r"\bameac",
    r"\bemergencia\b",
    r"\balerta\b",
    r"\bcrise climatica\b",
]

POSITIVE_PATTERNS = [
    r"\bpreserv",
    r"\bconserv",
    r"\brecuper",
    r"\breflorest",
    r"\brestaur",
    r"\bregener",
    r"\bsustentavel\b",
    r"\bsustentabilidade\b",
    r"\bresgata",
    r"\bresgate\b",
    r"\breintroduz",
    r"\bsolta",
    r"\bdevolvid[ao]s? a natureza\b",
    r"\bfiscaliz",
    r"\bapreend",
    r"\bmulta\b",
    r"\bcombate\b",
    r"\boperacao .* combate\b",
    r"\boperacao .* contra\b",
    r"\bprotec",
    r"\bparque nacional\b",
    r"\bunidade de conservacao\b",
    r"\benergia limpa\b",
    r"\breducao\b",
    r"\breduz",
]

NEUTRAL_PATTERNS = [
    r"\bprevisao do tempo\b",
    r"\bconfira previsao\b",
    r"\bveja previsao\b",
    r"\bdeve chover\b",
    r"\bfrente fria\b",
    r"\btemperatura\b",
    r"\bcalor\b",
    r"\bfrio\b",
    r"\bexplica\b",
    r"\bsaiba\b",
    r"\bconheca\b",
    r"\bentenda\b",
]

FALSE_ENVIRONMENTAL_PATTERNS = [
    r"\barena pantanal\b",
    r"\bavenida pantanal\b",
    r"\bbairro pantanal\b",
    r"\bshopping\b",
    r"\bshow\b",
    r"\bcarnaval\b",
    r"\bfutebol\b",
    r"\bestadio\b",
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / value


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def score_patterns(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def label_row(row: dict) -> tuple[str, str]:
    title = normalize(row.get("title", ""))
    text = normalize(f"{row.get('title', '')}\n{row.get('text', '')}")

    false_environment = score_patterns(title, FALSE_ENVIRONMENTAL_PATTERNS)
    positive = score_patterns(text, POSITIVE_PATTERNS)
    negative = score_patterns(text, NEGATIVE_PATTERNS)
    neutral = score_patterns(title, NEUTRAL_PATTERNS)

    if false_environment and not negative and not positive:
        return "neutro", "referencia ambiental indireta/local"

    if negative >= positive + 1:
        return "negativo", "impacto ambiental, risco, desastre ou crime"

    if positive >= negative + 1:
        return "positivo", "acao de preservacao, recuperacao, fiscalizacao ou resgate"

    if neutral:
        return "neutro", "informativo/previsao sem polaridade ambiental clara"

    fallback = row.get("sentiment") or row.get("heuristic_label") or "neutro"
    if fallback not in VALID_LABELS:
        fallback = "neutro"
    return fallback, "mantido do classificador inicial por falta de evidência forte"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sentiment_dataset.csv")
    parser.add_argument("--output", default="sentiment_dataset_rotulado.csv")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    with input_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
        fieldnames = list(rows[0].keys()) if rows else []

    for extra in ["human_label", "label_reason"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    for row in rows:
        label, reason = label_row(row)
        row["human_label"] = label
        row["label_reason"] = reason

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(dict(Counter(row["human_label"] for row in rows)))
    print(output_path)


if __name__ == "__main__":
    main()
