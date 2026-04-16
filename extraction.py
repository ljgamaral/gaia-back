from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    root_path = PROJECT_ROOT / value
    if root_path.exists():
        return root_path
    return path


def resolve_output_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return PROJECT_ROOT / value


def extract_g1(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    og_title = soup.find("meta", attrs={"property": "og:title"})
    h1 = soup.find("h1")
    title = (og_title.get("content") if og_title else "") or (
        h1.get_text(" ", strip=True) if h1 else ""
    )

    meta_date = soup.find("meta", attrs={"property": "article:published_time"})
    time_tag = soup.find("time")
    published_at = (
        (meta_date.get("content") if meta_date else "")
        or (time_tag.get("datetime") if time_tag else "")
        or ""
    )

    article = soup.find("article") or soup
    paragraphs = []
    for paragraph in article.find_all("p"):
        text = paragraph.get_text(" ", strip=True)
        if text and len(text) >= 30:
            paragraphs.append(text)

    return {
        "url": url,
        "title": title.strip(),
        "published_at": published_at.strip(),
        "content": "\n".join(paragraphs).strip(),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }


def is_valid_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_urls(urls: Iterable[str] | None = None, urls_file: str = "urls.txt") -> list[str]:
    if urls:
        normalized = [str(url).strip() for url in urls if str(url).strip()]
        invalid = [url for url in normalized if not is_valid_url(url)]
        if invalid:
            raise ValueError(f"URLs inválidas: {', '.join(invalid)}")
        return normalized

    path = resolve_input_path(urls_file)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de URLs não encontrado: {urls_file}")

    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        if not is_valid_url(first):
            raise ValueError(f"URL inválida em {urls_file}: {first}")
        loaded.append(first)

    if not loaded:
        raise ValueError(f"Nenhuma URL válida encontrada em {urls_file}")
    return loaded


def extract_many(
    urls: Iterable[str],
    continue_on_error: bool = True,
    timeout_seconds: float = 10,
    max_workers: int = 20,
) -> list[dict]:
    url_list = [str(url) for url in urls]
    worker_count = max(1, min(max_workers, len(url_list) or 1))
    timeout = max(0.1, timeout_seconds)

    def fetch_one(client: httpx.Client, url: str) -> dict:
        try:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return extract_g1(response.text, url)
        except Exception as exc:
            if not continue_on_error:
                raise
            return {
                "url": url,
                "error": str(exc),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

    with httpx.Client(
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
    ) as client:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(lambda url: fetch_one(client, url), url_list))


def save_articles(items: list[dict], out: str) -> None:
    data: object = items[0] if len(items) == 1 else items
    resolve_output_path(out).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_articles(path: str) -> list[dict]:
    raw = json.loads(resolve_input_path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Formato inválido em {path}")


def article_from_raw_text(raw_text: str, title: str) -> dict:
    text = raw_text.strip()
    if not text:
        raise ValueError("Texto bruto vazio.")
    return {
        "url": "",
        "title": title.strip() or "Texto bruto informado pelo usuário",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "content": text,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "raw_text",
    }
