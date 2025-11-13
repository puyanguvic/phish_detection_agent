"""Placeholder URL analysis tool."""
from __future__ import annotations

from typing import Dict, Iterable


class UrlModelTool:
    """Evaluates URLs extracted from an email."""

    name = "url-model"

    def score_urls(self, urls: Iterable[str]) -> Dict[str, float]:
        """Return scores derived from URL heuristics."""

        urls = list(urls)
        if not urls:
            return {"url_score": 0.0}
        suspicious = sum("login" in url or url.startswith("http://") for url in urls)
        score = suspicious / len(urls)
        return {"url_score": round(score, 3)}
