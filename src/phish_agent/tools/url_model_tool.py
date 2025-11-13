"""Heuristic URL analysis tool."""
from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Dict, Iterable, List


class UrlModelTool:
    """Evaluates URLs extracted from an email."""

    name = "url-model"

    _SUSPICIOUS_KEYWORDS = (
        "login",
        "verify",
        "reset",
        "account",
        "secure",
        "update",
        "billing",
    )
    _SUSPICIOUS_TLDS = {"ru", "cn", "tk", "top", "xyz", "zip"}

    def score_urls(self, urls: Iterable[str]) -> Dict[str, float | int | List[str]]:
        """Return scores derived from URL heuristics."""

        url_list = [url for url in urls if url]
        if not url_list:
            return {"url_score": 0.0, "url_count": 0, "flagged_urls": []}

        total_score = 0.0
        flagged: List[str] = []
        for url in url_list:
            suspicion = self._score_single_url(url)
            total_score += suspicion
            if suspicion >= 0.5:
                flagged.append(url)

        average_score = round(total_score / len(url_list), 3)
        return {
            "url_score": average_score,
            "url_count": len(url_list),
            "flagged_urls": flagged,
        }

    def _score_single_url(self, url: str) -> float:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        lowered = url.lower()

        suspicion = 0.0
        if parsed.scheme == "http":
            suspicion += 0.3

        if any(keyword in lowered for keyword in self._SUSPICIOUS_KEYWORDS):
            suspicion += 0.3

        if self._looks_like_ip(hostname):
            suspicion += 0.2

        if hostname.count(".") >= 3:
            suspicion += 0.1

        if hostname.split(".")[-1] in self._SUSPICIOUS_TLDS:
            suspicion += 0.15

        if len(parsed.path) > 30 or len(lowered) > 80:
            suspicion += 0.1

        return min(suspicion, 1.0)

    def _looks_like_ip(self, host: str) -> bool:
        return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))
