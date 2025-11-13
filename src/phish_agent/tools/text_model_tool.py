"""Heuristic text analysis tool for phishing detection."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TextIndicators:
    """Container describing which textual indicators were triggered."""

    keywords: List[str]
    credential_requests: List[str]
    urgency_phrases: List[str]


class TextModelTool:
    """Scores textual content for phishing indicators."""

    name = "text-model"

    #: Keyword fragments that often appear in phishing emails.
    _SUSPICIOUS_KEYWORDS: Iterable[str] = (
        "verify",
        "confirm",
        "reset",
        "password",
        "account",
        "update",
        "security alert",
        "suspend",
        "login",
        "bank",
        "billing",
    )

    #: Phrases explicitly asking the reader for credentials or sensitive data.
    _CREDENTIAL_REQUESTS: Iterable[str] = (
        "enter your password",
        "provide your password",
        "provide your credentials",
        "verify your account",
        "confirm your details",
        "reset your password",
        "update your billing information",
    )

    #: Urgency signalling meant to pressure the recipient.
    _URGENCY_PHRASES: Iterable[str] = (
        "immediately",
        "urgent",
        "within 24 hours",
        "as soon as possible",
        "final notice",
        "last warning",
    )

    def score_text(self, content: str) -> Dict[str, float | int | List[str]]:
        """Return heuristic scores for text content."""

        if not content:
            return {
                "text_score": 0.0,
                "keyword_hits": [],
                "credential_requests": [],
                "urgency_hits": [],
                "exclamation_count": 0,
            }

        lowered = content.lower()
        indicators = self._extract_indicators(lowered)
        keyword_score = min(len(indicators.keywords) / 3.0, 1.0)
        credential_score = 1.0 if indicators.credential_requests else 0.0
        urgency_score = min(len(indicators.urgency_phrases) / 2.0, 1.0)
        exclamation_count = content.count("!")
        exclamation_score = min(exclamation_count / 3.0, 1.0)

        score = (
            0.45 * keyword_score
            + 0.35 * credential_score
            + 0.15 * urgency_score
            + 0.05 * exclamation_score
        )

        return {
            "text_score": round(min(score, 1.0), 3),
            "keyword_hits": indicators.keywords,
            "credential_requests": indicators.credential_requests,
            "urgency_hits": indicators.urgency_phrases,
            "exclamation_count": exclamation_count,
        }

    def _extract_indicators(self, lowered_content: str) -> TextIndicators:
        """Collect matched indicators from the lowercase email body."""

        keywords = sorted(
            {phrase for phrase in self._SUSPICIOUS_KEYWORDS if phrase in lowered_content}
        )
        credential_requests = sorted(
            {
                phrase
                for phrase in self._CREDENTIAL_REQUESTS
                if phrase in lowered_content
            }
        )
        urgency_phrases = sorted(
            {phrase for phrase in self._URGENCY_PHRASES if phrase in lowered_content}
        )

        return TextIndicators(
            keywords=list(keywords),
            credential_requests=list(credential_requests),
            urgency_phrases=list(urgency_phrases),
        )
