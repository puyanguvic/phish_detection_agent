"""Placeholder text analysis tool."""
from __future__ import annotations

from typing import Dict


class TextModelTool:
    """Scores textual content for phishing indicators."""

    name = "text-model"

    def score_text(self, content: str) -> Dict[str, float]:
        """Return heuristic scores for text content."""

        if not content:
            return {"text_score": 0.0}
        # Placeholder scoring: length-normalized heuristic.
        normalized = min(len(content) / 1000.0, 1.0)
        return {"text_score": round(normalized, 3)}
