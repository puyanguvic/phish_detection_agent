"""Placeholder attachment analysis tool."""
from __future__ import annotations

from typing import Dict, Iterable


class AttachmentModelTool:
    """Examines file attachments for known phishing indicators."""

    name = "attachment-model"

    def score_attachments(self, attachments: Iterable[str]) -> Dict[str, float]:
        """Return mock attachment scores based on extensions."""

        attachments = list(attachments)
        if not attachments:
            return {"attachment_score": 0.0}
        risky_ext = {".exe", ".scr", ".js"}
        suspicious = sum(any(att.endswith(ext) for ext in risky_ext) for att in attachments)
        score = suspicious / len(attachments)
        return {"attachment_score": round(score, 3)}
