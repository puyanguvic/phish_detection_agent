"""Placeholder image analysis tool."""
from __future__ import annotations

from typing import Dict, Iterable


class ImageModelTool:
    """Analyzes images attached to the email."""

    name = "image-model"

    def score_images(self, images: Iterable[bytes]) -> Dict[str, float]:
        """Return mock image risk scores."""

        images = list(images)
        if not images:
            return {"image_score": 0.0}
        # Simplistic heuristic: assume higher risk with more images.
        score = min(len(images) * 0.1, 1.0)
        return {"image_score": round(score, 3)}
