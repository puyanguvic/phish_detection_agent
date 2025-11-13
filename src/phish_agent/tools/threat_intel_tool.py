"""Placeholder threat intelligence integration tool."""
from __future__ import annotations

from typing import Dict, Iterable


class ThreatIntelTool:
    """Queries threat intelligence sources for indicators of compromise."""

    name = "threat-intel"

    def lookup(self, indicators: Iterable[str]) -> Dict[str, float]:
        """Return static scores for the provided indicators."""

        indicators = list(indicators)
        if not indicators:
            return {"intel_score": 0.0}
        return {"intel_score": 0.5, "matches": len(indicators)}
