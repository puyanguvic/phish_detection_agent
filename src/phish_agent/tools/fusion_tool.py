"""Placeholder signal fusion tool."""
from __future__ import annotations

from typing import Dict, Iterable, List


class FusionTool:
    """Combines signals from multiple tools into a unified representation."""

    name = "signal-fusion"

    def combine(self, signal_sets: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Merge signal dictionaries and compute an aggregate score."""

        merged: Dict[str, float] = {}
        for signals in signal_sets:
            merged.update(signals)
        merged["combined_score"] = self._average_score(merged)
        return merged

    def _average_score(self, signals: Dict[str, float]) -> float:
        relevant: List[float] = [value for value in signals.values() if isinstance(value, (int, float))]
        if not relevant:
            return 0.0
        return round(sum(relevant) / len(relevant), 3)
