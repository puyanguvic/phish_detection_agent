"""Dataclasses describing email analysis requests and results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EmailAnalysisRequest:
    """Incoming email payload to be analyzed."""

    subject: str = ""
    body: str = ""
    urls: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    images: Optional[List[bytes]] = None


@dataclass
class EmailAnalysisResult:
    """Agent output after running the phishing detection pipeline."""

    verdict: str
    score: float
    signals: Dict[str, float]
    metadata: Dict[str, object] = field(default_factory=dict)
