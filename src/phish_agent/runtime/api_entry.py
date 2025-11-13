"""FastAPI entry point exposing the phishing scan API."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from phish_agent.runtime.loader import load_runtime
from phish_agent.schemas.email_schema import EmailAnalysisRequest, EmailAnalysisResult

app = FastAPI(title="Phish Agent")
_agent = load_runtime(Path(__file__).resolve().parents[2] / "configs" / "deepagents.local.yaml")


@app.post("/scan", response_model=EmailAnalysisResult)
def scan_email(payload: EmailAnalysisRequest) -> EmailAnalysisResult:
    """Analyze a submitted email payload and return the agent verdict."""

    return _agent.analyze(payload)
