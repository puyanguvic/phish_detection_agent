"""Basic tests for the phishing detection agent."""
from __future__ import annotations

from phish_agent.agents.phish_detection_agent import PhishDetectionAgent
from phish_agent.schemas.email_schema import EmailAnalysisRequest


def test_agent_returns_result() -> None:
    agent = PhishDetectionAgent()
    payload = EmailAnalysisRequest(body="Please login", urls=["http://example.com/login"])

    result = agent.analyze(payload)

    assert result.verdict in {"phishing", "benign"}
    assert 0.0 <= result.score <= 1.0
