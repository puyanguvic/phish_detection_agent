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


def test_agent_detects_phishing_email() -> None:
    agent = PhishDetectionAgent()
    payload = EmailAnalysisRequest(
        body=(
            "Urgent: your account will be suspended. Verify your account by entering "
            "your password now."
        ),
        urls=["http://login.example.com/update-account"],
    )

    result = agent.analyze(payload)

    assert result.verdict == "phishing"
    assert result.score >= 0.5


def test_agent_handles_benign_email() -> None:
    agent = PhishDetectionAgent()
    payload = EmailAnalysisRequest(
        subject="Team lunch",
        body="Let's meet for lunch tomorrow at the cafe.",
        urls=["https://example.com/menu"],
    )

    result = agent.analyze(payload)

    assert result.verdict == "benign"
    assert result.score < 0.5
