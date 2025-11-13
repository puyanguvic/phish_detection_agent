"""Tests for the command line interface."""
from __future__ import annotations

from phish_agent.runtime.cli import run_cli


def test_cli_analyzes_email(capsys) -> None:
    result = run_cli(
        [
            "--body",
            "Verify your account immediately",
            "--urls",
            "http://login.example.com/reset",
            "--json",
        ]
    )

    output = capsys.readouterr().out
    assert "phishing" in output.lower()
    assert result.verdict in {"phishing", "benign"}
