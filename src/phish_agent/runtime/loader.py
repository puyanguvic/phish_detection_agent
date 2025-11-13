"""Runtime loader utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - exercised indirectly
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    yaml = None

from phish_agent.agents.phish_detection_agent import PhishDetectionAgent


def load_runtime(config_path: str | Path) -> PhishDetectionAgent:
    """Instantiate the phishing detection agent from a DeepAgents config."""

    config = _load_yaml(config_path)
    agent = PhishDetectionAgent()
    agent.load_context(config.get("context", {}))
    return agent


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
