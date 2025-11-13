"""Smoke tests for individual tools."""
from __future__ import annotations

from phish_agent.tools.text_model_tool import TextModelTool
from phish_agent.tools.url_model_tool import UrlModelTool
from phish_agent.tools.fusion_tool import FusionTool


def test_text_model_scores_content() -> None:
    tool = TextModelTool()
    result = tool.score_text("hello world")
    assert "text_score" in result


def test_url_model_scores_urls() -> None:
    tool = UrlModelTool()
    result = tool.score_urls(["http://phish.test", "https://safe.test"])
    assert "url_score" in result


def test_fusion_combines_signals() -> None:
    fusion = FusionTool()
    combined = fusion.combine([{ "text_score": 0.2 }, { "url_score": 0.8 }])
    assert combined["combined_score"] == 0.5
