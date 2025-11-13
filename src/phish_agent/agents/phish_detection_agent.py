"""Core phishing detection agent orchestrating analysis tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from phish_agent.schemas.email_schema import EmailAnalysisRequest, EmailAnalysisResult
from phish_agent.tools.fusion_tool import FusionTool
from phish_agent.tools.policy_tool import PolicyTool
from phish_agent.tools.text_model_tool import TextModelTool
from phish_agent.tools.url_model_tool import UrlModelTool


@dataclass
class PhishDetectionAgent:
    """Coordinates specialized tools to evaluate phishing risk."""

    text_tool: TextModelTool = field(default_factory=TextModelTool)
    url_tool: UrlModelTool = field(default_factory=UrlModelTool)
    fusion_tool: FusionTool = field(default_factory=FusionTool)
    policy_tool: PolicyTool = field(default_factory=PolicyTool)

    def analyze(self, payload: EmailAnalysisRequest) -> EmailAnalysisResult:
        """Run the phishing analysis pipeline and return a result."""

        text_signals = self.text_tool.score_text(payload.body)
        url_signals = self.url_tool.score_urls(payload.urls)
        fused = self.fusion_tool.combine([text_signals, url_signals])
        decision = self.policy_tool.evaluate(fused)
        return EmailAnalysisResult(
            verdict=decision["verdict"],
            score=decision["score"],
            signals=fused,
            metadata={"analyzed_tools": self._active_tools()},
        )

    def _active_tools(self) -> List[str]:
        """Return names of tools currently wired into the agent."""

        return [
            self.text_tool.name,
            self.url_tool.name,
            self.fusion_tool.name,
            self.policy_tool.name,
        ]

    def load_context(self, context: Dict[str, Any]) -> None:
        """Placeholder for loading contextual data such as tenant policies."""

        self.policy_tool.update_policies(context.get("policies", {}))
