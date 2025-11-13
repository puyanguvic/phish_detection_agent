"""Command line interface for the phishing detection agent."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

from phish_agent.runtime.loader import load_runtime
from phish_agent.schemas.email_schema import EmailAnalysisRequest, EmailAnalysisResult

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "deepagents.local.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an email with the phishing detection agent. Provide the body "
            "via --body or standard input, and optional URLs via --urls."
        )
    )
    parser.add_argument("--subject", default="", help="Subject line of the email")
    parser.add_argument(
        "--body",
        help="Plain text body of the email. If omitted, reads from standard input.",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="Space separated list of URLs contained in the email body.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the DeepAgents YAML configuration file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as formatted JSON (default is a human readable summary).",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> EmailAnalysisResult:
    """Run the CLI and return the analysis result."""

    parser = build_parser()
    args = parser.parse_args(argv)

    body = args.body if args.body is not None else sys.stdin.read().strip()
    if not body:
        parser.error("Email body is required via --body or stdin.")

    urls: List[str] = list(args.urls)

    agent = load_runtime(args.config)
    payload = EmailAnalysisRequest(subject=args.subject, body=body, urls=urls)
    result = agent.analyze(payload)

    if args.json:
        output = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    else:
        output = _format_human_readable(result)
    print(output)
    return result


def _format_human_readable(result: EmailAnalysisResult) -> str:
    signals = result.signals
    keyword_hits = signals.get("keyword_hits", [])
    flagged_urls = signals.get("flagged_urls", [])
    lines = [
        f"Verdict: {result.verdict} (score={result.score:.3f})",
        f"Text score: {signals.get('text_score', 0.0)}",
        f"URL score: {signals.get('url_score', 0.0)}",
    ]
    if keyword_hits:
        lines.append("Triggered keywords: " + ", ".join(keyword_hits))
    if flagged_urls:
        lines.append("Flagged URLs: " + ", ".join(flagged_urls))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    run_cli(argv)


if __name__ == "__main__":  # pragma: no cover - manual execution entry
    main()
