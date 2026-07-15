"""
CLI entrypoint for agent mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any

from rich.console import Console

from ..orchestrator import AgentStatus, AIOrchestrator
from .runner import AgentRunner, AgentRunOptions


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="AI Orchestrator Agent CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the local agent")
    parser.add_argument(
        "--model", "-m", help="Model override (default: local agent model)"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Persistent conversation session id",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--incognito", action="store_true", help="Disable conversation persistence"
    )
    parser.add_argument(
        "--browser-automation",
        action="store_true",
        help="Enable dangerous browser automation tool",
    )
    parser.add_argument("--disable-web-tools", action="store_true")
    parser.add_argument("--disable-mcp", action="store_true")
    parser.add_argument("--disable-skills", action="store_true")
    parser.add_argument("--output", "-o", help="Export response to .md or .json")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if not args.prompt:
        parser.print_help()
        return 0

    console = Console()
    orchestrator = AIOrchestrator(prefer_local=True, verbose=args.verbose)
    runner = AgentRunner(orchestrator)

    if args.browser_automation:
        console.print(
            "[bold red]Warning:[/] browser automation is enabled and may execute "
            "real interactions on websites."
        )

    status_handle: Any = None

    def on_status(status: AgentStatus) -> None:
        nonlocal status_handle
        if status_handle is not None:
            status_handle.update(f"[cyan]{status.message}")

    options = AgentRunOptions(
        session_id=args.session,
        model_override=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_web_tools=not args.disable_web_tools,
        enable_mcp=not args.disable_mcp,
        enable_skills=not args.disable_skills,
        enable_browser_automation=args.browser_automation,
        incognito=args.incognito,
    )

    with console.status("[cyan]Running agent...", spinner="dots") as status:
        status_handle = status
        response = await runner.run(
            args.prompt, options=options, status_callback=on_status
        )

    if response.success:
        console.print(
            f"\n[bold cyan][{response.model}][/]"
            f" [dim]({response.latency_ms:.0f}ms, provider={response.provider})[/]"
        )
        console.print("-" * 60)
        console.print(response.content)
        console.print("-" * 60)
        console.print(
            f"[dim]Tokens: {response.usage.get('input_tokens', 0)} in / "
            f"{response.usage.get('output_tokens', 0)} out[/]"
        )
    else:
        console.print(f"\n[red]Error:[/] {response.error}")
        return 1

    if args.output:
        if args.output.endswith(".json"):
            payload = {
                "date": datetime.now().isoformat(),
                "prompt": args.prompt,
                "response": response.content,
                "model": response.model,
                "provider": response.provider,
                "usage": response.usage,
                "metadata": response.metadata,
            }
            output_text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            output_text = (
                "# AI Orchestrator Agent Response\n\n"
                f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"**Model:** {response.model}\n"
                f"**Provider:** {response.provider}\n"
                f"**Latency:** {response.latency_ms:.0f}ms\n\n"
                "## Prompt\n\n"
                f"{args.prompt}\n\n"
                "## Response\n\n"
                f"{response.content}\n"
            )
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output_text)
        console.print(f"[green]Exported:[/] {args.output}")

    return 0


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    sys.exit(main())
