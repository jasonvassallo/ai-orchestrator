"""
Utilities to bootstrap and health-check a local SearxNG instance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class BootstrapConfig:
    container_name: str
    image: str
    base_url: str
    host: str
    port: int


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        check=False,
    )  # nosec B603


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _container_state(container_name: str) -> str | None:
    result = _run_command(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.State}}",
        ]
    )
    if result.returncode != 0:
        return None
    state = result.stdout.strip().splitlines()
    if not state:
        return None
    return state[0].strip().lower() or None


def _wait_for_health(base_url: str, timeout_seconds: int = 20) -> tuple[bool, str]:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/search"
    last_error = ""
    while time.time() < deadline:
        try:
            response = httpx.get(
                url,
                params={"q": "health", "format": "json"},
                timeout=3.0,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return True, "SearxNG is healthy."
            last_error = "Unexpected JSON response format."
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1)
    return False, last_error or "Health check timed out."


def bootstrap(config: BootstrapConfig) -> int:
    if not _docker_available():
        print("Docker is required. Install Docker Desktop first.")
        return 1

    state = _container_state(config.container_name)
    if state == "running":
        print(f"Container '{config.container_name}' is already running.")
    elif state is not None:
        result = _run_command(["docker", "start", config.container_name])
        if result.returncode != 0:
            print(f"Failed to start container '{config.container_name}': {result.stderr}")
            return 1
        print(f"Started existing container '{config.container_name}'.")
    else:
        run_result = _run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                config.container_name,
                "--restart",
                "unless-stopped",
                "-p",
                f"{config.host}:{config.port}:8080",
                "-e",
                f"BASE_URL={config.base_url.rstrip('/')}/",
                "-e",
                "INSTANCE_NAME=AI Orchestrator SearxNG",
                config.image,
            ]
        )
        if run_result.returncode != 0:
            print(f"Failed to create container '{config.container_name}': {run_result.stderr}")
            return 1
        print(
            f"Created container '{config.container_name}' using image '{config.image}'."
        )

    healthy, message = _wait_for_health(config.base_url)
    if not healthy:
        print(f"SearxNG health check failed: {message}")
        return 1

    print(message)
    print("\nSuggested config snippet:")
    print(
        json.dumps(
            {
                "agent": {
                    "web": {
                        "searxngBaseUrl": config.base_url,
                    }
                }
            },
            indent=2,
        )
    )
    return 0


def health(base_url: str) -> int:
    url = f"{base_url.rstrip('/')}/search"
    try:
        response = httpx.get(
            url,
            params={"q": "open source llm", "format": "json"},
            timeout=6.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        print(f"Health check failed: {exc}")
        print("Run `ai-searx-bootstrap` to start a local instance.")
        return 1

    if not isinstance(payload, dict):
        print("Health check failed: response was not a JSON object.")
        return 1

    results = payload.get("results")
    if not isinstance(results, list):
        print("Health check failed: missing results array.")
        return 1

    print(f"SearxNG healthy at {base_url} (results={len(results)}).")
    if results and isinstance(results[0], dict):
        top_title = results[0].get("title")
        top_url = results[0].get("url")
        if isinstance(top_title, str) and isinstance(top_url, str):
            print(f"Top result: {top_title}")
            print(f"URL: {top_url}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SearxNG bootstrap/health utility")
    sub = parser.add_subparsers(dest="command")

    bootstrap_parser = sub.add_parser("bootstrap", help="Create/start local SearxNG")
    bootstrap_parser.add_argument(
        "--container-name",
        default="ai-orchestrator-searxng",
        help="Docker container name",
    )
    bootstrap_parser.add_argument(
        "--image",
        default="searxng/searxng:latest",
        help="SearxNG Docker image",
    )
    bootstrap_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host bind address",
    )
    bootstrap_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Host bind port",
    )

    health_parser = sub.add_parser("health", help="Check local SearxNG health")
    health_parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="SearxNG base URL",
    )

    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "bootstrap":
        base_url = f"http://{args.host}:{args.port}"
        return bootstrap(
            BootstrapConfig(
                container_name=args.container_name,
                image=args.image,
                base_url=base_url,
                host=args.host,
                port=args.port,
            )
        )

    if args.command == "health":
        return health(args.base_url)

    parser.print_help()
    return 1


def bootstrap_main() -> int:
    return bootstrap(
        BootstrapConfig(
            container_name="ai-orchestrator-searxng",
            image="searxng/searxng:latest",
            base_url="http://127.0.0.1:8080",
            host="127.0.0.1",
            port=8080,
        )
    )


def health_main() -> int:
    return health("http://127.0.0.1:8080")


if __name__ == "__main__":
    sys.exit(main())
