import json
from pathlib import Path

from src.agent.mcp import discover_mcp_servers


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_mcp_servers_from_multiple_sources(tmp_path) -> None:
    codex_home = tmp_path / "codex"
    claude_home = tmp_path / "claude"
    gemini_home = tmp_path / "gemini"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    _write(
        codex_home / "config.toml",
        """
[mcp_servers.microsoft-learn]
url = "https://learn.microsoft.com/api/mcp"
""".strip(),
    )

    # Claude global + plugin MCP
    _write(
        tmp_path / ".claude.json",
        json.dumps(
            {
                "projects": {
                    str(workspace): {
                        "mcpServers": {
                            "huggingface": {
                                "httpUrl": "https://huggingface.co/mcp?login"
                            }
                        }
                    }
                }
            }
        ),
    )
    plugin_install = (
        claude_home / "plugins" / "cache" / "vendor" / "playwright" / "1.0.0"
    )
    _write(
        claude_home / "plugins" / "installed_plugins.json",
        json.dumps(
            {
                "plugins": {
                    "playwright@vendor": [
                        {
                            "installPath": str(plugin_install),
                        }
                    ]
                }
            }
        ),
    )
    _write(
        plugin_install / ".mcp.json",
        json.dumps(
            {
                "playwright": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                }
            }
        ),
    )

    # Gemini extension MCP
    extension_root = gemini_home / "extensions" / "chrome-devtools-mcp"
    _write(
        extension_root / "gemini-extension.json",
        json.dumps(
            {
                "name": "chrome-devtools-mcp",
                "mcpServers": {
                    "chrome-devtools": {
                        "command": "npx",
                        "args": ["chrome-devtools-mcp@latest"],
                    }
                },
            }
        ),
    )

    servers = discover_mcp_servers(
        codex_home=codex_home,
        claude_home=claude_home,
        gemini_home=gemini_home,
        sources=("codex", "claude", "gemini"),
        workspace=workspace,
    )

    names = {server.name for server in servers}
    assert "microsoft-learn" in names
    assert "huggingface" in names
    assert "playwright" in names
    assert "chrome-devtools" in names
