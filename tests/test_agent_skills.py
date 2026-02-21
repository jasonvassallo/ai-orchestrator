import json

from src.agent.skills import discover_skills, select_skills_for_prompt


def _write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_skills_from_codex_claude_gemini(tmp_path) -> None:
    codex_home = tmp_path / "codex"
    claude_home = tmp_path / "claude"
    gemini_home = tmp_path / "gemini"

    _write(
        codex_home / "skills" / "playwright" / "SKILL.md",
        "---\nname: playwright\ndescription: Browser automation\n---\nUse playwright.",
    )

    plugin_root = claude_home / "plugins" / "cache" / "market" / "plugin" / "1.0.0"
    _write(
        plugin_root / "skills" / "reviewer" / "SKILL.md",
        "---\nname: reviewer\ndescription: Review code\n---\nReview guidance.",
    )
    _write(
        claude_home / "plugins" / "installed_plugins.json",
        json.dumps(
            {
                "plugins": {
                    "reviewer@market": [
                        {
                            "installPath": str(plugin_root),
                        }
                    ]
                }
            }
        ),
    )

    extension_root = gemini_home / "extensions" / "code-review"
    _write(
        extension_root / "gemini-extension.json",
        json.dumps(
            {
                "name": "code-review",
                "description": "Gemini code review extension",
                "contextFileName": "GEMINI.md",
            }
        ),
    )
    _write(extension_root / "GEMINI.md", "Gemini context for code review")

    skills = discover_skills(
        codex_home=codex_home,
        claude_home=claude_home,
        gemini_home=gemini_home,
        sources=["codex", "claude", "gemini"],
    )
    names = {skill.name for skill in skills}
    assert "playwright" in names
    assert "reviewer" in names
    assert "gemini-code-review" in names


def test_select_skills_for_prompt_explicit_and_keyword() -> None:
    from src.agent.types import SkillDoc

    skills = [
        SkillDoc(
            name="playwright",
            description="Browser automation and snapshots",
            body="...",
            source="codex",
            path="skills/playwright/SKILL.md",
        ),
        SkillDoc(
            name="security-review",
            description="Perform security best-practices review",
            body="...",
            source="codex",
            path="skills/security-review/SKILL.md",
        ),
    ]

    selected = select_skills_for_prompt(
        "Please use $playwright and check browser flow",
        skills,
    )
    names = {item.name for item in selected}
    assert "playwright" in names
