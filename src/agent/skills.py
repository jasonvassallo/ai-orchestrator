"""
Cross-ecosystem skill discovery (Codex, Claude, Gemini).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from .types import SkillDoc

_FRONTMATTER_PATTERN = re.compile(r"^\s*---\s*\n(?P<fm>[\s\S]*?)\n---\s*\n?")
_FRONTMATTER_LINE_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text

    fm_raw = match.group("fm")
    body = text[match.end() :]
    parsed: dict[str, str] = {}
    for raw_line in fm_raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed_line = _FRONTMATTER_LINE_PATTERN.match(line)
        if not parsed_line:
            continue
        key = parsed_line.group(1).strip()
        value = parsed_line.group(2).strip().strip("\"'")
        parsed[key] = value
    return parsed, body


def _skill_from_path(path: Path, source: str, default_name: str) -> SkillDoc | None:
    text = _read_text(path)
    if text is None:
        return None

    frontmatter, body = _parse_frontmatter(text)
    name = frontmatter.get("name", default_name).strip() or default_name
    description = frontmatter.get("description", "").strip()
    if not description:
        description = f"Imported from {source} ({path.name})"

    return SkillDoc(
        name=name,
        description=description,
        body=body.strip(),
        source=source,
        path=str(path),
    )


def _discover_codex_skills(codex_home: Path) -> list[SkillDoc]:
    skills_root = codex_home / "skills"
    if not skills_root.exists():
        return []

    discovered: list[SkillDoc] = []
    for skill_file in skills_root.glob("*/SKILL.md"):
        default_name = skill_file.parent.name
        skill_doc = _skill_from_path(skill_file, "codex", default_name)
        if skill_doc:
            discovered.append(skill_doc)
    return discovered


def _discover_claude_skills(claude_home: Path) -> list[SkillDoc]:
    installed_plugins_path = claude_home / "plugins" / "installed_plugins.json"
    text = _read_text(installed_plugins_path)
    if text is None:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []
    plugins = payload.get("plugins")
    if not isinstance(plugins, dict):
        return []

    discovered: list[SkillDoc] = []
    for plugin_entries in plugins.values():
        if not isinstance(plugin_entries, list):
            continue
        for plugin_entry in plugin_entries:
            if not isinstance(plugin_entry, dict):
                continue
            install_path_raw = plugin_entry.get("installPath")
            if not isinstance(install_path_raw, str):
                continue
            install_path = Path(install_path_raw)
            if not install_path.exists():
                continue
            for skill_file in install_path.glob("skills/*/SKILL.md"):
                default_name = f"{install_path.name}:{skill_file.parent.name}"
                skill_doc = _skill_from_path(skill_file, "claude", default_name)
                if skill_doc:
                    discovered.append(skill_doc)

    return discovered


def _discover_gemini_context_skills(gemini_home: Path) -> list[SkillDoc]:
    extensions_root = gemini_home / "extensions"
    if not extensions_root.exists():
        return []

    discovered: list[SkillDoc] = []
    for extension_dir in extensions_root.iterdir():
        if not extension_dir.is_dir():
            continue
        manifest_path = extension_dir / "gemini-extension.json"
        text = _read_text(manifest_path)
        if text is None:
            continue
        try:
            manifest = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(manifest, dict):
            continue

        context_filename = manifest.get("contextFileName")
        if not isinstance(context_filename, str):
            continue
        context_path = extension_dir / context_filename
        if not context_path.exists():
            continue

        default_name = f"gemini-{extension_dir.name}"
        description = manifest.get("description")
        skill_doc = _skill_from_path(context_path, "gemini", default_name)
        if skill_doc and isinstance(description, str) and description.strip():
            skill_doc = SkillDoc(
                name=skill_doc.name,
                description=description.strip(),
                body=skill_doc.body,
                source=skill_doc.source,
                path=skill_doc.path,
            )
        if skill_doc:
            discovered.append(skill_doc)

    return discovered


def discover_skills(
    *,
    codex_home: Path,
    claude_home: Path,
    gemini_home: Path,
    sources: Iterable[str],
) -> list[SkillDoc]:
    """Discover skill docs from selected ecosystems."""
    normalized_sources = {source.strip().lower() for source in sources}
    discovered: list[SkillDoc] = []
    if "codex" in normalized_sources:
        discovered.extend(_discover_codex_skills(codex_home))
    if "claude" in normalized_sources:
        discovered.extend(_discover_claude_skills(claude_home))
    if "gemini" in normalized_sources:
        discovered.extend(_discover_gemini_context_skills(gemini_home))

    unique: dict[str, SkillDoc] = {}
    for skill in discovered:
        key = f"{skill.source}:{skill.name.lower()}"
        if key not in unique:
            unique[key] = skill
    return list(unique.values())


def select_skills_for_prompt(
    prompt: str,
    skills: list[SkillDoc],
    *,
    max_auto: int = 4,
) -> list[SkillDoc]:
    """
    Select relevant skills from explicit mentions and keyword overlap.
    """
    prompt_lower = prompt.lower()
    selected: list[SkillDoc] = []
    seen: set[str] = set()

    explicit_mentions = re.findall(r"\$([A-Za-z0-9._:-]+)", prompt)
    explicit_set = {mention.strip().lower() for mention in explicit_mentions}

    for skill in skills:
        skill_names = {
            skill.name.lower(),
            skill.name.lower().replace("_", "-"),
            skill.name.lower().replace("-", "_"),
        }
        if explicit_set & skill_names:
            key = f"{skill.source}:{skill.name.lower()}"
            if key not in seen:
                selected.append(skill)
                seen.add(key)

    if len(selected) >= max_auto:
        return selected[:max_auto]

    scored: list[tuple[float, SkillDoc]] = []
    for skill in skills:
        key = f"{skill.source}:{skill.name.lower()}"
        if key in seen:
            continue
        description_words = set(
            re.findall(r"[A-Za-z0-9]{3,}", skill.description.lower())
        )
        if not description_words:
            continue
        overlap = sum(1 for word in description_words if word in prompt_lower)
        if overlap > 0:
            score = overlap / max(len(description_words), 1)
            scored.append((score, skill))

    scored.sort(key=lambda item: item[0], reverse=True)
    for _, skill in scored:
        key = f"{skill.source}:{skill.name.lower()}"
        if key in seen:
            continue
        selected.append(skill)
        seen.add(key)
        if len(selected) >= max_auto:
            break

    return selected


def render_skill_context(skills: list[SkillDoc], max_chars: int) -> str:
    """Render selected skills into compact prompt context."""
    if not skills or max_chars <= 0:
        return ""

    sections: list[str] = []
    for skill in skills:
        sections.append(
            f"### Skill: {skill.name} ({skill.source})\n"
            f"Description: {skill.description}\n"
            f"Path: {skill.path}\n\n"
            f"{skill.body}"
        )

    full_text = "\n\n".join(sections).strip()
    if len(full_text) <= max_chars:
        return full_text
    return full_text[: max_chars - 3] + "..."


def serialize_skill_catalog(skills: list[SkillDoc]) -> list[dict[str, str]]:
    """Prepare skill list for the skills_list tool."""
    catalog: list[dict[str, str]] = []
    for skill in skills:
        catalog.append(
            {
                "name": skill.name,
                "description": skill.description,
                "source": skill.source,
                "path": skill.path,
            }
        )
    return catalog


def find_skill_by_name(skills: list[SkillDoc], name: str) -> SkillDoc | None:
    """Find a skill by loose name matching."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        return None

    for skill in skills:
        candidates = {
            skill.name.lower(),
            skill.name.lower().replace("_", "-"),
            skill.name.lower().replace("-", "_"),
        }
        if normalized_name in candidates:
            return skill
    return None


def load_json_object(path: Path) -> dict[str, Any] | None:
    """Utility helper used by MCP discovery."""
    text = _read_text(path)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)
    return None
