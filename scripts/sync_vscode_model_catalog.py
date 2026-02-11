#!/usr/bin/env python3
"""
Sync VS Code extension model catalog from src.orchestrator.ModelRegistry.

This script is intended to be run locally and in scheduled CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.orchestrator import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "vscode-extension" / "model-catalog.json"
PACKAGE_JSON_PATH = ROOT / "vscode-extension" / "package.json"

SUPPORTED_PROVIDERS = {"openai", "anthropic", "mistral", "mlx"}
EXCLUDED_MODEL_KEYS = {
    # Maps to gpt-4o internally; keep a single canonical 4o entry in the extension.
    "gpt-4.5-preview",
}

PROVIDER_PRIORITY = {"openai": 0, "anthropic": 1, "mistral": 2, "mlx": 3}

TASK_TYPE_MAP = {
    "GENERAL_NLP": "general",
    "CODE_GENERATION": "code",
    "REASONING": "reasoning",
    "DEEP_REASONING": "reasoning",
    "CREATIVE_WRITING": "creative",
    "SUMMARIZATION": "summarization",
    "LONG_CONTEXT": "long-context",
    "MATH": "math",
    "MULTIMODAL": "multimodal",
    "LOCAL_MODEL": "local",
}


def _task_types_from_model(model: Any) -> list[str]:
    task_types: set[str] = set()
    for task_type in model.task_types:
        mapped = TASK_TYPE_MAP.get(task_type.name)
        if mapped:
            task_types.add(mapped)
    if model.provider == "mlx":
        task_types.add("local")
    if not task_types:
        task_types.add("general")
    return sorted(task_types)


def _build_catalog() -> dict[str, dict[str, Any]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for model_key, model in ModelRegistry.MODELS.items():
        if model.provider not in SUPPORTED_PROVIDERS:
            continue
        if model_key in EXCLUDED_MODEL_KEYS:
            continue

        strengths = [str(item) for item in model.strengths if isinstance(item, str)]

        row = {
            "name": model.name,
            "provider": model.provider,
            "modelId": model.model_id,
            "contextWindow": int(model.context_window),
            "costPer1kInput": float(model.cost_per_1k_input),
            "costPer1kOutput": float(model.cost_per_1k_output),
            "strengths": strengths,
            "taskTypes": _task_types_from_model(model),
        }
        rows.append((model_key, row))

    rows.sort(
        key=lambda pair: (
            PROVIDER_PRIORITY.get(pair[1]["provider"], 99),
            pair[1]["name"].lower(),
            pair[0],
        )
    )
    return dict(rows)


def _description_for_model(model_data: dict[str, Any]) -> str:
    strengths = model_data.get("strengths", [])
    top_strengths = ", ".join(strengths[:3]) if strengths else "general purpose"
    return f"{model_data['name']} - {top_strengths}"


def _update_package_json(catalog: dict[str, dict[str, Any]]) -> None:
    with PACKAGE_JSON_PATH.open("r", encoding="utf-8") as file:
        package = json.load(file)

    properties = package["contributes"]["configuration"]["properties"]
    selected_model = properties["ai-orchestrator.selectedModel"]

    enum_values = ["", *catalog.keys()]
    enum_descriptions = [
        "Automatic (let orchestrator choose)",
        *[_description_for_model(catalog[key]) for key in catalog],
    ]

    selected_model["enum"] = enum_values
    selected_model["enumDescriptions"] = enum_descriptions
    properties["ai-orchestrator.preferLocal"][
        "description"
    ] = "Prefer local models (MLX) for privacy and lower latency"

    # Remove stale Ollama-specific setting and ensure MLX runtime settings exist.
    properties.pop("ai-orchestrator.ollamaBaseUrl", None)
    properties["ai-orchestrator.pythonExecutable"] = {
        "type": "string",
        "default": "python3",
        "description": "Python executable used to run local MLX models via src.orchestrator",
    }
    properties["ai-orchestrator.pythonProjectPath"] = {
        "type": "string",
        "default": "",
        "description": "Path to ai-orchestrator project root (defaults to first opened workspace folder)",
    }

    keywords = package.get("keywords", [])
    if isinstance(keywords, list):
        filtered = [str(keyword) for keyword in keywords if str(keyword) != "ollama"]
        if "mlx" not in filtered:
            filtered.append("mlx")
        package["keywords"] = filtered

    with PACKAGE_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(package, file, indent=2)
        file.write("\n")


def main() -> None:
    catalog = _build_catalog()
    with CATALOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2)
        file.write("\n")
    _update_package_json(catalog)


if __name__ == "__main__":
    main()
