#!/bin/bash
# Wrapper to run the AI Chat TUI using the uv-managed project environment.
# This avoids manual activation and keeps interpreter/deps pinned by uv.
exec uv run python -m src.tui.app "$@"
