#!/bin/bash
# Wrapper to run the AI Chat TUI using the virtual environment's python
# This avoids "externally-managed-environment" issues and PATH problems.
source .venv/bin/activate
exec python -m src.tui.app "$@"
