#!/usr/bin/env python3
"""
AI Orchestrator GUI Application
================================

A native Mac GUI application for the AI Orchestrator.

Usage:
    python -m src.gui.app

Or after installation:
    ai-app
"""

from __future__ import annotations

import asyncio

# Ensure Qt platform is set for macOS
import os
import sys

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")


def main() -> int:
    """Main entry point for the GUI application."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Error: PySide6 is required for the GUI app.")
        print("Install it with: pip install pyside6")
        return 1

    try:
        import qasync
    except ImportError:
        print("Error: qasync is required for async support.")
        print("Install it with: pip install qasync")
        return 1

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AI Orchestrator")
    app.setApplicationDisplayName("AI Orchestrator")
    app.setOrganizationName("AI Orchestrator")

    # Set macOS-specific attributes
    if sys.platform == "darwin":
        app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)

    # Set default font
    font = QFont("-apple-system", 14)
    app.setFont(font)

    # Create async event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Create and show main window
    # Use absolute import for PyInstaller compatibility
    try:
        from src.gui.main_window import MainWindow
    except ImportError:
        from .main_window import MainWindow
    window = MainWindow()
    window.show()

    # Run event loop
    with loop:
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
