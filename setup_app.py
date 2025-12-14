#!/usr/bin/env python3
"""
py2app build script for AI Orchestrator Mac App
================================================

Build commands:
    # Development build (faster, with alias)
    python setup_app.py py2app -A

    # Production build (standalone .app)
    python setup_app.py py2app

Output:
    dist/AI Orchestrator.app
"""

import sys
from setuptools import setup

# Ensure we're on macOS
if sys.platform != "darwin":
    print("Error: py2app is only supported on macOS")
    sys.exit(1)

APP = ["src/gui/app.py"]
DATA_FILES = []

OPTIONS = {
    "argv_emulation": False,
    "iconfile": None,  # TODO: Add icon file
    "plist": {
        "CFBundleName": "AI Orchestrator",
        "CFBundleDisplayName": "AI Orchestrator",
        "CFBundleIdentifier": "com.aiorchestrator.app",
        "CFBundleVersion": "2.0.0",
        "CFBundleShortVersionString": "2.0.0",
        "LSMinimumSystemVersion": "10.15",
        "LSUIElement": False,  # Show in Dock
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,  # Support dark mode
    },
    "packages": [
        "src",
        "PySide6",
        "qasync",
        "httpx",
        "openai",
        "anthropic",
        "keyring",
        "cryptography",
        "midiutil",
    ],
    "includes": [
        "src.gui",
        "src.gui.app",
        "src.gui.main_window",
        "src.gui.chat_widget",
        "src.gui.input_widget",
        "src.gui.sidebar",
        "src.gui.styles",
        "src.gui.settings_dialog",
        "src.orchestrator",
        "src.credentials",
        "src.storage",
        "src.music",
    ],
    "excludes": [
        "tkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "PIL",
        "cv2",
    ],
    "semi_standalone": False,
    "site_packages": True,
}

setup(
    name="AI Orchestrator",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
