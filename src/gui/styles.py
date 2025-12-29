"""
macOS-Native Styling for the GUI
================================

Provides consistent styling that matches the macOS aesthetic.
"""

# Color palette - macOS inspired
COLORS = {
    # Background colors
    "bg_primary": "#1E1E1E",  # Main dark background
    "bg_secondary": "#252526",  # Sidebar, cards
    "bg_tertiary": "#2D2D2D",  # Input fields, hover
    "bg_hover": "#3C3C3C",  # Hover state
    "bg_active": "#094771",  # Selected/active item
    # Text colors
    "text_primary": "#CCCCCC",  # Main text
    "text_secondary": "#969696",  # Secondary text
    "text_muted": "#6E6E6E",  # Muted/placeholder
    "text_accent": "#569CD6",  # Links, accents
    # Border colors
    "border": "#3C3C3C",
    "border_focus": "#007ACC",
    # Message colors
    "user_bg": "#2B5278",
    "assistant_bg": "#2D2D2D",
    # Status colors
    "success": "#4EC9B0",
    "warning": "#CE9178",
    "error": "#F44747",
    "info": "#569CD6",
    # Button colors
    "button_primary": "#0E639C",
    "button_primary_hover": "#1177BB",
    "button_secondary": "#3C3C3C",
    "button_secondary_hover": "#4C4C4C",
    # Toggle colors
    "toggle_on": "#0E639C",
    "toggle_off": "#3C3C3C",
}

# Main application stylesheet
STYLESHEET = f"""
/* Main Window */
QMainWindow {{
    background-color: {COLORS["bg_primary"]};
}}

/* General Widget Styling */
QWidget {{
    background-color: {COLORS["bg_primary"]};
    color: {COLORS["text_primary"]};
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 14px;
}}

/* Scroll Areas */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {COLORS["bg_primary"]};
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS["border"]};
    border-radius: 5px;
    min-height: 30px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS["text_muted"]};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS["bg_primary"]};
    height: 10px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS["border"]};
    border-radius: 5px;
    min-width: 30px;
    margin: 2px;
}}

/* Labels */
QLabel {{
    color: {COLORS["text_primary"]};
    background-color: transparent;
}}

/* Text Input */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLORS["bg_tertiary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 8px 12px;
    color: {COLORS["text_primary"]};
    selection-background-color: {COLORS["bg_active"]};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS["border_focus"]};
}}

QLineEdit::placeholder, QTextEdit::placeholder {{
    color: {COLORS["text_muted"]};
}}

/* Push Buttons */
QPushButton {{
    background-color: {COLORS["button_secondary"]};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    color: {COLORS["text_primary"]};
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS["button_secondary_hover"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["bg_active"]};
}}

QPushButton:disabled {{
    background-color: {COLORS["bg_tertiary"]};
    color: {COLORS["text_muted"]};
}}

/* Primary Button */
QPushButton#primaryButton {{
    background-color: {COLORS["button_primary"]};
    color: white;
}}

QPushButton#primaryButton:hover {{
    background-color: {COLORS["button_primary_hover"]};
}}

/* Toggle Buttons */
QPushButton#toggleButton {{
    background-color: {COLORS["toggle_off"]};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

QPushButton#toggleButton:checked {{
    background-color: {COLORS["toggle_on"]};
}}

/* Combo Box */
QComboBox {{
    background-color: {COLORS["bg_tertiary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 6px 12px;
    color: {COLORS["text_primary"]};
    min-width: 150px;
}}

QComboBox:hover {{
    border-color: {COLORS["text_muted"]};
}}

QComboBox:focus {{
    border-color: {COLORS["border_focus"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {COLORS["text_secondary"]};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS["bg_secondary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    selection-background-color: {COLORS["bg_active"]};
    color: {COLORS["text_primary"]};
    padding: 4px;
}}

/* List Widget */
QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
}}

QListWidget::item {{
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 4px;
}}

QListWidget::item:hover {{
    background-color: {COLORS["bg_hover"]};
}}

QListWidget::item:selected {{
    background-color: {COLORS["bg_active"]};
}}

/* Splitter */
QSplitter::handle {{
    background-color: {COLORS["border"]};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* Menu */
QMenu {{
    background-color: {COLORS["bg_secondary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS["bg_active"]};
}}

QMenu::separator {{
    height: 1px;
    background-color: {COLORS["border"]};
    margin: 4px 8px;
}}

/* Tooltips */
QToolTip {{
    background-color: {COLORS["bg_secondary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 4px 8px;
    color: {COLORS["text_primary"]};
}}

/* Frame for sidebar */
QFrame#sidebar {{
    background-color: {COLORS["bg_secondary"]};
    border-right: 1px solid {COLORS["border"]};
}}

/* Chat message bubbles */
QFrame#userMessage {{
    background-color: {COLORS["user_bg"]};
    border-radius: 12px;
    padding: 12px;
}}

QFrame#assistantMessage {{
    background-color: {COLORS["assistant_bg"]};
    border-radius: 12px;
    padding: 12px;
}}

/* Input container */
QFrame#inputContainer {{
    background-color: {COLORS["bg_secondary"]};
    border-top: 1px solid {COLORS["border"]};
}}

/* Header */
QFrame#header {{
    background-color: {COLORS["bg_secondary"]};
    border-bottom: 1px solid {COLORS["border"]};
}}

/* Progress indicator */
QProgressBar {{
    background-color: {COLORS["bg_tertiary"]};
    border: none;
    border-radius: 4px;
    height: 4px;
}}

QProgressBar::chunk {{
    background-color: {COLORS["button_primary"]};
    border-radius: 4px;
}}
"""

# Code block styling for markdown
CODE_BLOCK_STYLE = f"""
    background-color: {COLORS["bg_tertiary"]};
    border-radius: 6px;
    padding: 12px;
    font-family: "SF Mono", "Menlo", "Monaco", monospace;
    font-size: 13px;
"""
