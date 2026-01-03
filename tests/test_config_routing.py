import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.orchestrator import AIOrchestrator, TaskType


class TestConfigRouting(unittest.TestCase):
    def setUp(self):
        # Reset singleton-like behavior if any (none in this class, but good practice)
        pass

    @patch("src.orchestrator.AIOrchestrator._load_user_config")
    def test_custom_task_routing(self, mock_load_config):
        # Configure a custom route for code generation
        # Force it to use a specific model that might not be the #1 default choice
        # "codestral" is a valid model key.
        mock_load_config.return_value = {"taskRouting": {"code": ["codestral"]}}

        orchestrator = AIOrchestrator()

        # Verify config was loaded
        self.assertEqual(
            orchestrator._user_config["taskRouting"]["code"], ["codestral"]
        )

        # Test selection for code task
        # Normally Claude Sonnet 4.5 or GPT-4o might win.
        # Here we force Codestral.
        task_types = [(TaskType.CODE_GENERATION, 1.0)]
        selected_model = orchestrator.select_model(task_types)

        self.assertIsNotNone(selected_model)
        self.assertEqual(
            selected_model.model_id, "codestral-latest"
        )  # The ID for key "codestral"

    @patch("src.orchestrator.AIOrchestrator._load_user_config")
    def test_disabled_model(self, mock_load_config):
        # Disable the likely winner for code generation (e.g. Claude Sonnet 4.5)
        # Note: ModelRegistry keys are used in config usually.
        mock_load_config.return_value = {
            "models": {"claude-sonnet-4.5": {"enabled": False}}
        }

        orchestrator = AIOrchestrator()

        # Verify that Claude Sonnet 4.5 is NOT selected for code
        task_types = [(TaskType.CODE_GENERATION, 1.0)]

        # We need to make sure we don't pick the disabled one.
        # We can't easily guarantee what WILL be picked without mocking everything,
        # but we can guarantee what WON'T be picked.
        selected_model = orchestrator.select_model(task_types)

        self.assertIsNotNone(selected_model)
        self.assertNotEqual(selected_model.model_id, "claude-sonnet-4-5-20250929")

    @patch("src.orchestrator.AIOrchestrator._load_user_config")
    def test_model_priority(self, mock_load_config):
        # Set a high priority for a specific model to make it win
        # Let's boost 'gpt-4o-mini' which usually wouldn't beat 'gpt-4o' for general tasks
        mock_load_config.return_value = {
            "models": {
                "gpt-4o-mini": {"priority": 100},  # Max priority
                "gpt-4o": {"priority": 0},  # Min priority
            }
        }

        orchestrator = AIOrchestrator()

        # Task: General NLP
        task_types = [(TaskType.GENERAL_NLP, 1.0)]

        selected_model = orchestrator.select_model(task_types)

        self.assertIsNotNone(selected_model)
        # GPT-4o Mini should win due to massive priority difference
        self.assertEqual(selected_model.model_id, "gpt-4o-mini")

    @patch("src.orchestrator.logging.FileHandler")
    @patch("src.orchestrator.AIOrchestrator._load_user_config")
    def test_logging_setup(self, mock_load_config, mock_file_handler):
        import os
        import tempfile

        log_path = os.path.join(tempfile.gettempdir(), "test_ai_orchestrator.log")

        mock_load_config.return_value = {
            "logging": {"level": "DEBUG", "file": log_path}
        }

        _ = AIOrchestrator()

        # Verify FileHandler was initialized with correct path
        mock_file_handler.assert_called_with(log_path, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
