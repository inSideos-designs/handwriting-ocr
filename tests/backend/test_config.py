import os
import pytest
from backend.core.config import AppConfig


class TestAppConfig:
    def test_default_trocr_model(self):
        config = AppConfig()
        assert config.trocr_model == "microsoft/trocr-large-handwritten"

    def test_default_corrector_model(self):
        config = AppConfig()
        assert config.corrector_model == "llm/checkpoints/ocr-corrector"

    def test_corrector_enabled_default(self):
        config = AppConfig()
        assert config.corrector_enabled is True

    def test_env_override_trocr_model(self, monkeypatch):
        monkeypatch.setenv("TROCR_MODEL", "microsoft/trocr-small-handwritten")
        config = AppConfig()
        assert config.trocr_model == "microsoft/trocr-small-handwritten"

    def test_env_override_corrector_enabled_false(self, monkeypatch):
        monkeypatch.setenv("CORRECTOR_ENABLED", "false")
        config = AppConfig()
        assert config.corrector_enabled is False
