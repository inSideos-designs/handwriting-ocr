import pytest
from unittest.mock import MagicMock
from PIL import Image

from model.segmentation.pipeline import PageRecognizer


class TestPageRecognizer:
    def _make_recognizer(self):
        """Create PageRecognizer with a mock predictor."""
        mock_predictor = MagicMock()
        mock_predictor.predict_with_confidence.return_value = ("Hello World", 0.95)
        recognizer = PageRecognizer(predictor=mock_predictor)
        return recognizer, mock_predictor

    def test_accepts_predictor_argument(self):
        mock_predictor = MagicMock()
        recognizer = PageRecognizer(predictor=mock_predictor)
        assert recognizer.predictor is mock_predictor

    def test_recognize_direct_returns_dict(self):
        recognizer, _ = self._make_recognizer()
        img = Image.new("RGB", (200, 50), color="white")
        result = recognizer.recognize(img)
        assert "text" in result
        assert "lines" in result

    def test_recognize_uses_injected_predictor(self):
        recognizer, mock_predictor = self._make_recognizer()
        img = Image.new("RGB", (200, 50), color="white")
        recognizer.recognize(img)
        assert mock_predictor.predict_with_confidence.called
