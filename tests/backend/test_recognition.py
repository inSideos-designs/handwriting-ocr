import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.services.recognition import CorrectedRecognitionService


class TestCorrectedRecognitionService:
    def _make_service(self):
        """Create service with mocked TrOCR and corrector."""
        mock_predictor = MagicMock()
        mock_predictor.predict_with_confidence.return_value = ("Helo Wrld", 0.85)

        mock_corrector = MagicMock()
        mock_corrector.correct.return_value = "Hello World"

        service = CorrectedRecognitionService.__new__(CorrectedRecognitionService)
        service.predictor = mock_predictor
        service.corrector = mock_corrector
        service.page_recognizer = MagicMock()
        service.page_recognizer.recognize.return_value = {
            "text": "Line one\nLine two",
            "lines": [
                {"text": "Line one", "confidence": 0.9},
                {"text": "Line two", "confidence": 0.8},
            ],
            "num_lines": 2,
        }
        return service

    def test_recognize_returns_text_and_raw_text(self):
        service = self._make_service()
        img = Image.new("RGB", (200, 50), color="white")
        result = service.recognize(img)
        assert result["raw_text"] == "Helo Wrld"
        assert result["text"] == "Hello World"
        assert "confidence" in result

    def test_recognize_calls_corrector(self):
        service = self._make_service()
        img = Image.new("RGB", (200, 50), color="white")
        service.recognize(img)
        service.corrector.correct.assert_called_once_with("Helo Wrld")

    def test_recognize_page_corrects_each_line(self):
        service = self._make_service()
        service.corrector.correct.side_effect = ["Line one corrected", "Line two corrected"]
        img = Image.new("RGB", (400, 200), color="white")
        result = service.recognize_page(img)
        assert service.corrector.correct.call_count == 2
        assert "Line one corrected" in result["text"]
        assert "Line two corrected" in result["text"]
