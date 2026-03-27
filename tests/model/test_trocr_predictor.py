import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch


class TestTrOCRPredictor:
    def _make_predictor(self):
        """Create predictor with mocked model loading."""
        with patch("model.inference.trocr_predictor.VisionEncoderDecoderModel") as mock_model_cls, \
             patch("model.inference.trocr_predictor.TrOCRProcessor") as mock_proc_cls:

            mock_processor = MagicMock()
            # Processor call returns object with .pixel_values attribute
            proc_output = MagicMock()
            proc_output.pixel_values = torch.randn(1, 3, 384, 384)
            mock_processor.return_value = proc_output
            mock_processor.batch_decode.return_value = ["Hello World"]
            mock_proc_cls.from_pretrained.return_value = mock_processor

            mock_model = MagicMock()
            # Generate returns object with .sequences and .scores attributes
            gen_output = MagicMock()
            gen_output.sequences = torch.tensor([[1, 2, 3]])
            gen_output.scores = (torch.randn(1, 50000),)  # one step of scores
            mock_model.generate.return_value = gen_output
            mock_model.to.return_value = mock_model
            mock_model_cls.from_pretrained.return_value = mock_model

            from model.inference.trocr_predictor import TrOCRPredictor
            predictor = TrOCRPredictor.__new__(TrOCRPredictor)
            predictor.device = torch.device("cpu")
            predictor.processor = mock_processor
            predictor.model = mock_model

        return predictor

    def test_predict_returns_string(self):
        predictor = self._make_predictor()
        img = Image.new("RGB", (200, 50), color="white")
        result = predictor.predict(img)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_predict_with_confidence_returns_tuple(self):
        predictor = self._make_predictor()
        img = Image.new("RGB", (200, 50), color="white")
        text, confidence = predictor.predict_with_confidence(img)
        assert isinstance(text, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_batch_returns_list(self):
        predictor = self._make_predictor()
        images = [Image.new("RGB", (200, 50)) for _ in range(3)]
        results = predictor.predict_batch(images)
        assert isinstance(results, list)
        assert len(results) == 3
