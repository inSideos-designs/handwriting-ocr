import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from model.data.dataset import NUM_CLASSES
from model.inference.predictor import Predictor
from model.networks.crnn import CRNN
from model.training.config import TrainConfig


@pytest.fixture
def saved_model(tmp_path):
    model = CRNN(
        img_height=32,
        num_channels=1,
        num_classes=NUM_CLASSES,
        hidden_size=32,
        num_lstm_layers=1,
    )
    path = str(tmp_path / "test_model.pt")
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 1.0,
        },
        path,
    )
    return path


@pytest.fixture
def config(saved_model):
    return TrainConfig(hidden_size=32, num_lstm_layers=1)


class TestPredictor:
    def test_predict_returns_string(self, saved_model, config):
        predictor = Predictor(saved_model, config)
        img = Image.fromarray(
            np.random.randint(0, 255, (50, 200), dtype=np.uint8), mode="L"
        )
        result = predictor.predict(img)
        assert isinstance(result, str)

    def test_predict_with_confidence(self, saved_model, config):
        predictor = Predictor(saved_model, config)
        img = Image.fromarray(
            np.random.randint(0, 255, (50, 200), dtype=np.uint8), mode="L"
        )
        text, confidence = predictor.predict_with_confidence(img)
        assert isinstance(text, str)
        assert 0.0 <= confidence <= 1.0

    def test_predict_rgb_image(self, saved_model, config):
        predictor = Predictor(saved_model, config)
        img = Image.fromarray(
            np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8), mode="RGB"
        )
        result = predictor.predict(img)
        assert isinstance(result, str)

    def test_predict_batch(self, saved_model, config):
        predictor = Predictor(saved_model, config)
        images = [
            Image.fromarray(
                np.random.randint(0, 255, (50, 200), dtype=np.uint8), mode="L"
            )
            for _ in range(3)
        ]
        results = predictor.predict_batch(images)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
