import io
import os
import tempfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from model.data.dataset import NUM_CLASSES
from model.networks.crnn import CRNN


@pytest.fixture
def dummy_checkpoint(tmp_path):
    model = CRNN(
        img_height=32,
        num_channels=1,
        num_classes=NUM_CLASSES,
        hidden_size=32,
        num_lstm_layers=1,
    )
    path = str(tmp_path / "best_model.pt")
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
def client(dummy_checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT", dummy_checkpoint)
    monkeypatch.setenv("MODEL_HIDDEN_SIZE", "32")
    monkeypatch.setenv("MODEL_NUM_LSTM_LAYERS", "1")

    from backend.main import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture
def test_image():
    img = Image.fromarray(
        np.random.randint(0, 255, (50, 200), dtype=np.uint8), mode="L"
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestRecognizeEndpoint:
    def test_upload_image(self, client, test_image):
        resp = client.post(
            "/api/recognize",
            files={"file": ("test.png", test_image, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "confidence" in data
        assert isinstance(data["text"], str)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_reject_non_image(self, client):
        buf = io.BytesIO(b"not an image")
        resp = client.post(
            "/api/recognize",
            files={"file": ("test.txt", buf, "text/plain")},
        )
        assert resp.status_code == 400

    def test_reject_no_file(self, client):
        resp = client.post("/api/recognize")
        assert resp.status_code == 422
