import io
import importlib

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    mock_service = MagicMock()
    mock_service.recognize.return_value = {
        "text": "Hello World",
        "raw_text": "Helo Wrld",
        "confidence": 0.85,
    }

    with patch("backend.services.recognition.CorrectedRecognitionService", return_value=mock_service):
        import backend.main
        importlib.reload(backend.main)
        app = backend.main.app
        yield TestClient(app)


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
