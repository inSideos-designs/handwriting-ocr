import os
import numpy as np
import pytest
import torch
import cv2
from PIL import Image

from model.segmentation.pipeline import PageRecognizer
from model.networks.crnn import CRNN
from model.data.dataset import NUM_CLASSES


@pytest.fixture
def dummy_checkpoint(tmp_path):
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
def recognizer(dummy_checkpoint):
    from model.training.config import TrainConfig
    config = TrainConfig(hidden_size=32, num_lstm_layers=1)
    return PageRecognizer(dummy_checkpoint, config)


def make_page_with_text(width=800, height=400, num_lines=2, words_per_line=3):
    """Create a synthetic page with dark text blobs on white background."""
    img = np.ones((height, width), dtype=np.uint8) * 255

    y = 50
    for _ in range(num_lines):
        x = 50
        for _ in range(words_per_line):
            # Draw a dark rectangle simulating a word
            cv2.rectangle(img, (x, y), (x + 80, y + 30), 0, -1)
            x += 120
        y += 80

    return Image.fromarray(img, mode="L")


class TestPageRecognizer:
    def test_recognize_returns_dict(self, recognizer):
        page = make_page_with_text()
        result = recognizer.recognize(page)
        assert isinstance(result, dict)
        assert "text" in result
        assert "lines" in result
        assert "num_lines" in result
        assert "num_words" in result

    def test_finds_lines(self, recognizer):
        page = make_page_with_text(num_lines=2)
        result = recognizer.recognize(page)
        assert result["num_lines"] >= 1

    def test_finds_words(self, recognizer):
        page = make_page_with_text(num_lines=1, words_per_line=3)
        result = recognizer.recognize(page)
        assert result["num_words"] >= 1

    def test_empty_page(self, recognizer):
        page = Image.fromarray(np.ones((400, 800), dtype=np.uint8) * 255, mode="L")
        result = recognizer.recognize(page)
        assert result["text"] == ""
        assert result["num_lines"] == 0

    def test_lines_have_words(self, recognizer):
        page = make_page_with_text(num_lines=2, words_per_line=2)
        result = recognizer.recognize(page)
        for line in result["lines"]:
            assert "text" in line
            assert "words" in line
            assert "confidence" in line

    def test_words_have_confidence(self, recognizer):
        page = make_page_with_text(num_lines=1, words_per_line=2)
        result = recognizer.recognize(page)
        for line in result["lines"]:
            for word in line["words"]:
                assert "text" in word
                assert "confidence" in word
                assert 0.0 <= word["confidence"] <= 1.0

    def test_rgb_input(self, recognizer):
        gray_page = make_page_with_text()
        rgb_arr = np.stack([np.array(gray_page)] * 3, axis=-1)
        rgb_page = Image.fromarray(rgb_arr, mode="RGB")
        result = recognizer.recognize(rgb_page)
        assert isinstance(result, dict)
