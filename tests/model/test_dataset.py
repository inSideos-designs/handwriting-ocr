import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from model.data.dataset import HandwritingDataset, CHARS
from model.training.config import TrainConfig


@pytest.fixture
def sample_dataset(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    labels = []
    for i, name in enumerate(["HELLO", "WORLD", "TEST"]):
        img = Image.fromarray(
            np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
        )
        fname = f"IMG_{i:04d}.jpg"
        img.save(img_dir / fname)
        labels.append({"FILENAME": fname, "IDENTITY": name})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(labels).to_csv(csv_path, index=False)
    return str(csv_path), str(img_dir)


class TestHandwritingDataset:
    def test_length(self, sample_dataset):
        csv_path, img_dir = sample_dataset
        ds = HandwritingDataset(csv_path, img_dir)
        assert len(ds) == 3

    def test_getitem_returns_tensor_and_target(self, sample_dataset):
        csv_path, img_dir = sample_dataset
        ds = HandwritingDataset(csv_path, img_dir)
        img, target, target_length = ds[0]
        assert img.shape == (1, 32, 128)
        assert len(target) > 0
        assert target_length > 0

    def test_filters_unreadable(self, sample_dataset):
        csv_path, img_dir = sample_dataset
        df = pd.read_csv(csv_path)
        df.loc[0, "IDENTITY"] = "UNREADABLE"
        df.to_csv(csv_path, index=False)
        ds = HandwritingDataset(csv_path, img_dir)
        assert len(ds) == 2

    def test_target_encoding(self, sample_dataset):
        csv_path, img_dir = sample_dataset
        ds = HandwritingDataset(csv_path, img_dir)
        _, target, target_length = ds[0]  # "HELLO"
        assert target_length == 5
        for idx in target[:target_length]:
            assert 0 <= idx < len(CHARS)


class TestChars:
    def test_chars_contains_uppercase(self):
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert c in CHARS

    def test_chars_contains_lowercase(self):
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert c in CHARS

    def test_chars_contains_digits(self):
        for c in "0123456789":
            assert c in CHARS

    def test_chars_contains_special(self):
        for c in " '-":
            assert c in CHARS
