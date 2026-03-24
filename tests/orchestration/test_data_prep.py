import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from orchestration.assets.data_prep import (
    load_and_clean_labels,
    validate_images,
    split_dataset,
)


@pytest.fixture
def raw_dataset(tmp_path):
    img_dir = tmp_path / "train"
    img_dir.mkdir()

    labels = []
    for i in range(50):
        name = ["HELLO", "WORLD", "UNREADABLE", "TEST", "ABC"][i % 5]
        img = Image.fromarray(
            np.random.randint(0, 255, (50, 200), dtype=np.uint8), mode="L"
        )
        fname = f"TRAIN_{i:05d}.jpg"
        img.save(img_dir / fname)
        labels.append({"FILENAME": fname, "IDENTITY": name})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(labels).to_csv(csv_path, index=False)
    return str(csv_path), str(img_dir)


class TestLoadAndCleanLabels:
    def test_removes_unreadable(self, raw_dataset):
        csv_path, _ = raw_dataset
        df = load_and_clean_labels(csv_path)
        assert "UNREADABLE" not in df["IDENTITY"].values

    def test_returns_dataframe(self, raw_dataset):
        csv_path, _ = raw_dataset
        df = load_and_clean_labels(csv_path)
        assert "FILENAME" in df.columns
        assert "IDENTITY" in df.columns

    def test_removes_empty_labels(self, raw_dataset):
        csv_path, _ = raw_dataset
        df = pd.read_csv(csv_path)
        df.loc[0, "IDENTITY"] = ""
        df.to_csv(csv_path, index=False)
        result = load_and_clean_labels(csv_path)
        assert "" not in result["IDENTITY"].values


class TestValidateImages:
    def test_filters_missing_images(self, raw_dataset):
        csv_path, img_dir = raw_dataset
        df = load_and_clean_labels(csv_path)
        df = pd.concat([df, pd.DataFrame([{
            "FILENAME": "MISSING.jpg", "IDENTITY": "GHOST"
        }])], ignore_index=True)
        valid = validate_images(df, img_dir)
        assert "MISSING.jpg" not in valid["FILENAME"].values


class TestSplitDataset:
    def test_split_proportions(self, raw_dataset):
        csv_path, _ = raw_dataset
        df = load_and_clean_labels(csv_path)
        train, val = split_dataset(df, val_ratio=0.2)
        total = len(train) + len(val)
        assert total == len(df)
        assert len(val) == pytest.approx(len(df) * 0.2, abs=2)
