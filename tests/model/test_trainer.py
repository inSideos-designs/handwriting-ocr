import tempfile
import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from model.training.config import TrainConfig
from model.training.trainer import Trainer
from model.data.dataset import HandwritingDataset


@pytest.fixture
def tiny_dataset(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    labels = []
    for i in range(20):
        name = "HELLO" if i % 2 == 0 else "WORLD"
        img = Image.fromarray(
            np.random.randint(100, 200, (50, 200), dtype=np.uint8), mode="L"
        )
        fname = f"IMG_{i:04d}.jpg"
        img.save(img_dir / fname)
        labels.append({"FILENAME": fname, "IDENTITY": name})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(labels).to_csv(csv_path, index=False)

    return HandwritingDataset(str(csv_path), str(img_dir))


@pytest.fixture
def config(tmp_path):
    return TrainConfig(
        batch_size=4,
        learning_rate=0.001,
        num_epochs=2,
        hidden_size=32,
        num_lstm_layers=1,
        patience=5,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


class TestTrainer:
    def test_train_runs_without_error(self, tiny_dataset, config):
        trainer = Trainer(config)
        history = trainer.train(tiny_dataset, tiny_dataset)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_loss_decreases_or_stays_bounded(self, tiny_dataset, config):
        trainer = Trainer(config)
        history = trainer.train(tiny_dataset, tiny_dataset)
        assert all(loss < 1000 for loss in history["train_loss"])

    def test_saves_checkpoint(self, tiny_dataset, config):
        trainer = Trainer(config)
        trainer.train(tiny_dataset, tiny_dataset)
        checkpoint_files = os.listdir(config.checkpoint_dir)
        assert len(checkpoint_files) > 0
        assert any(f.endswith(".pt") for f in checkpoint_files)

    def test_checkpoint_is_loadable(self, tiny_dataset, config):
        trainer = Trainer(config)
        trainer.train(tiny_dataset, tiny_dataset)
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint
