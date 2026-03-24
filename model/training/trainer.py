import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.data.dataset import NUM_CLASSES, HandwritingDataset
from model.networks.crnn import CRNN
from model.training.config import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CRNN(
            img_height=config.img_height,
            num_channels=1,
            num_classes=NUM_CLASSES,
            hidden_size=config.hidden_size,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout,
        ).to(self.device)

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def train(
        self,
        train_dataset: HandwritingDataset,
        val_dataset: HandwritingDataset,
    ) -> dict[str, list[float]]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, targets, target_lengths in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            output = self.model(images)
            seq_len = output.shape[0]
            batch_size = output.shape[1]
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

            target_flat = []
            for i in range(batch_size):
                target_flat.extend(targets[i, : target_lengths[i]].tolist())
            target_flat = torch.tensor(target_flat, dtype=torch.long)

            loss = self.criterion(output, target_flat, input_lengths, target_lengths)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets, target_lengths in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                output = self.model(images)
                seq_len = output.shape[0]
                batch_size = output.shape[1]
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

                target_flat = []
                for i in range(batch_size):
                    target_flat.extend(targets[i, : target_lengths[i]].tolist())
                target_flat = torch.tensor(target_flat, dtype=torch.long)

                loss = self.criterion(
                    output, target_flat, input_lengths, target_lengths
                )
                total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
