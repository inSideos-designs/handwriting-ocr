"""Train CRNN on IAM handwriting line dataset for mixed-case cursive recognition."""

import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from model.data.iam_dataset import IAMLineDataset, IAM_IMG_WIDTH
from model.data.dataset import NUM_CLASSES
from model.networks.crnn import CRNN


def train_iam(
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    max_samples=None,
    checkpoint_dir="model/checkpoints",
    resume_from=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print("Loading IAM dataset from HuggingFace...")

    full_ds = IAMLineDataset(split="train", max_label_len=64, max_samples=max_samples)
    print(f"Loaded {len(full_ds)} samples")

    # Split 90/10
    val_size = max(1, len(full_ds) // 10)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CRNN(
        img_height=32,
        num_channels=1,
        num_classes=NUM_CLASSES,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.1,
    ).to(device)

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, targets, target_lengths in train_loader:
            images = images.to(device)

            output = model(images)
            seq_len = output.shape[0]
            batch_sz = output.shape[1]
            input_lengths = torch.full((batch_sz,), seq_len, dtype=torch.long)

            target_flat = []
            for i in range(batch_sz):
                target_flat.extend(targets[i, :target_lengths[i]].tolist())
            target_flat = torch.tensor(target_flat, dtype=torch.long)

            loss = criterion(output, target_flat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)

                output = model(images)
                seq_len = output.shape[0]
                batch_sz = output.shape[1]
                input_lengths = torch.full((batch_sz,), seq_len, dtype=torch.long)

                target_flat = []
                for i in range(batch_sz):
                    target_flat.extend(targets[i, :target_lengths[i]].tolist())
                target_flat = torch.tensor(target_flat, dtype=torch.long)

                loss = criterion(output, target_flat, input_lengths, target_lengths)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, path)
            print(f"  Saved checkpoint (val_loss={val_loss:.4f})")

    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_iam(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resume_from=args.resume,
    )
