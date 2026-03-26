"""Cloud training script for CRNN. Downloads Kaggle data from GCS, loads IAM from HuggingFace, trains combined."""

import argparse
import json
import os
import tarfile
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from google.cloud import storage

from model.data.dataset import HandwritingDataset, NUM_CLASSES
from model.data.combined_dataset import IAMWordDataset
from model.networks.crnn import CRNN
from orchestration.assets.data_prep import load_and_clean_labels, validate_images, split_dataset


def download_blob(bucket_name, blob_name, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} -> {local_path}")


def upload_blob(local_path, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> gs://{bucket_name}/{blob_name}")


def main():
    parser = argparse.ArgumentParser(description="Train CRNN on Kaggle + IAM")
    parser.add_argument("--gcs-bucket", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tmpdir = tempfile.mkdtemp()
    bucket = args.gcs_bucket

    # Download and extract Kaggle dataset
    print("Downloading Kaggle dataset tarball...")
    tar_path = os.path.join(tmpdir, "dataset.tar.gz")
    download_blob(bucket, "data/dataset.tar.gz", tar_path)

    print("Extracting...")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")
    os.remove(tar_path)

    # Prepare Kaggle data
    train_csv = os.path.join(data_dir, "written_name_train_v2.csv")
    img_dir = os.path.join(data_dir, "train_v2", "train")

    df = load_and_clean_labels(train_csv)
    df = validate_images(df, img_dir)
    train_df, val_df = split_dataset(df, val_ratio=0.1)

    processed_dir = os.path.join(tmpdir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)

    kaggle_train = HandwritingDataset(os.path.join(processed_dir, "train.csv"), img_dir)
    kaggle_val = HandwritingDataset(os.path.join(processed_dir, "val.csv"), img_dir)
    print(f"Kaggle: {len(kaggle_train)} train, {len(kaggle_val)} val")

    # Load IAM from HuggingFace
    print("Loading IAM dataset from HuggingFace...")
    iam_train = IAMWordDataset(split="train", max_label_len=32, target_width=128)
    iam_val = IAMWordDataset(split="validation", max_label_len=32, target_width=128)
    print(f"IAM: {len(iam_train)} train, {len(iam_val)} val")

    # Combine
    train_ds = ConcatDataset([kaggle_train, iam_train])
    val_ds = ConcatDataset([kaggle_val, iam_val])
    print(f"Combined: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = CRNN(32, 1, NUM_CLASSES, 256, 2, 0.1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

    best_val = float("inf")
    ckpt_dir = os.path.join(tmpdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for imgs, targets, tlens in train_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            seq_len, bs = out.shape[0], out.shape[1]
            il = torch.full((bs,), seq_len, dtype=torch.long)
            tf = []
            for i in range(bs):
                tf.extend(targets[i, :tlens[i]].tolist())
            tf = torch.tensor(tf, dtype=torch.long)
            loss = criterion(out, tf, il, tlens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
        tl = total / len(train_loader)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for imgs, targets, tlens in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                seq_len, bs = out.shape[0], out.shape[1]
                il = torch.full((bs,), seq_len, dtype=torch.long)
                tf = []
                for i in range(bs):
                    tf.extend(targets[i, :tlens[i]].tolist())
                tf = torch.tensor(tf, dtype=torch.long)
                loss = criterion(out, tf, il, tlens)
                vtotal += loss.item()
        vl = vtotal / len(val_loader)

        scheduler.step(vl)
        print(f"Epoch {epoch+1}/{args.epochs}: train={tl:.4f}, val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl,
            }, os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  Saved checkpoint (val={vl:.4f})")

    # Upload results
    upload_blob(os.path.join(ckpt_dir, "best_model.pt"), bucket, "checkpoints/best_model.pt")
    print("Done!")


if __name__ == "__main__":
    main()
