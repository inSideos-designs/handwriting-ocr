import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile

from google.cloud import storage

from model.data.dataset import HandwritingDataset
from model.training.config import TrainConfig
from model.training.trainer import Trainer


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
    parser = argparse.ArgumentParser(description="Train CRNN model")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket name (without gs://)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp()
    bucket = args.gcs_bucket

    print("Downloading dataset tarball from GCS...")
    tar_path = os.path.join(tmpdir, "dataset.tar.gz")
    download_blob(bucket, "data/dataset.tar.gz", tar_path)

    print("Extracting dataset...")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    os.remove(tar_path)
    print("Extraction complete")

    train_csv = os.path.join(data_dir, "written_name_train_v2.csv")
    val_csv = os.path.join(data_dir, "written_name_validation_v2.csv")
    img_dir = os.path.join(data_dir, "train_v2", "train")

    # clean and split using the data prep functions
    from orchestration.assets.data_prep import load_and_clean_labels, validate_images, split_dataset

    df = load_and_clean_labels(train_csv)
    print(f"After cleaning: {len(df)} samples")

    df = validate_images(df, img_dir)
    print(f"After image validation: {len(df)} samples")

    train_df, val_df = split_dataset(df, val_ratio=0.1)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    processed_dir = os.path.join(tmpdir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_csv_path = os.path.join(processed_dir, "train_labels.csv")
    val_csv_path = os.path.join(processed_dir, "val_labels.csv")
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
    )

    train_dataset = HandwritingDataset(train_csv_path, img_dir)
    val_dataset = HandwritingDataset(val_csv_path, img_dir)

    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = Trainer(config)
    history = trainer.train(train_dataset, val_dataset)

    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    print(f"Training complete. Train loss: {final_train:.4f}, Val loss: {final_val:.4f}")

    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    upload_blob(checkpoint_path, bucket, "checkpoints/best_model.pt")

    history_path = os.path.join(tmpdir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    upload_blob(history_path, bucket, "artifacts/training_history.json")

    print("Done.")


if __name__ == "__main__":
    main()
