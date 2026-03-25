import json
import os
import tempfile

import dagster

from orchestration.gcs import download_file, download_directory, upload_file
from model.data.dataset import HandwritingDataset
from model.training.config import TrainConfig
from model.training.trainer import Trainer


K8S_GPU_CONFIG = {
    "dagster-k8s/config": {
        "container_config": {
            "resources": {
                "requests": {"nvidia.com/gpu": "1", "memory": "8Gi", "cpu": "3"},
                "limits": {"nvidia.com/gpu": "1", "memory": "12Gi", "cpu": "4"},
            },
        },
        "pod_spec_config": {
            "node_selector": {
                "cloud.google.com/gke-accelerator": "nvidia-tesla-t4",
            },
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
        },
    }
}


@dagster.asset(
    deps=["cleaned_dataset"],
    description="Train CRNN model on cleaned handwriting dataset",
    op_tags=K8S_GPU_CONFIG,
)
def trained_model(context: dagster.AssetExecutionContext) -> dict:
    gcs_bucket = os.environ.get("GCS_BUCKET", "")
    data_dir = os.environ.get("DATA_OUTPUT_DIR", "data/processed")
    img_dir = os.environ.get("KAGGLE_IMG_DIR", "")

    config = TrainConfig(
        batch_size=int(os.environ.get("BATCH_SIZE", "64")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "0.001")),
        num_epochs=int(os.environ.get("NUM_EPOCHS", "50")),
        checkpoint_dir=os.environ.get("CHECKPOINT_DIR", "model/checkpoints"),
    )

    if gcs_bucket:
        tmpdir = tempfile.mkdtemp()
        context.log.info("Downloading processed CSVs from GCS...")
        train_csv = download_file(
            f"{gcs_bucket}/data/processed/train_labels.csv",
            os.path.join(tmpdir, "train_labels.csv"),
        )
        val_csv = download_file(
            f"{gcs_bucket}/data/processed/val_labels.csv",
            os.path.join(tmpdir, "val_labels.csv"),
        )

        context.log.info("Downloading training images from GCS (this may take a while)...")
        local_img_dir = os.path.join(tmpdir, "images")
        download_directory(f"{gcs_bucket}/data/raw/train_v2/train", local_img_dir)
        img_dir = local_img_dir
        context.log.info("Download complete")
    else:
        train_csv = os.path.join(data_dir, "train_labels.csv")
        val_csv = os.path.join(data_dir, "val_labels.csv")

    train_dataset = HandwritingDataset(train_csv, img_dir)
    val_dataset = HandwritingDataset(val_csv, img_dir)

    context.log.info(f"Training on {len(train_dataset)} samples")
    context.log.info(f"Validating on {len(val_dataset)} samples")

    trainer = Trainer(config)
    history = trainer.train(train_dataset, val_dataset)

    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    context.log.info(f"Final train loss: {final_train_loss:.4f}")
    context.log.info(f"Final val loss: {final_val_loss:.4f}")

    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    if gcs_bucket:
        gcs_checkpoint = f"{gcs_bucket}/checkpoints/best_model.pt"
        upload_file(checkpoint_path, gcs_checkpoint)
        context.log.info(f"Uploaded checkpoint to {gcs_checkpoint}")

        history_path = os.path.join(tmpdir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)
        upload_file(history_path, f"{gcs_bucket}/artifacts/training_history.json")

        return {
            "checkpoint_path": gcs_checkpoint,
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        }

    return {
        "checkpoint_path": checkpoint_path,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
