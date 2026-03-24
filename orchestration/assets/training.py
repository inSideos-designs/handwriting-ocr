import os

import dagster

from model.data.dataset import HandwritingDataset
from model.training.config import TrainConfig
from model.training.trainer import Trainer


@dagster.asset(
    deps=["cleaned_dataset"],
    description="Train CRNN model on cleaned handwriting dataset",
)
def trained_model(context: dagster.AssetExecutionContext) -> dict:
    data_dir = os.environ.get("DATA_OUTPUT_DIR", "data/processed")
    img_dir = os.environ.get(
        "KAGGLE_IMG_DIR", "/Users/cultistsid/Downloads/archive/train_v2/train"
    )

    train_csv = os.path.join(data_dir, "train_labels.csv")
    val_csv = os.path.join(data_dir, "val_labels.csv")

    config = TrainConfig(
        batch_size=int(os.environ.get("BATCH_SIZE", "64")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "0.001")),
        num_epochs=int(os.environ.get("NUM_EPOCHS", "50")),
        checkpoint_dir=os.environ.get("CHECKPOINT_DIR", "model/checkpoints"),
    )

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

    return {
        "checkpoint_path": os.path.join(config.checkpoint_dir, "best_model.pt"),
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
