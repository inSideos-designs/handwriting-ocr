import os
import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split

import dagster

from orchestration.gcs import download_file, upload_file


def load_and_clean_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["IDENTITY"] != "UNREADABLE"]
    df = df[df["IDENTITY"].str.strip().str.len() > 0]
    df = df.dropna(subset=["IDENTITY"])
    df = df.reset_index(drop=True)
    return df


def validate_images(df: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    valid_mask = df["FILENAME"].apply(
        lambda f: os.path.isfile(os.path.join(img_dir, f))
    )
    return df[valid_mask].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, val_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=42
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


@dagster.asset(
    description="Load, clean, and validate the Kaggle handwriting dataset",
)
def cleaned_dataset(context: dagster.AssetExecutionContext) -> dict:
    gcs_bucket = os.environ.get("GCS_BUCKET", "")
    csv_path = os.environ.get("KAGGLE_CSV_PATH", "")
    img_dir = os.environ.get("KAGGLE_IMG_DIR", "")

    if gcs_bucket:
        local_csv = os.path.join(tempfile.mkdtemp(), "labels.csv")
        download_file(f"{gcs_bucket}/data/raw/written_name_train_v2.csv", local_csv)
        csv_path = local_csv
        context.log.info("Downloaded CSV from GCS")

    df = load_and_clean_labels(csv_path)
    context.log.info(f"After cleaning: {len(df)} samples")

    if not gcs_bucket:
        df = validate_images(df, img_dir)
        context.log.info(f"After image validation: {len(df)} samples")
    else:
        context.log.info("Skipping local image validation (images are in GCS)")

    train_df, val_df = split_dataset(df, val_ratio=0.1)
    context.log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    if gcs_bucket:
        train_tmp = os.path.join(tempfile.mkdtemp(), "train_labels.csv")
        val_tmp = os.path.join(tempfile.mkdtemp(), "val_labels.csv")
        train_df.to_csv(train_tmp, index=False)
        val_df.to_csv(val_tmp, index=False)
        upload_file(train_tmp, f"{gcs_bucket}/data/processed/train_labels.csv")
        upload_file(val_tmp, f"{gcs_bucket}/data/processed/val_labels.csv")
        context.log.info("Uploaded processed CSVs to GCS")
        return {
            "train_csv": f"{gcs_bucket}/data/processed/train_labels.csv",
            "val_csv": f"{gcs_bucket}/data/processed/val_labels.csv",
            "img_dir": f"{gcs_bucket}/data/raw/train_v2/train",
            "train_size": len(train_df),
            "val_size": len(val_df),
        }

    output_dir = os.environ.get("DATA_OUTPUT_DIR", "data/processed")
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_labels.csv")
    val_path = os.path.join(output_dir, "val_labels.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return {
        "train_csv": train_path,
        "val_csv": val_path,
        "img_dir": img_dir,
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
