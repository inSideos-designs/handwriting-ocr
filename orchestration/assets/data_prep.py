import os

import pandas as pd
from sklearn.model_selection import train_test_split

import dagster


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
    csv_path = os.environ.get(
        "KAGGLE_CSV_PATH", "/Users/cultistsid/Downloads/archive/written_name_train_v2.csv"
    )
    img_dir = os.environ.get(
        "KAGGLE_IMG_DIR", "/Users/cultistsid/Downloads/archive/train_v2/train"
    )

    df = load_and_clean_labels(csv_path)
    context.log.info(f"After cleaning: {len(df)} samples")

    df = validate_images(df, img_dir)
    context.log.info(f"After image validation: {len(df)} samples")

    train_df, val_df = split_dataset(df, val_ratio=0.1)
    context.log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

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
