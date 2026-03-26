"""Combined dataset that loads both Kaggle (single word) and IAM (line) data."""

import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image

from model.data.preprocessing import preprocess_image, IMG_HEIGHT
from model.data.dataset import HandwritingDataset, CHAR_TO_IDX


class IAMWordDataset(Dataset):
    """IAM dataset resized to same width as Kaggle for combined training."""

    def __init__(self, split="train", max_label_len=32, max_samples=None, target_width=128):
        from datasets import load_dataset

        ds = load_dataset("Teklia/IAM-line", split=split)
        self.images = []
        self.labels = []
        self.max_label_len = max_label_len
        self.target_width = target_width

        for i, sample in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            text = sample["text"].strip()
            if not text:
                continue

            filtered = "".join(c for c in text if c in CHAR_TO_IDX)
            if len(filtered) < 2 or len(filtered) > max_label_len:
                continue

            self.images.append(sample["image"])
            self.labels.append(filtered)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img_tensor = preprocess_image(img, target_width=self.target_width)

        label = self.labels[idx]
        encoded = [CHAR_TO_IDX[c] for c in label if c in CHAR_TO_IDX]
        target_length = len(encoded)

        target = torch.zeros(self.max_label_len, dtype=torch.long)
        target[:target_length] = torch.tensor(encoded[:self.max_label_len], dtype=torch.long)

        return img_tensor, target, min(target_length, self.max_label_len)


def build_combined_dataset(
    kaggle_csv=None,
    kaggle_img_dir=None,
    iam_max_samples=None,
    target_width=256,
    max_label_len=32,
):
    """
    Build a combined dataset from Kaggle + IAM, all at the same image width.

    Uses width=256 as a compromise between word-level (128) and line-level (512).
    """
    datasets = []

    if kaggle_csv and kaggle_img_dir and os.path.exists(kaggle_csv):
        kaggle_ds = HandwritingDataset(kaggle_csv, kaggle_img_dir, max_label_len=max_label_len)
        datasets.append(kaggle_ds)

    iam_ds = IAMWordDataset(
        split="train",
        max_label_len=max_label_len,
        max_samples=iam_max_samples,
        target_width=target_width,
    )
    datasets.append(iam_ds)

    return ConcatDataset(datasets)
