import os
import string

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from model.data.preprocessing import preprocess_image

CHARS = string.ascii_uppercase + string.ascii_lowercase + string.digits + " '-"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for CTC blank
NUM_CLASSES = len(CHARS) + 1  # +1 for blank


def encode_label(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_label(indices: list[int]) -> str:
    idx_to_char = {v: k for k, v in CHAR_TO_IDX.items()}
    return "".join(idx_to_char.get(i, "") for i in indices)


class HandwritingDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, max_label_len: int = 32):
        df = pd.read_csv(csv_path)
        df = df[df["IDENTITY"] != "UNREADABLE"]
        df = df[df["IDENTITY"].str.len() <= max_label_len]
        df = df.reset_index(drop=True)

        self.img_dir = img_dir
        self.filenames = df["FILENAME"].tolist()
        self.labels = df["IDENTITY"].tolist()
        self.max_label_len = max_label_len

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(img_path)
        img_tensor = preprocess_image(img)

        label = self.labels[idx]
        encoded = encode_label(label)
        target_length = len(encoded)

        target = torch.zeros(self.max_label_len, dtype=torch.long)
        target[:target_length] = torch.tensor(encoded, dtype=torch.long)

        return img_tensor, target, target_length
