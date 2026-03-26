import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset

from model.data.preprocessing import preprocess_image, IMG_HEIGHT
from model.data.dataset import CHAR_TO_IDX, CHARS

IAM_IMG_WIDTH = 512  # IAM lines are much wider than single words


def encode_label(text, max_len=64):
    """Encode text label to indices, handling mixed case and punctuation."""
    encoded = []
    for c in text:
        if c in CHAR_TO_IDX:
            encoded.append(CHAR_TO_IDX[c])
    return encoded[:max_len]


class IAMLineDataset(Dataset):
    """IAM handwriting line dataset from HuggingFace."""

    def __init__(self, split="train", max_label_len=64, max_samples=None):
        ds = load_dataset("Teklia/IAM-line", split=split)

        self.images = []
        self.labels = []
        self.max_label_len = max_label_len

        for i, sample in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            text = sample["text"].strip()
            if not text:
                continue

            # Filter to only characters in our charset
            filtered = "".join(c for c in text if c in CHAR_TO_IDX)
            if len(filtered) < 2:
                continue

            self.images.append(sample["image"])
            self.labels.append(filtered)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img_tensor = preprocess_image(img, target_width=IAM_IMG_WIDTH)

        label = self.labels[idx]
        encoded = encode_label(label, self.max_label_len)
        target_length = len(encoded)

        target = torch.zeros(self.max_label_len, dtype=torch.long)
        target[:target_length] = torch.tensor(encoded, dtype=torch.long)

        return img_tensor, target, target_length
