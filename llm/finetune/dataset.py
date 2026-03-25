import random
import string

import torch
from torch.utils.data import Dataset


# OCR-like error types
CHAR_SUBSTITUTIONS = {
    "O": "0", "0": "O", "I": "1", "1": "I", "l": "1",
    "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z",
    "G": "6", "6": "G", "A": "4", "4": "A", "T": "7", "7": "T",
    "g": "9", "9": "g", "E": "3", "3": "E",
}


def corrupt_text(text, error_rate=0.15):
    """Apply OCR-like errors to clean text."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < error_rate:
            error_type = random.choice(["substitute", "delete", "duplicate", "swap"])

            if error_type == "substitute" and chars[i] in CHAR_SUBSTITUTIONS:
                chars[i] = CHAR_SUBSTITUTIONS[chars[i]]
            elif error_type == "delete" and len(chars) > 1:
                chars[i] = ""
            elif error_type == "duplicate":
                chars[i] = chars[i] * 2
            elif error_type == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]

    return "".join(chars)


def generate_training_pairs(clean_texts, num_augmentations=3, error_rate=0.15):
    """Generate (corrupted, clean) pairs from a list of clean texts."""
    pairs = []
    for text in clean_texts:
        for _ in range(num_augmentations):
            corrupted = corrupt_text(text, error_rate=error_rate)
            if corrupted != text:
                pairs.append((corrupted, text))
    return pairs


PROMPT_TEMPLATE = "Correct the OCR errors in the following text:\n{input}\n\nCorrected:\n{output}"


class OCRCorrectionDataset(Dataset):
    """Dataset of OCR error correction pairs formatted for causal LM training."""

    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        corrupted, clean = self.pairs[idx]
        prompt = PROMPT_TEMPLATE.format(input=corrupted, output=clean)

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
