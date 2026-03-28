import random
import string

import torch
from torch.utils.data import Dataset


# OCR-like error types — common confusions in handwriting recognition
CHAR_SUBSTITUTIONS = {
    # Letter-digit confusions
    "O": "0", "0": "O", "I": "1", "1": "I", "l": "1",
    "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z",
    "G": "6", "6": "G", "A": "4", "4": "A", "T": "7", "7": "T",
    "g": "9", "9": "g", "E": "3", "3": "E",
    # Common handwriting confusions
    "m": "rn", "rn": "m", "n": "ri", "u": "v", "v": "u",
    "c": "e", "e": "c", "a": "o", "o": "a",
    "d": "cl", "cl": "d", "h": "b", "b": "h",
    "w": "vv", "f": "t", "t": "f",
    "i": "j", "j": "i",
}


def corrupt_text(text, error_rate=0.15):
    """Apply OCR-like errors to clean text."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < error_rate:
            error_type = random.choice(["substitute", "delete", "duplicate", "swap", "space"])

            if error_type == "substitute" and chars[i] in CHAR_SUBSTITUTIONS:
                chars[i] = CHAR_SUBSTITUTIONS[chars[i]]
            elif error_type == "delete" and len(chars) > 1:
                chars[i] = ""
            elif error_type == "duplicate":
                chars[i] = chars[i] * 2
            elif error_type == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif error_type == "space":
                # Random space insertion/deletion
                if chars[i] == " ":
                    chars[i] = ""
                else:
                    chars[i] = chars[i] + " "

    return "".join(chars)


# Sentence corpus for synthetic training data
SENTENCE_CORPUS = [
    # Common phrases
    "Hello my name is", "How are you doing today",
    "The quick brown fox jumps over the lazy dog",
    "Please sign your name here", "Thank you very much",
    "Dear Sir or Madam", "Yours sincerely",
    "Best regards", "To whom it may concern",
    # Addresses
    "123 Main Street", "456 Oak Avenue", "789 Elm Boulevard",
    "New York NY 10001", "Los Angeles CA 90001",
    "San Francisco California", "Chicago Illinois",
    # Dates and numbers
    "January 15 2024", "March 3 1995", "December 25 2000",
    "Date of birth", "Social Security Number",
    "Phone 555 123 4567", "Account number 987654",
    # Medical/legal forms
    "Patient name", "Date of service", "Diagnosis code",
    "Signature of authorized representative",
    "I hereby certify that the above information is correct",
    "Policy number", "Claim reference",
    # General handwriting content
    "The meeting is scheduled for next Tuesday",
    "Please review the attached document",
    "I would like to request a refund",
    "The total amount due is", "Payment received with thanks",
    "Notes from the interview", "Action items from today",
    "Reminder to call back tomorrow morning",
    "Pick up groceries on the way home",
    "Happy birthday to you", "Congratulations on your achievement",
    "Weather forecast calls for rain", "Temperature is 72 degrees",
    # Names (common and uncommon)
    "John Smith", "Mary Johnson", "Robert Williams",
    "Jennifer Davis", "Michael Brown", "Sarah Wilson",
    "Alexander Hamilton", "Benjamin Franklin",
    "Elizabeth Warren", "Theodore Roosevelt",
    "Sid", "Hi this is Sid", "My name is Sid",
]


def generate_training_pairs(texts=None, num_augmentations=3, error_rate=0.15):
    """Generate (corrupted, clean) pairs from a list of clean texts."""
    if texts is None:
        texts = SENTENCE_CORPUS

    pairs = []
    for text in texts:
        for _ in range(num_augmentations):
            corrupted = corrupt_text(text, error_rate=error_rate)
            if corrupted != text:
                pairs.append((corrupted, text))
    return pairs


def generate_real_ocr_pairs(images_with_labels, predictor):
    """
    Generate training pairs from real OCR output.

    Args:
        images_with_labels: list of (PIL.Image, ground_truth_text) tuples
        predictor: OCR predictor with predict(img) method

    Returns:
        list of (ocr_output, ground_truth) tuples
    """
    pairs = []
    for img, ground_truth in images_with_labels:
        ocr_text = predictor.predict(img)
        if ocr_text.strip() and ocr_text != ground_truth:
            pairs.append((ocr_text, ground_truth))
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
