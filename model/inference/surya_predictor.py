"""Handwriting recognition using Surya OCR (detection + recognition)."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageOps
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

from model.segmentation.photo_preprocess import preprocess_photo


class SuryaPredictor:
    """Handwriting recognition using Surya OCR with built-in detection."""

    def __init__(self, device=None):
        self.foundation = FoundationPredictor(device=device or "cpu")
        self.det_predictor = DetectionPredictor(device=device or "cpu")
        self.rec_predictor = RecognitionPredictor(self.foundation)

    def predict(self, img: Image.Image) -> str:
        text, _ = self.predict_with_confidence(img)
        return text

    def predict_with_confidence(self, img: Image.Image) -> tuple[str, float]:
        # EXIF rotation
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Preprocess: shadow removal + contrast enhancement
        preprocessed = preprocess_photo(img)

        # Run Surya detection + recognition
        results = self.rec_predictor([preprocessed], det_predictor=self.det_predictor)
        result = results[0]

        # Filter out empty lines and low confidence noise
        lines = [
            line for line in result.text_lines
            if line.text.strip() and line.confidence < 0.99  # conf=1.0 means empty detection
        ]

        if not lines:
            return "", 0.0

        text = " ".join(line.text.strip() for line in lines)
        avg_confidence = float(np.mean([line.confidence for line in lines]))

        return text, avg_confidence

    def predict_page(self, img: Image.Image) -> dict:
        """Full page recognition returning per-line results."""
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        if img.mode != "RGB":
            img = img.convert("RGB")

        preprocessed = preprocess_photo(img)

        results = self.rec_predictor([preprocessed], det_predictor=self.det_predictor)
        result = results[0]

        lines = []
        for line in result.text_lines:
            if line.text.strip() and line.confidence < 0.99:
                lines.append({
                    "text": line.text.strip(),
                    "confidence": round(line.confidence, 4),
                })

        full_text = "\n".join(l["text"] for l in lines)

        return {
            "text": full_text,
            "lines": lines,
            "num_lines": len(lines),
        }

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        return [self.predict(img) for img in images]
