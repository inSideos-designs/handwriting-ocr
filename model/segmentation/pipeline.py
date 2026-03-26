import cv2
import numpy as np
from PIL import Image, ImageOps

from model.segmentation.deskew import preprocess_page, pil_to_cv2, cv2_to_pil
from model.segmentation.lines import extract_lines
from model.segmentation.words import extract_words
from model.inference.predictor import Predictor
from model.training.config import TrainConfig


class PageRecognizer:
    """
    Full page handwriting recognition pipeline.

    Strategy:
    1. First try direct recognition on the whole image (works for single lines/words)
    2. If confidence is low, try segmentation pipeline (works for multi-line pages)
    3. Pick the result with higher confidence
    """

    def __init__(self, checkpoint_path, config=None):
        if config is None:
            config = TrainConfig()
        self.predictor = Predictor(checkpoint_path, config)

    def _recognize_direct(self, img):
        """Try recognizing the image directly without segmentation."""
        text, confidence = self.predictor.predict_with_confidence(img)
        if not text.strip():
            return None
        return {
            "text": text,
            "lines": [{"text": text, "confidence": round(confidence, 4), "words": [{"text": text, "confidence": round(confidence, 4)}]}],
            "num_lines": 1,
            "num_words": 1,
            "avg_confidence": confidence,
        }

    def _recognize_segmented(self, img):
        """Segment the image into lines/words and recognize each."""
        binary, gray = preprocess_page(img)

        line_images = extract_lines(binary)
        if not line_images:
            return None

        result_lines = []
        all_words_text = []
        all_confidences = []

        for line_img in line_images:
            # First try the whole line directly
            line_inverted = 255 - line_img
            line_pil = cv2_to_pil(line_inverted)
            line_text, line_conf = self.predictor.predict_with_confidence(line_pil)

            if line_text.strip() and line_conf > 0.3:
                result_lines.append({
                    "text": line_text,
                    "confidence": round(line_conf, 4),
                    "words": [{"text": line_text, "confidence": round(line_conf, 4)}],
                })
                all_words_text.append(line_text)
                all_confidences.append(line_conf)
                continue

            # Fall back to word-level segmentation
            word_images = extract_words(line_img)
            if not word_images:
                continue

            line_words = []
            for word_img in word_images:
                word_inverted = 255 - word_img
                word_pil = cv2_to_pil(word_inverted)
                text, confidence = self.predictor.predict_with_confidence(word_pil)

                if text.strip():
                    line_words.append({
                        "text": text,
                        "confidence": round(confidence, 4),
                    })
                    all_confidences.append(confidence)

            if line_words:
                line_text = " ".join(w["text"] for w in line_words)
                avg_conf = sum(w["confidence"] for w in line_words) / len(line_words)
                result_lines.append({
                    "text": line_text,
                    "confidence": round(avg_conf, 4),
                    "words": line_words,
                })
                all_words_text.append(line_text)

        if not result_lines:
            return None

        full_text = "\n".join(all_words_text)
        total_words = sum(len(line["words"]) for line in result_lines)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        return {
            "text": full_text,
            "lines": result_lines,
            "num_lines": len(result_lines),
            "num_words": total_words,
            "avg_confidence": avg_confidence,
        }

    def recognize(self, img):
        """
        Recognize all text in an image. Tries direct recognition first,
        then falls back to segmentation if needed.
        """
        # Apply EXIF rotation
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Try direct recognition (works for single lines/words)
        direct = self._recognize_direct(img)

        # Try segmentation pipeline (works for multi-line pages)
        segmented = self._recognize_segmented(img)

        # Pick the best result
        if direct and segmented:
            # Use whichever has higher average confidence
            direct_conf = direct.get("avg_confidence", 0)
            seg_conf = segmented.get("avg_confidence", 0)
            result = direct if direct_conf >= seg_conf else segmented
        elif direct:
            result = direct
        elif segmented:
            result = segmented
        else:
            return {"text": "", "lines": [], "num_lines": 0, "num_words": 0}

        # Remove internal avg_confidence from output
        result.pop("avg_confidence", None)
        return result
