import cv2
import numpy as np
from PIL import Image

from model.segmentation.deskew import preprocess_page, pil_to_cv2, cv2_to_pil
from model.segmentation.lines import extract_lines
from model.segmentation.words import extract_words
from model.inference.predictor import Predictor
from model.training.config import TrainConfig


class PageRecognizer:
    """
    Full page handwriting recognition pipeline.

    Preprocesses a page image, segments into lines and words,
    runs CRNN inference on each word, and returns structured results.
    """

    def __init__(self, checkpoint_path, config=None):
        if config is None:
            config = TrainConfig()
        self.predictor = Predictor(checkpoint_path, config)

    def _try_orientation(self, img):
        """
        Try all 4 rotations and pick the one that produces the best segmentation.
        Best = most lines with reasonable word counts.
        """
        best_result = None
        best_score = -1

        for angle in [0, 90, 180, 270]:
            rotated = img.rotate(angle, expand=True) if angle != 0 else img
            binary, gray = preprocess_page(rotated)
            lines = extract_lines(binary)

            # Score: prefer orientations with multiple lines containing 1-5 words each
            score = 0
            for line_img in lines:
                words = extract_words(line_img)
                if 1 <= len(words) <= 10:
                    score += len(words)

            if score > best_score:
                best_score = score
                best_result = (binary, gray, rotated)

        return best_result

    def recognize(self, img):
        """
        Recognize all text in a page image.

        Args:
            img: PIL Image of a handwritten page

        Returns:
            dict with:
                - "text": full recognized text
                - "lines": list of line dicts, each with "text", "words"
                - "words": each word has "text", "confidence"
        """
        # Apply EXIF rotation if present
        from PIL import ImageOps
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Try all orientations and pick the best
        binary, gray_deskewed, img = self._try_orientation(img)

        # Extract lines
        line_images = extract_lines(binary)

        if not line_images:
            return {"text": "", "lines": [], "num_lines": 0, "num_words": 0}

        result_lines = []
        all_words_text = []

        for line_img in line_images:
            # Extract words from this line
            word_images = extract_words(line_img)

            if not word_images:
                continue

            line_words = []
            for word_img in word_images:
                # Convert binary word image to PIL for the predictor
                # Invert: CRNN expects dark text on light background
                word_inverted = 255 - word_img
                word_pil = cv2_to_pil(word_inverted)

                text, confidence = self.predictor.predict_with_confidence(word_pil)

                if text.strip():
                    line_words.append({
                        "text": text,
                        "confidence": round(confidence, 4),
                    })

            if line_words:
                line_text = " ".join(w["text"] for w in line_words)
                avg_confidence = sum(w["confidence"] for w in line_words) / len(line_words)
                result_lines.append({
                    "text": line_text,
                    "confidence": round(avg_confidence, 4),
                    "words": line_words,
                })
                all_words_text.append(line_text)

        full_text = "\n".join(all_words_text)
        total_words = sum(len(line["words"]) for line in result_lines)

        return {
            "text": full_text,
            "lines": result_lines,
            "num_lines": len(result_lines),
            "num_words": total_words,
        }
