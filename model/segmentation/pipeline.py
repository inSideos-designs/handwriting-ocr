import cv2
import numpy as np
from PIL import Image, ImageOps

from model.segmentation.deskew import preprocess_page, pil_to_cv2, cv2_to_pil
from model.segmentation.lines import extract_lines


class PageRecognizer:
    """
    Full page handwriting recognition pipeline.

    Strategy:
    1. First try direct recognition on the whole image
    2. If confidence is low, segment into lines and recognize each
    3. Pick the result with higher confidence
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def _recognize_direct(self, img):
        """Try recognizing the image directly without segmentation."""
        text, confidence = self.predictor.predict_with_confidence(img)
        if not text.strip():
            return None
        return {
            "text": text,
            "lines": [{"text": text, "confidence": round(confidence, 4)}],
            "num_lines": 1,
            "avg_confidence": confidence,
        }

    def _recognize_segmented(self, img):
        """Segment the image into lines and recognize each."""
        binary, gray = preprocess_page(img)

        line_images = extract_lines(binary)
        if not line_images:
            return None

        result_lines = []
        all_confidences = []

        for line_img in line_images:
            line_inverted = 255 - line_img
            line_pil = cv2_to_pil(line_inverted)
            line_text, line_conf = self.predictor.predict_with_confidence(line_pil)

            if line_text.strip():
                result_lines.append({
                    "text": line_text,
                    "confidence": round(line_conf, 4),
                })
                all_confidences.append(line_conf)

        if not result_lines:
            return None

        full_text = "\n".join(line["text"] for line in result_lines)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        return {
            "text": full_text,
            "lines": result_lines,
            "num_lines": len(result_lines),
            "avg_confidence": avg_confidence,
        }

    def recognize(self, img):
        """
        Recognize all text in an image. Tries direct recognition first,
        then falls back to line segmentation if needed.
        """
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        direct = self._recognize_direct(img)
        segmented = self._recognize_segmented(img)

        if direct and segmented:
            direct_conf = direct.get("avg_confidence", 0)
            seg_conf = segmented.get("avg_confidence", 0)
            result = direct if direct_conf >= seg_conf else segmented
        elif direct:
            result = direct
        elif segmented:
            result = segmented
        else:
            return {"text": "", "lines": [], "num_lines": 0}

        result.pop("avg_confidence", None)
        return result
