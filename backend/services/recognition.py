from PIL import Image

from model.inference.surya_predictor import SuryaPredictor
from llm.inference.corrector import OCRCorrector
from backend.core.config import AppConfig


class CorrectedRecognitionService:
    """Two-stage recognition: Surya OCR, then Gemma for error correction."""

    def __init__(self, app_config: AppConfig):
        self.predictor = SuryaPredictor()

        if app_config.corrector_enabled:
            self.corrector = OCRCorrector(model_path=app_config.corrector_model)
        else:
            self.corrector = None

    def _correct(self, text: str) -> str:
        if self.corrector and text.strip():
            return self.corrector.correct(text)
        return text

    def recognize(self, img: Image.Image) -> dict:
        raw_text, confidence = self.predictor.predict_with_confidence(img)
        corrected = self._correct(raw_text)
        return {
            "text": corrected,
            "raw_text": raw_text,
            "confidence": round(confidence, 4),
        }

    def recognize_page(self, img: Image.Image) -> dict:
        result = self.predictor.predict_page(img)

        for line in result.get("lines", []):
            line["raw_text"] = line["text"]
            line["text"] = self._correct(line["text"])

        corrected_lines = [line["text"] for line in result.get("lines", [])]
        result["text"] = "\n".join(corrected_lines)

        return result
