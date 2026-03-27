import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRPredictor:
    """Handwriting recognition using TrOCR (Vision Encoder-Decoder)."""

    def __init__(self, model_name="microsoft/trocr-large-handwritten"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, img: Image.Image) -> str:
        text, _ = self.predict_with_confidence(img)
        return text

    def predict_with_confidence(self, img: Image.Image) -> tuple[str, float]:
        if img.mode != "RGB":
            img = img.convert("RGB")

        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # Compute confidence from generation scores
        if outputs.scores:
            log_probs = []
            for score in outputs.scores:
                probs = torch.softmax(score, dim=-1)
                max_prob = probs.max(dim=-1).values
                log_probs.append(max_prob.item())
            confidence = float(np.mean(log_probs)) if log_probs else 0.0
        else:
            confidence = 0.0

        return text, confidence

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        return [self.predict(img) for img in images]
