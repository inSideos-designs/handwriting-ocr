import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import numpy as np
from PIL import Image

from model.data.dataset import NUM_CLASSES, CHARS
from model.data.preprocessing import preprocess_image
from model.networks.crnn import CRNN
from model.training.config import TrainConfig


class Predictor:
    def __init__(self, checkpoint_path: str, config: TrainConfig):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = CRNN(
            img_height=config.img_height,
            num_channels=1,
            num_classes=NUM_CLASSES,
            hidden_size=config.hidden_size,
            num_lstm_layers=config.num_lstm_layers,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, img: Image.Image) -> str:
        text, _ = self.predict_with_confidence(img)
        return text

    def predict_with_confidence(self, img: Image.Image) -> tuple[str, float]:
        tensor = preprocess_image(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        output = output.squeeze(1)
        probs = torch.exp(output)

        max_probs, indices = probs.max(dim=1)
        text, confidences = self._decode_greedy(indices.cpu().numpy(), max_probs.cpu().numpy())

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return text, avg_confidence

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        return [self.predict(img) for img in images]

    def _decode_greedy(
        self, indices: np.ndarray, probs: np.ndarray
    ) -> tuple[str, list[float]]:
        chars = []
        confidences = []
        prev_idx = -1

        for i, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:
                char_idx = int(idx) - 1
                if 0 <= char_idx < len(CHARS):
                    chars.append(CHARS[char_idx])
                    confidences.append(float(probs[i]))
            prev_idx = idx

        return "".join(chars), confidences
