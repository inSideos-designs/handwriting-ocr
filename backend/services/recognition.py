from PIL import Image

from model.inference.predictor import Predictor
from model.training.config import TrainConfig
from backend.core.config import AppConfig


class RecognitionService:
    def __init__(self, app_config: AppConfig):
        train_config = TrainConfig(
            hidden_size=app_config.model_hidden_size,
            num_lstm_layers=app_config.model_num_lstm_layers,
        )
        self.predictor = Predictor(app_config.model_checkpoint, train_config)

    def recognize(self, img: Image.Image) -> dict:
        text, confidence = self.predictor.predict_with_confidence(img)
        return {"text": text, "confidence": round(confidence, 4)}
