import os
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    model_checkpoint: str = ""
    model_hidden_size: int = 0
    model_num_lstm_layers: int = 0
    max_file_size: int = 10 * 1024 * 1024
    allowed_types: tuple = ("image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp")

    def __post_init__(self):
        if not self.model_checkpoint:
            self.model_checkpoint = os.environ.get(
                "MODEL_CHECKPOINT", "model/checkpoints/best_model.pt"
            )
        if not self.model_hidden_size:
            self.model_hidden_size = int(os.environ.get("MODEL_HIDDEN_SIZE", "256"))
        if not self.model_num_lstm_layers:
            self.model_num_lstm_layers = int(os.environ.get("MODEL_NUM_LSTM_LAYERS", "2"))
