from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    img_height: int = 32
    img_width: int = 128
    hidden_size: int = 256
    num_lstm_layers: int = 2
    dropout: float = 0.1
    patience: int = 5
    beam_width: int = 10
    data_dir: str = ""
    csv_path: str = ""
    checkpoint_dir: str = "model/checkpoints"
