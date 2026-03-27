import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    trocr_model: str = ""
    corrector_model: str = ""
    corrector_enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024
    allowed_types: tuple = ("image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp")

    def __post_init__(self):
        if not self.trocr_model:
            self.trocr_model = os.environ.get(
                "TROCR_MODEL", "microsoft/trocr-large-handwritten"
            )
        if not self.corrector_model:
            self.corrector_model = os.environ.get(
                "CORRECTOR_MODEL", "llm/checkpoints/ocr-corrector"
            )
        env_enabled = os.environ.get("CORRECTOR_ENABLED", "true")
        self.corrector_enabled = env_enabled.lower() not in ("false", "0", "no")
