# TrOCR + TurboQuant Gemma Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CRNN OCR model with TrOCR-large and wire in a TurboQuant-compressed Gemma 2 2B as an always-on error corrector, creating a two-stage handwriting recognition pipeline.

**Architecture:** TrOCRPredictor wraps HuggingFace's TrOCR-large-handwritten model behind the same predict/predict_with_confidence interface. CorrectedRecognitionService composes TrOCRPredictor + OCRCorrector. PageRecognizer is updated to accept any predictor and use line-level recognition (dropping word segmentation fallback).

**Tech Stack:** PyTorch, HuggingFace transformers (TrOCR, Gemma), existing TurboQuant quantizer, pytest

**Spec:** `docs/superpowers/specs/2026-03-29-trocr-gemma-pipeline-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `model/inference/trocr_predictor.py` | Create | TrOCR-large wrapper with predict/predict_with_confidence interface |
| `tests/model/test_trocr_predictor.py` | Create | Unit tests for TrOCRPredictor |
| `backend/core/config.py` | Modify | New env vars for TrOCR + corrector, remove CRNN-specific ones |
| `tests/backend/test_config.py` | Create | Unit tests for updated AppConfig |
| `model/segmentation/pipeline.py` | Modify | Accept predictor, line-level only, drop word fallback |
| `tests/model/test_pipeline.py` | Create | Unit tests for updated PageRecognizer |
| `backend/services/recognition.py` | Modify | Compose TrOCR + corrector into CorrectedRecognitionService |
| `tests/backend/test_recognition.py` | Create | Unit tests for CorrectedRecognitionService |
| `requirements.txt` | Modify | Add accelerate dependency |

---

### Task 1: TrOCRPredictor

**Files:**
- Create: `model/inference/trocr_predictor.py`
- Create: `tests/model/test_trocr_predictor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_trocr_predictor.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch


class TestTrOCRPredictor:
    def _make_predictor(self):
        """Create predictor with mocked model loading."""
        with patch("model.inference.trocr_predictor.VisionEncoderDecoderModel") as mock_model_cls, \
             patch("model.inference.trocr_predictor.TrOCRProcessor") as mock_proc_cls:

            mock_processor = MagicMock()
            mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 384, 384)}
            mock_processor.batch_decode.return_value = ["Hello World"]
            mock_proc_cls.from_pretrained.return_value = mock_processor

            mock_model = MagicMock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
            mock_model.to.return_value = mock_model
            mock_model_cls.from_pretrained.return_value = mock_model

            from model.inference.trocr_predictor import TrOCRPredictor
            predictor = TrOCRPredictor.__new__(TrOCRPredictor)
            predictor.device = torch.device("cpu")
            predictor.processor = mock_processor
            predictor.model = mock_model

        return predictor

    def test_predict_returns_string(self):
        predictor = self._make_predictor()
        img = Image.new("RGB", (200, 50), color="white")
        result = predictor.predict(img)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_predict_with_confidence_returns_tuple(self):
        predictor = self._make_predictor()
        img = Image.new("RGB", (200, 50), color="white")
        text, confidence = predictor.predict_with_confidence(img)
        assert isinstance(text, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_batch_returns_list(self):
        predictor = self._make_predictor()
        images = [Image.new("RGB", (200, 50)) for _ in range(3)]
        results = predictor.predict_batch(images)
        assert isinstance(results, list)
        assert len(results) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/model/test_trocr_predictor.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'model.inference.trocr_predictor'`

- [ ] **Step 3: Write the implementation**

Create `model/inference/trocr_predictor.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/model/test_trocr_predictor.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add model/inference/trocr_predictor.py tests/model/test_trocr_predictor.py
git commit -m "feat: add TrOCRPredictor with predict/predict_with_confidence interface"
```

---

### Task 2: Update AppConfig

**Files:**
- Modify: `backend/core/config.py`
- Create: `tests/backend/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backend/test_config.py`:

```python
import os
import pytest
from backend.core.config import AppConfig


class TestAppConfig:
    def test_default_trocr_model(self):
        config = AppConfig()
        assert config.trocr_model == "microsoft/trocr-large-handwritten"

    def test_default_corrector_model(self):
        config = AppConfig()
        assert config.corrector_model == "llm/checkpoints/ocr-corrector"

    def test_corrector_enabled_default(self):
        config = AppConfig()
        assert config.corrector_enabled is True

    def test_env_override_trocr_model(self, monkeypatch):
        monkeypatch.setenv("TROCR_MODEL", "microsoft/trocr-small-handwritten")
        config = AppConfig()
        assert config.trocr_model == "microsoft/trocr-small-handwritten"

    def test_env_override_corrector_enabled_false(self, monkeypatch):
        monkeypatch.setenv("CORRECTOR_ENABLED", "false")
        config = AppConfig()
        assert config.corrector_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/backend/test_config.py -v`

Expected: FAIL with `AttributeError: 'AppConfig' object has no attribute 'trocr_model'`

- [ ] **Step 3: Update AppConfig**

Replace contents of `backend/core/config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/backend/test_config.py -v`

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/core/config.py tests/backend/test_config.py
git commit -m "feat: update AppConfig for TrOCR + Gemma corrector"
```

---

### Task 3: Update PageRecognizer

**Files:**
- Modify: `model/segmentation/pipeline.py`
- Create: `tests/model/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_pipeline.py`:

```python
import pytest
from unittest.mock import MagicMock
from PIL import Image

from model.segmentation.pipeline import PageRecognizer


class TestPageRecognizer:
    def _make_recognizer(self):
        """Create PageRecognizer with a mock predictor."""
        mock_predictor = MagicMock()
        mock_predictor.predict_with_confidence.return_value = ("Hello World", 0.95)
        recognizer = PageRecognizer(predictor=mock_predictor)
        return recognizer, mock_predictor

    def test_accepts_predictor_argument(self):
        mock_predictor = MagicMock()
        recognizer = PageRecognizer(predictor=mock_predictor)
        assert recognizer.predictor is mock_predictor

    def test_recognize_direct_returns_dict(self):
        recognizer, _ = self._make_recognizer()
        img = Image.new("RGB", (200, 50), color="white")
        result = recognizer.recognize(img)
        assert "text" in result
        assert "lines" in result

    def test_recognize_uses_injected_predictor(self):
        recognizer, mock_predictor = self._make_recognizer()
        img = Image.new("RGB", (200, 50), color="white")
        recognizer.recognize(img)
        assert mock_predictor.predict_with_confidence.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/model/test_pipeline.py -v`

Expected: FAIL with `TypeError` (PageRecognizer doesn't accept `predictor` kwarg)

- [ ] **Step 3: Update PageRecognizer**

Replace contents of `model/segmentation/pipeline.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/model/test_pipeline.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add model/segmentation/pipeline.py tests/model/test_pipeline.py
git commit -m "refactor: PageRecognizer accepts injected predictor, line-level only"
```

---

### Task 4: CorrectedRecognitionService

**Files:**
- Modify: `backend/services/recognition.py`
- Create: `tests/backend/test_recognition.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backend/test_recognition.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.services.recognition import CorrectedRecognitionService


class TestCorrectedRecognitionService:
    def _make_service(self):
        """Create service with mocked TrOCR and corrector."""
        mock_predictor = MagicMock()
        mock_predictor.predict_with_confidence.return_value = ("Helo Wrld", 0.85)

        mock_corrector = MagicMock()
        mock_corrector.correct.return_value = "Hello World"

        service = CorrectedRecognitionService.__new__(CorrectedRecognitionService)
        service.predictor = mock_predictor
        service.corrector = mock_corrector
        service.page_recognizer = MagicMock()
        service.page_recognizer.recognize.return_value = {
            "text": "Line one\nLine two",
            "lines": [
                {"text": "Line one", "confidence": 0.9},
                {"text": "Line two", "confidence": 0.8},
            ],
            "num_lines": 2,
        }
        return service

    def test_recognize_returns_text_and_raw_text(self):
        service = self._make_service()
        img = Image.new("RGB", (200, 50), color="white")
        result = service.recognize(img)
        assert result["raw_text"] == "Helo Wrld"
        assert result["text"] == "Hello World"
        assert "confidence" in result

    def test_recognize_calls_corrector(self):
        service = self._make_service()
        img = Image.new("RGB", (200, 50), color="white")
        service.recognize(img)
        service.corrector.correct.assert_called_once_with("Helo Wrld")

    def test_recognize_page_corrects_each_line(self):
        service = self._make_service()
        service.corrector.correct.side_effect = ["Line one corrected", "Line two corrected"]
        img = Image.new("RGB", (400, 200), color="white")
        result = service.recognize_page(img)
        assert service.corrector.correct.call_count == 2
        assert "Line one corrected" in result["text"]
        assert "Line two corrected" in result["text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/backend/test_recognition.py -v`

Expected: FAIL with `ImportError: cannot import name 'CorrectedRecognitionService'`

- [ ] **Step 3: Write the implementation**

Replace contents of `backend/services/recognition.py`:

```python
from PIL import Image

from model.inference.trocr_predictor import TrOCRPredictor
from model.segmentation.pipeline import PageRecognizer
from llm.inference.corrector import OCRCorrector
from backend.core.config import AppConfig


class CorrectedRecognitionService:
    """Two-stage recognition: TrOCR for OCR, then Gemma for error correction."""

    def __init__(self, app_config: AppConfig):
        self.predictor = TrOCRPredictor(model_name=app_config.trocr_model)
        self.page_recognizer = PageRecognizer(predictor=self.predictor)

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
        result = self.page_recognizer.recognize(img)

        for line in result.get("lines", []):
            line["raw_text"] = line["text"]
            line["text"] = self._correct(line["text"])

        corrected_lines = [line["text"] for line in result.get("lines", [])]
        result["text"] = "\n".join(corrected_lines)

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/backend/test_recognition.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/services/recognition.py tests/backend/test_recognition.py
git commit -m "feat: CorrectedRecognitionService with TrOCR + Gemma corrector"
```

---

### Task 5: Update dependencies and run all tests

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add accelerate to requirements.txt**

Add `accelerate>=0.25.0` to `requirements.txt` after the `datasets` line.

- [ ] **Step 2: Install dependencies**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/pip install accelerate`

- [ ] **Step 3: Run all tests**

Run: `cd "/Users/cultistsid/Documents/Personal Projects/handwriting-ocr" && .venv/bin/python -m pytest tests/ -v`

Expected: All tests pass (existing preprocessing tests + all new tests)

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add accelerate dependency for Gemma model loading"
```
