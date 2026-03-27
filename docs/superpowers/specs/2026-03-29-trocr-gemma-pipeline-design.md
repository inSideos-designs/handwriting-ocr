# TrOCR + TurboQuant Gemma 2 2B Pipeline

## Summary

Replace the custom CRNN OCR model with TrOCR-large and add a TurboQuant-compressed Gemma 2 2B as an always-on error corrector. The full pipeline runs two stages: TrOCR for handwriting recognition, then Gemma for text correction. Both models load at startup.

## Motivation

The current CRNN model (~2M params) is trained from scratch with CTC decoding, which suffers from repeated-character and alignment issues. TrOCR is pretrained on handwriting data with autoregressive decoding and achieves significantly lower character error rates. Adding an LLM corrector exploits language-level context to fix residual OCR errors (e.g., "rn" vs "m", ambiguous characters). TurboQuant 4-bit compression makes the LLM deployable alongside TrOCR on consumer hardware.

## Architecture

```
Image
  |
  v
[Segmentation Pipeline]  (existing: line detection + deskewing, no word splitting)
  |
  v
Line images
  |
  v
[TrOCR-large]  (microsoft/trocr-large-handwritten, ~558M params)
  |
  v
Raw text per line
  |
  v
[Gemma 2 2B Corrector]  (google/gemma-2-2b-it, TurboQuant 4-bit, ~4.2GB)
  |
  v
Corrected text
```

Both models are loaded into memory at startup. The corrector runs on every result (no confidence gating).

## Components

### 1. TrOCRPredictor

**Location:** `model/inference/trocr_predictor.py`

Replaces `Predictor`. Same public interface:
- `predict(img: Image) -> str`
- `predict_with_confidence(img: Image) -> tuple[str, float]`
- `predict_batch(images: list[Image]) -> list[str]`

Internally uses:
- `TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")` for image preprocessing
- `VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")` for inference
- `model.generate()` with `max_new_tokens=64` for autoregressive decoding
- Confidence derived from generation log-probabilities

Device selection: CUDA > MPS > CPU (same as current).

### 2. CorrectedRecognitionService

**Location:** `backend/services/recognition.py` (replaces existing `RecognitionService`)

Composes TrOCRPredictor + OCRCorrector:
- Loads both models at init
- `recognize(img)` runs TrOCR then corrector, returns `{"text": str, "raw_text": str, "confidence": float}`
- `recognize_page(img)` runs PageRecognizer (line segmentation + TrOCR), then corrects each line
- `raw_text` field preserves the pre-correction output for debugging

The corrector uses the existing `OCRCorrector` class from `llm/inference/corrector.py`, loaded with the TurboQuant-quantized Gemma checkpoint.

### 3. Updated PageRecognizer

**Location:** `model/segmentation/pipeline.py` (modified)

Changes:
- Accepts a predictor instance instead of constructing its own CRNN
- Feeds detected lines (not words) to the predictor
- Removes word-level segmentation from the recognition path
- Line detection and deskewing remain unchanged

### 4. Updated AppConfig

**Location:** `backend/core/config.py`

New environment variables:
- `TROCR_MODEL`: HuggingFace model ID (default: `microsoft/trocr-large-handwritten`)
- `CORRECTOR_MODEL`: path to quantized Gemma checkpoint (default: `llm/checkpoints/ocr-corrector`)
- `CORRECTOR_ENABLED`: whether to run the LLM corrector (default: `true`)

Removed:
- `MODEL_CHECKPOINT` (CRNN checkpoint path)
- `MODEL_HIDDEN_SIZE` (CRNN-specific)
- `MODEL_NUM_LSTM_LAYERS` (CRNN-specific)

## Data Flow

**Single image:**
1. `CorrectedRecognitionService.recognize(img)` called
2. `TrOCRPredictor.predict_with_confidence(img)` returns raw text + confidence
3. `OCRCorrector.correct(raw_text)` returns corrected text
4. Return `{"text": corrected, "raw_text": raw, "confidence": conf}`

**Full page:**
1. `CorrectedRecognitionService.recognize_page(img)` called
2. `PageRecognizer.recognize(img)` segments into lines
3. Each line passed through TrOCR
4. Each line's raw text passed through corrector
5. Return combined result with per-line text and confidence

## What Changes

| File | Change |
|---|---|
| `model/inference/trocr_predictor.py` | New file: TrOCR wrapper |
| `backend/services/recognition.py` | Rewrite: compose TrOCR + corrector |
| `backend/core/config.py` | Update: new env vars, remove CRNN-specific ones |
| `model/segmentation/pipeline.py` | Modify: accept predictor, use line-level recognition |
| `requirements.txt` / `pyproject.toml` | Add: `transformers`, `pillow` (if not present) |

## What Stays

- Segmentation pipeline (line detection, deskewing)
- Backend API routes and response format
- LLM corrector inference code (`llm/inference/corrector.py`)
- TurboQuant quantization code (`llm/turboquant/`)
- CRNN code (kept but no longer default)

## Dependencies

- `transformers>=4.30.0` (for TrOCR and Gemma)
- `torch>=2.1.0`
- `Pillow>=10.0.0`
- `accelerate` (for Gemma loading)

## Testing

- Unit test: TrOCRPredictor returns non-empty string for a sample handwriting image
- Unit test: CorrectedRecognitionService.recognize returns dict with text, raw_text, confidence
- Integration test: full page recognition produces reasonable multi-line output
- Comparison test: run same images through old CRNN and new TrOCR, log accuracy difference
