# Handwriting OCR

A two-stage handwriting recognition pipeline that combines **Surya OCR** with a **TurboQuant-compressed Gemma 2 2B** error corrector, designed to run on consumer hardware.

## Pipeline

```
Phone Photo
    |
    v
[Preprocessing]         Shadow removal, contrast enhancement
    |
    v
[Surya OCR]             Text detection (EfficientViT) + recognition (SuryaModel)
    |
    v
Raw OCR text
    |
    v
[Gemma 2 2B Corrector]  TurboQuant 4-bit compressed LLM error correction
    |
    v
Clean text
```

## Model Sizes

| Component | Model | Params | Original | TurboQuant 4-bit |
|---|---|---|---|---|
| Text Detection | Surya EfficientViT | 38M | 73 MB | 73 MB (not quantized) |
| Text Recognition | Surya Foundation | 719M | 1,372 MB | 788 MB (1.7x) |
| Error Corrector | Gemma 2 2B | 2,614M | 9,973 MB | 4,183 MB (2.4x) |
| **Total** | | **3,371M** | **11,418 MB** | **5,044 MB** |

**Total pipeline: ~5 GB compressed** (down from ~11.4 GB, 2.3x overall compression).

## TurboQuant Compression

TurboQuant implements random rotation + optimal scalar quantization for weight compression, based on the [QJL paper](https://arxiv.org/abs/2406.03482):

1. **Random rotation** via QR decomposition — Gaussianizes weight distributions
2. **Max-Lloyd codebook** — optimal scalar quantizer for the rotated coordinates
3. **4-bit quantization** — 16 centroids per coordinate, stored as int8 indices + float16 row norms

Results on Gemma 2 2B:
- Compression: 9,973 MB to 4,183 MB (2.4x)
- Perplexity: 51.48 to 57.20 (+11.1%) — acceptable for OCR error correction

Key files:
- `llm/turboquant/quantizer.py` — Core TurboQuantizer with rotation + codebook quantization
- `llm/turboquant/weight_quantizer.py` — QuantizedLinear module, replaces nn.Linear in-place
- `llm/turboquant/codebook.py` — Max-Lloyd optimal scalar quantizer
- `llm/turboquant/benchmark.py` — Gemma 2 2B compression benchmark
- `llm/turboquant/benchmark_surya.py` — Surya foundation model compression benchmark

## Project Structure

```
model/
  inference/
    surya_predictor.py      # Surya OCR wrapper (detection + recognition)
    trocr_predictor.py      # TrOCR wrapper (alternative, for scanned docs)
    predictor.py            # Legacy CRNN predictor
  segmentation/
    photo_preprocess.py     # Shadow removal, contrast enhancement
    pipeline.py             # Page recognition with line segmentation
    deskew.py               # Skew detection and correction
    lines.py                # Line extraction
  networks/
    crnn.py                 # Legacy CRNN model
  data/                     # Dataset loaders and preprocessing
  training/                 # Training scripts and config

llm/
  turboquant/               # TurboQuant weight quantization
  finetune/                 # LLM fine-tuning for OCR error correction
  inference/
    corrector.py            # OCR error correction using fine-tuned LLM

backend/
  services/
    recognition.py          # CorrectedRecognitionService (Surya + Gemma)
  api/
    routes.py               # FastAPI endpoints
  core/
    config.py               # AppConfig with environment variable overrides

tests/                      # pytest test suite
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install surya-ocr accelerate
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CORRECTOR_MODEL` | `llm/checkpoints/ocr-corrector` | Path to fine-tuned Gemma corrector |
| `CORRECTOR_ENABLED` | `true` | Enable/disable LLM error correction |
| `TROCR_MODEL` | `microsoft/trocr-large-handwritten` | TrOCR model (if using TrOCR instead of Surya) |

## Development

```bash
# Run tests
pytest tests/ -v

# Start backend
uvicorn backend.main:app --reload

# Run TurboQuant benchmark on Gemma
python -m llm.turboquant.benchmark --model google/gemma-2-2b --bits 4

# Run TurboQuant benchmark on Surya
python -m llm.turboquant.benchmark_surya
```

## Docker

```bash
docker compose up --build
```
