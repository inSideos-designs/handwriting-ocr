<p align="center">
  <h1 align="center">Handwriting OCR</h1>
  <p align="center">
    On-device handwriting recognition from phone camera photos, powered by compressed neural networks.
  </p>
</p>

<p align="center">
  <a href="#pipeline-architecture">Architecture</a> &bull;
  <a href="#compression-results">Compression</a> &bull;
  <a href="#getting-started">Getting Started</a> &bull;
  <a href="#api-reference">API</a> &bull;
  <a href="#training">Training</a>
</p>

---

## Overview

This project implements a two-stage handwriting recognition pipeline designed to run entirely on consumer hardware:

1. **Surya OCR** detects and recognizes handwritten text from phone camera photos
2. **Gemma 2 2B**, compressed to 4-bit using TurboQuant, corrects OCR errors using language-level context

The entire pipeline fits in **~5 GB of memory** — down from 11.4 GB — enabling deployment on laptops, phones, and edge devices without cloud dependencies.

## Pipeline Architecture

```
                          +-----------------------+
                          |    Phone Camera Photo  |
                          +-----------+-----------+
                                      |
                                      v
                          +-----------+-----------+
                          |    Photo Preprocessing |
                          |  - Shadow removal      |
                          |  - Contrast enhancement |
                          +-----------+-----------+
                                      |
                          +-----------+-----------+
                          |       Surya OCR        |
                          |  Detection: EfficientViT|
                          |  Recognition: SuryaModel|
                          +-----------+-----------+
                                      |
                                      v
                              Raw OCR Text
                                      |
                          +-----------+-----------+
                          |  Gemma 2 2B Corrector  |
                          |  TurboQuant 4-bit      |
                          |  compressed (2.4x)     |
                          +-----------+-----------+
                                      |
                                      v
                              Corrected Text
```

## Compression Results

All models are compressed using **TurboQuant**, a weight quantization method based on random rotation + optimal scalar quantization ([QJL paper](https://arxiv.org/abs/2406.03482)).

### Pipeline Memory Footprint

| Component | Model | Parameters | Original Size | Compressed (4-bit) | Ratio |
|:--|:--|--:|--:|--:|--:|
| Text Detection | Surya EfficientViT | 38M | 73 MB | 73 MB | 1.0x |
| Text Recognition | Surya Foundation | 719M | 1,372 MB | 788 MB | 1.7x |
| Error Correction | Gemma 2 2B | 2,614M | 9,973 MB | 4,183 MB | 2.4x |
| **Total** | | **3,371M** | **11,418 MB** | **5,044 MB** | **2.3x** |

### TurboQuant Quality Impact

| Model | Metric | Original | Compressed | Degradation |
|:--|:--|--:|--:|--:|
| Gemma 2 2B | Perplexity | 51.48 | 57.20 | +11.1% |

The +11.1% perplexity increase is acceptable for OCR error correction, where the model only needs to fix character-level mistakes — not generate creative text.

### How TurboQuant Works

1. **Random Rotation** — Weight rows are multiplied by a random orthogonal matrix (QR decomposition), distributing information uniformly across coordinates
2. **Max-Lloyd Codebook** — An optimal scalar quantizer is computed for the rotated coordinate distribution using the Max-Lloyd algorithm
3. **4-bit Quantization** — Each coordinate is mapped to one of 16 centroids, stored as `int8` indices alongside `float16` row norms

This achieves compression without requiring calibration data, fine-tuning, or model-specific tuning.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.1+
- 8 GB RAM minimum (for compressed inference)

### Installation

```bash
git clone https://github.com/inSideos-designs/handwriting-ocr.git
cd handwriting-ocr

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install surya-ocr accelerate
```

### Quick Test

```python
from PIL import Image
from model.inference.surya_predictor import SuryaPredictor

predictor = SuryaPredictor()

img = Image.open("path/to/handwriting.jpg")
text, confidence = predictor.predict_with_confidence(img)
print(f"Recognized: {text} (confidence: {confidence:.2f})")
```

### Running the Backend

```bash
# Start the API server
uvicorn backend.main:app --reload

# The API is now available at http://localhost:8000
```

## API Reference

### `POST /api/recognize`

Upload an image for single-region handwriting recognition.

**Request:** `multipart/form-data` with `file` field (PNG, JPEG, TIFF, BMP)

**Response:**
```json
{
  "text": "Hello World",
  "raw_text": "Helo Wrld",
  "confidence": 0.85
}
```

- `text` — corrected output (after LLM error correction)
- `raw_text` — raw OCR output (before correction)
- `confidence` — OCR model confidence (0.0 to 1.0)

### `POST /api/recognize/page`

Upload a full page image for multi-line recognition.

**Response:**
```json
{
  "text": "First line\nSecond line",
  "lines": [
    {"text": "First line", "raw_text": "Flrst line", "confidence": 0.92},
    {"text": "Second line", "raw_text": "Second 1ine", "confidence": 0.88}
  ],
  "num_lines": 2
}
```

### `GET /api/health`

Health check endpoint. Returns `{"status": "healthy"}`.

### Configuration

| Variable | Default | Description |
|:--|:--|:--|
| `CORRECTOR_MODEL` | `llm/checkpoints/ocr-corrector` | Path to fine-tuned Gemma corrector checkpoint |
| `CORRECTOR_ENABLED` | `true` | Set to `false` to disable LLM error correction |
| `TROCR_MODEL` | `microsoft/trocr-large-handwritten` | HuggingFace model ID (if using TrOCR backend) |

## Training

### Fine-tuning the Error Corrector

The Gemma 2 2B corrector is fine-tuned on a mix of synthetic and real OCR error pairs:

- **Synthetic data** — 60+ sentence templates with character-level corruptions mimicking common OCR errors (`m`/`rn`, `d`/`cl`, `O`/`0`, etc.)
- **Real OCR data** — IAM handwriting dataset images processed through Surya, capturing actual recognition errors

Training runs on a Google Colab T4 GPU:

```bash
# Open the training notebook
# notebooks/train_gemma_corrector.ipynb
```

Or train locally (requires GPU with 16GB+ VRAM):

```bash
python -m llm.finetune.train \
  --model google/gemma-2-2b-it \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4
```

### Running Compression Benchmarks

```bash
# Benchmark TurboQuant on Gemma 2 2B
python -m llm.turboquant.benchmark --model google/gemma-2-2b --bits 4

# Benchmark TurboQuant on Surya foundation model
python -m llm.turboquant.benchmark_surya
```

## Project Structure

```
handwriting-ocr/
  model/
    inference/
      surya_predictor.py          Surya OCR wrapper (detection + recognition)
      trocr_predictor.py          TrOCR wrapper (for scanned documents)
      predictor.py                Legacy CRNN predictor
    segmentation/
      photo_preprocess.py         Shadow removal, contrast enhancement for photos
      pipeline.py                 Page recognition with line segmentation
      deskew.py                   Skew correction via Hough transform
      lines.py                    Horizontal projection line extraction
    networks/
      crnn.py                     CNN-BiLSTM-CTC architecture
    data/                         Dataset loaders, preprocessing, character sets
    training/                     Training loops, configs, schedulers

  llm/
    turboquant/
      quantizer.py                Core TurboQuantizer (rotation + codebook)
      weight_quantizer.py         QuantizedLinear drop-in for nn.Linear
      codebook.py                 Max-Lloyd optimal scalar quantizer
      benchmark.py                LLM compression benchmark
      benchmark_surya.py          Surya model compression benchmark
    finetune/
      dataset.py                  OCR error pair generation (synthetic + real)
      train.py                    Gemma fine-tuning script
    inference/
      corrector.py                OCR error correction at inference time

  backend/
    services/recognition.py       CorrectedRecognitionService (Surya + Gemma)
    api/routes.py                 FastAPI REST endpoints
    core/config.py                Environment-based configuration
    main.py                       Application factory

  notebooks/
    train_gemma_corrector.ipynb   Colab notebook for corrector fine-tuning
    train_combined.ipynb          Colab notebook for CRNN training

  tests/                          pytest suite (27 tests)

  docs/
    superpowers/
      specs/                      Design specifications
      plans/                      Implementation plans
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/backend/ -v          # API and service tests
pytest tests/model/ -v            # OCR model tests
```

## Docker

```bash
docker compose up --build
```

## License

MIT
