"""Benchmark TurboQuant compression on Surya's foundation/recognition model."""

import time
import torch
from surya.foundation import FoundationPredictor

from llm.turboquant.weight_quantizer import quantize_model


def model_size_mb(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def benchmark_surya_quantization(bits=4):
    print("Loading Surya foundation model...")
    foundation = FoundationPredictor(device="cpu")
    model = foundation.model

    original_size = model_size_mb(model)
    params = sum(p.numel() for p in model.parameters())
    print(f"Original size: {original_size:.1f} MB")
    print(f"Parameters: {params:,}")

    # Skip embedding layers and lm_head (if present)
    skip_layers = ["embed", "lm_head", "head"]

    print(f"\nQuantizing to {bits}-bit with TurboQuant...")
    start = time.time()
    model, stats = quantize_model(model, bits=bits, skip_layers=skip_layers)
    quant_time = time.time() - start
    print(f"Quantization took {quant_time:.1f}s")
    print(f"Layers quantized: {stats['layers_quantized']}, skipped: {stats['layers_skipped']}")

    quantized_size = model_size_mb(model)
    ratio = original_size / quantized_size if quantized_size > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results ({bits}-bit TurboQuant on Surya Foundation Model):")
    print(f"  Size: {original_size:.1f} MB -> {quantized_size:.1f} MB ({ratio:.1f}x compression)")
    print(f"  Quantization time: {quant_time:.1f}s")
    print(f"{'='*60}")

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": ratio,
    }


if __name__ == "__main__":
    benchmark_surya_quantization(bits=4)
