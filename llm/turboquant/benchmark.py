import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.turboquant.weight_quantizer import QuantizedLinear, quantize_model


def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """Estimate model size in MB (parameters only)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def compute_perplexity(model, tokenizer, texts, max_length=512, device="cpu"):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encodings["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * (input_ids.shape[1] - 1)
            total_tokens += input_ids.shape[1] - 1

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


def benchmark_quantization(
    model_name="meta-llama/Llama-3.2-1B",
    bits=4,
    skip_layers=None,
    eval_texts=None,
    device="cpu",
):
    """
    Full benchmark: load model, quantize, compare perplexity and size.
    """
    skip_layers = skip_layers or ["lm_head", "embed_tokens"]

    if eval_texts is None:
        eval_texts = [
            "The quick brown fox jumps over the lazy dog and runs across the field.",
            "Machine learning models can be compressed using quantization techniques.",
            "Handwriting recognition requires both visual feature extraction and sequence modeling.",
            "The architecture of transformer models enables parallel processing of input sequences.",
            "Neural networks learn hierarchical representations of data through multiple layers.",
        ]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_size = model_size_mb(model)
    print(f"Original model size: {original_size:.1f} MB")
    print(f"Parameters: {count_parameters(model):,}")

    print("Computing original perplexity...")
    original_ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
    print(f"Original perplexity: {original_ppl:.2f}")

    print(f"\nQuantizing to {bits}-bit with TurboQuant...")
    start = time.time()
    quantize_model(model, bits=bits, skip_layers=skip_layers)
    quant_time = time.time() - start
    print(f"Quantization took {quant_time:.1f}s")

    quantized_size = model_size_mb(model)
    print(f"Quantized model size: {quantized_size:.1f} MB")

    print("Computing quantized perplexity...")
    quantized_ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
    print(f"Quantized perplexity: {quantized_ppl:.2f}")

    ratio = original_size / quantized_size if quantized_size > 0 else 0
    ppl_change = ((quantized_ppl - original_ppl) / original_ppl) * 100

    print(f"\n{'='*50}")
    print(f"Results ({bits}-bit TurboQuant on {model_name}):")
    print(f"  Size: {original_size:.1f} MB -> {quantized_size:.1f} MB ({ratio:.1f}x compression)")
    print(f"  Perplexity: {original_ppl:.2f} -> {quantized_ppl:.2f} ({ppl_change:+.1f}%)")
    print(f"  Quantization time: {quant_time:.1f}s")
    print(f"{'='*50}")

    return {
        "model": model_name,
        "bits": bits,
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": ratio,
        "original_perplexity": original_ppl,
        "quantized_perplexity": quantized_ppl,
        "perplexity_change_pct": ppl_change,
        "quantization_time_s": quant_time,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    benchmark_quantization(
        model_name=args.model,
        bits=args.bits,
        device=args.device,
    )
