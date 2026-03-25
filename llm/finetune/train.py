import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from llm.finetune.dataset import OCRCorrectionDataset, generate_training_pairs, corrupt_text


# Common English names for training data generation
SAMPLE_NAMES = [
    "BALTHAZAR", "SIMON", "ELIZABETH", "WILLIAM", "CATHERINE",
    "ALEXANDER", "MARGARET", "BENJAMIN", "CHARLOTTE", "NATHANIEL",
    "VICTORIA", "THEODORE", "JOSEPHINE", "CHRISTOPHER", "ANASTASIA",
    "MONTGOMERY", "KATHERINE", "FREDERICK", "GENEVIEVE", "SEBASTIAN",
    "ISABELLE", "MAXIMILIAN", "PENELOPE", "ARCHIBALD", "ROSALIND",
    "BARTHOLOMEW", "EVANGELINE", "CORNELIUS", "HENRIETTA", "AUGUSTINE",
    "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES",
    "GARCIA", "MILLER", "DAVIS", "RODRIGUEZ", "MARTINEZ",
    "HERNANDEZ", "LOPEZ", "GONZALEZ", "WILSON", "ANDERSON",
    "THOMAS", "TAYLOR", "MOORE", "JACKSON", "MARTIN",
    "HELLO", "WORLD", "TEST", "DOCUMENT", "PROCESSING",
    "HANDWRITING", "RECOGNITION", "NEURAL", "NETWORK", "MACHINE",
]


def train_corrector(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir="llm/checkpoints",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=128,
    num_augmentations=5,
    error_rate=0.15,
    device=None,
):
    """Fine-tune a causal LM for OCR error correction using LoRA-style training."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    ).to(device)

    # Freeze all parameters except the last 2 transformer layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h

    if layers is not None:
        for layer in layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Unfreeze lm_head
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Generate training data
    print("Generating training pairs...")
    pairs = generate_training_pairs(SAMPLE_NAMES, num_augmentations=num_augmentations, error_rate=error_rate)
    print(f"Generated {len(pairs)} training pairs")

    dataset = OCRCorrectionDataset(pairs, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=num_epochs * len(loader)
    )

    os.makedirs(output_dir, exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    save_path = os.path.join(output_dir, "ocr-corrector")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    return save_path


if __name__ == "__main__":
    train_corrector(num_epochs=3)
