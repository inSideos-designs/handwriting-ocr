import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.finetune.dataset import PROMPT_TEMPLATE


class OCRCorrector:
    """Corrects OCR errors using a fine-tuned LLM."""

    def __init__(self, model_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32
        ).to(device)
        self.model.eval()

    def correct(self, text, max_new_tokens=64):
        """Correct OCR errors in the given text."""
        prompt = f"Correct the OCR errors in the following text:\n{text}\n\nCorrected:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the corrected text after "Corrected:\n"
        marker = "Corrected:\n"
        if marker in full_text:
            corrected = full_text.split(marker)[-1].strip()
            # Take only the first line
            corrected = corrected.split("\n")[0].strip()
            return corrected

        return text  # fallback to original if parsing fails

    def correct_batch(self, texts, max_new_tokens=64):
        """Correct OCR errors in a batch of texts."""
        return [self.correct(t, max_new_tokens) for t in texts]
