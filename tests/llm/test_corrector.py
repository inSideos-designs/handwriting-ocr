import pytest
from llm.inference.corrector import OCRCorrector


class TestOCRCorrector:
    def test_correct_returns_string(self):
        # Skip if no model checkpoint available
        pytest.skip("Requires fine-tuned model checkpoint")

    def test_correct_batch(self):
        pytest.skip("Requires fine-tuned model checkpoint")
