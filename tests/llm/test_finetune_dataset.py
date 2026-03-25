import pytest
from llm.finetune.dataset import corrupt_text, generate_training_pairs, CHAR_SUBSTITUTIONS


class TestCorruptText:
    def test_returns_string(self):
        result = corrupt_text("HELLO")
        assert isinstance(result, str)

    def test_high_error_rate_changes_text(self):
        changed = False
        for _ in range(20):
            if corrupt_text("HELLO", error_rate=0.5) != "HELLO":
                changed = True
                break
        assert changed

    def test_zero_error_rate_preserves(self):
        assert corrupt_text("HELLO", error_rate=0.0) == "HELLO"

    def test_substitution_uses_mapping(self):
        results = set()
        for _ in range(100):
            results.add(corrupt_text("O", error_rate=1.0))
        assert "0" in results or "" in results or "OO" in results


class TestGenerateTrainingPairs:
    def test_returns_pairs(self):
        pairs = generate_training_pairs(["HELLO", "WORLD"], num_augmentations=3)
        assert len(pairs) > 0
        for corrupted, clean in pairs:
            assert isinstance(corrupted, str)
            assert isinstance(clean, str)
            assert corrupted != clean

    def test_num_augmentations(self):
        pairs = generate_training_pairs(["HELLO"], num_augmentations=10, error_rate=0.5)
        assert len(pairs) <= 10
        assert len(pairs) >= 1
