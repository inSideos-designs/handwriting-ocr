import torch
import pytest

from model.networks.crnn import CRNN
from model.data.dataset import NUM_CLASSES


class TestCRNN:
    def setup_method(self):
        self.model = CRNN(
            img_height=32,
            num_channels=1,
            num_classes=NUM_CLASSES,
            hidden_size=256,
            num_lstm_layers=2,
            dropout=0.1,
        )
        self.model.eval()

    def test_output_shape(self):
        batch = torch.randn(4, 1, 32, 128)
        output = self.model(batch)
        assert output.shape[1] == 4
        assert output.shape[2] == NUM_CLASSES
        assert output.shape[0] > 0

    def test_single_image(self):
        batch = torch.randn(1, 1, 32, 128)
        output = self.model(batch)
        assert output.shape[1] == 1
        assert output.shape[2] == NUM_CLASSES

    def test_output_is_log_softmax(self):
        batch = torch.randn(2, 1, 32, 128)
        output = self.model(batch)
        assert output.max().item() <= 0.0 + 1e-5
        probs = torch.exp(output)
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_different_width_inputs(self):
        for width in [64, 128, 256]:
            batch = torch.randn(2, 1, 32, width)
            output = self.model(batch)
            assert output.shape[1] == 2
            assert output.shape[2] == NUM_CLASSES

    def test_parameter_count_reasonable(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 100_000
        assert total < 50_000_000
