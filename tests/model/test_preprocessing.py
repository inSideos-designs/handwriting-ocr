import numpy as np
import pytest
from PIL import Image

from model.data.preprocessing import (
    convert_to_grayscale,
    resize_and_pad,
    normalize_image,
    preprocess_image,
)

IMG_HEIGHT = 32
IMG_WIDTH = 128


def make_test_image(width=200, height=50, mode="RGB"):
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


class TestConvertToGrayscale:
    def test_rgb_to_grayscale(self):
        img = make_test_image(mode="RGB")
        result = convert_to_grayscale(img)
        assert result.mode == "L"

    def test_already_grayscale(self):
        img = Image.fromarray(np.zeros((50, 200), dtype=np.uint8), mode="L")
        result = convert_to_grayscale(img)
        assert result.mode == "L"

    def test_rgba_to_grayscale(self):
        arr = np.random.randint(0, 255, (50, 200, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        result = convert_to_grayscale(img)
        assert result.mode == "L"


class TestResizeAndPad:
    def test_output_dimensions(self):
        img = Image.fromarray(np.zeros((50, 200), dtype=np.uint8), mode="L")
        result = resize_and_pad(img, IMG_HEIGHT, IMG_WIDTH)
        assert result.size == (IMG_WIDTH, IMG_HEIGHT)

    def test_narrow_image_gets_padded(self):
        img = Image.fromarray(np.zeros((50, 30), dtype=np.uint8), mode="L")
        result = resize_and_pad(img, IMG_HEIGHT, IMG_WIDTH)
        assert result.size == (IMG_WIDTH, IMG_HEIGHT)

    def test_wide_image_gets_squeezed(self):
        img = Image.fromarray(np.zeros((50, 500), dtype=np.uint8), mode="L")
        result = resize_and_pad(img, IMG_HEIGHT, IMG_WIDTH)
        assert result.size == (IMG_WIDTH, IMG_HEIGHT)

    def test_aspect_ratio_preserved_before_padding(self):
        img = Image.fromarray(
            np.full((50, 100), 128, dtype=np.uint8), mode="L"
        )
        result = resize_and_pad(img, IMG_HEIGHT, IMG_WIDTH)
        arr = np.array(result)
        assert arr[:, -1].mean() < 10


class TestNormalizeImage:
    def test_output_range(self):
        img = Image.fromarray(
            np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8),
            mode="L",
        )
        tensor = normalize_image(img)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0
        assert tensor.shape == (1, IMG_HEIGHT, IMG_WIDTH)


class TestPreprocessImage:
    def test_full_pipeline(self):
        img = make_test_image(width=200, height=50, mode="RGB")
        tensor = preprocess_image(img)
        assert tensor.shape == (1, IMG_HEIGHT, IMG_WIDTH)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0
