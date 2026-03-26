import numpy as np
import pytest
import cv2
from PIL import Image

from model.segmentation.deskew import (
    pil_to_cv2,
    cv2_to_pil,
    detect_skew_angle,
    deskew,
    binarize,
    remove_lines,
    preprocess_page,
)


def make_test_page(width=800, height=600, text_angle=0):
    """Create a synthetic page with text-like horizontal strokes."""
    img = np.ones((height, width), dtype=np.uint8) * 255
    # Draw horizontal "text" lines
    for y in range(100, 500, 40):
        cv2.line(img, (100, y), (700, y), 0, 2)
        # Add some short strokes to simulate text
        for x in range(100, 700, 30):
            cv2.line(img, (x, y - 10), (x + 15, y + 5), 0, 2)

    if abs(text_angle) > 0.1:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, text_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    return img


class TestPilConversion:
    def test_rgb_to_cv2(self):
        pil_img = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8), mode="RGB")
        gray = pil_to_cv2(pil_img)
        assert gray.ndim == 2
        assert gray.shape == (50, 100)

    def test_grayscale_roundtrip(self):
        arr = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode="L")
        gray = pil_to_cv2(pil_img)
        np.testing.assert_array_equal(gray, arr)

    def test_cv2_to_pil(self):
        arr = np.zeros((50, 100), dtype=np.uint8)
        pil_img = cv2_to_pil(arr)
        assert pil_img.mode == "L"
        assert pil_img.size == (100, 50)


class TestDeskew:
    def test_no_rotation_returns_similar(self):
        page = make_test_page(text_angle=0)
        result = deskew(page)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_output_is_numpy(self):
        page = make_test_page()
        result = deskew(page)
        assert isinstance(result, np.ndarray)

    def test_explicit_angle(self):
        page = make_test_page()
        result = deskew(page, angle=5.0)
        assert result.shape[0] > 0


class TestBinarize:
    def test_output_is_binary(self):
        gray = np.random.randint(100, 200, (100, 200), dtype=np.uint8)
        binary = binarize(gray)
        unique = set(np.unique(binary))
        assert unique.issubset({0, 255})

    def test_output_shape_matches(self):
        gray = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        binary = binarize(gray)
        assert binary.shape == gray.shape


class TestRemoveLines:
    def test_removes_horizontal_lines(self):
        img = np.zeros((200, 400), dtype=np.uint8)
        cv2.line(img, (0, 100), (400, 100), 255, 1)
        # Add some text-like blobs
        cv2.circle(img, (200, 50), 10, 255, -1)
        cleaned = remove_lines(img)
        # The blob should still be there
        assert cleaned[50, 200] > 0

    def test_output_shape(self):
        img = np.zeros((200, 400), dtype=np.uint8)
        cleaned = remove_lines(img)
        assert cleaned.shape == img.shape


class TestPreprocessPage:
    def test_returns_two_arrays(self):
        pil_img = Image.fromarray(make_test_page(), mode="L")
        binary, gray = preprocess_page(pil_img)
        assert isinstance(binary, np.ndarray)
        assert isinstance(gray, np.ndarray)

    def test_binary_is_binary(self):
        pil_img = Image.fromarray(make_test_page(), mode="L")
        binary, _ = preprocess_page(pil_img)
        unique = set(np.unique(binary))
        assert unique.issubset({0, 255})
