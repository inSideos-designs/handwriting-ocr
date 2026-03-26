import numpy as np
import pytest

from model.segmentation.words import (
    vertical_projection,
    find_word_gaps,
    extract_words,
)


def make_word_line(width=400, height=40, num_words=3, word_width=60, gap=40):
    """Create a binary image of a text line with word-like blobs."""
    img = np.zeros((height, width), dtype=np.uint8)
    x = 20
    for _ in range(num_words):
        img[5:35, x:x + word_width] = 255
        x += word_width + gap
    return img


class TestVerticalProjection:
    def test_empty_image(self):
        img = np.zeros((40, 200), dtype=np.uint8)
        proj = vertical_projection(img)
        assert len(proj) == 200
        assert np.all(proj == 0)

    def test_full_column(self):
        img = np.zeros((40, 200), dtype=np.uint8)
        img[:, 100] = 255
        proj = vertical_projection(img)
        assert proj[100] == 40
        assert proj[0] == 0

    def test_output_length(self):
        img = np.zeros((50, 300), dtype=np.uint8)
        proj = vertical_projection(img)
        assert len(proj) == 300


class TestFindWordGaps:
    def test_finds_correct_count(self):
        img = make_word_line(num_words=3)
        proj = vertical_projection(img)
        words = find_word_gaps(proj)
        assert len(words) == 3

    def test_single_word(self):
        img = make_word_line(num_words=1)
        proj = vertical_projection(img)
        words = find_word_gaps(proj)
        assert len(words) == 1

    def test_empty_projection(self):
        proj = np.zeros(200)
        words = find_word_gaps(proj)
        assert len(words) == 0

    def test_words_are_ordered(self):
        img = make_word_line(num_words=4)
        proj = vertical_projection(img)
        words = find_word_gaps(proj)
        for i in range(len(words) - 1):
            assert words[i][1] <= words[i + 1][0]


class TestExtractWords:
    def test_extracts_correct_count(self):
        img = make_word_line(num_words=3)
        words = extract_words(img)
        assert len(words) == 3

    def test_each_word_is_array(self):
        img = make_word_line(num_words=2)
        words = extract_words(img)
        for word in words:
            assert isinstance(word, np.ndarray)
            assert word.ndim == 2

    def test_empty_image(self):
        img = np.zeros((40, 200), dtype=np.uint8)
        words = extract_words(img)
        assert len(words) == 0

    def test_word_dimensions(self):
        img = make_word_line(num_words=2, word_width=80)
        words = extract_words(img, padding=3)
        for word in words:
            assert word.shape[0] > 0
            assert word.shape[1] > 0
