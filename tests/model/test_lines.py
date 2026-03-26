import numpy as np
import pytest
import cv2

from model.segmentation.lines import (
    horizontal_projection,
    find_line_boundaries,
    extract_lines,
)


def make_lined_image(width=400, height=300, num_lines=3, line_height=20, gap=40):
    """Create a binary image with horizontal text-like bands."""
    img = np.zeros((height, width), dtype=np.uint8)
    y = 30
    for _ in range(num_lines):
        # Draw a band of white pixels simulating a text line
        img[y:y + line_height, 50:350] = 255
        y += line_height + gap
    return img


class TestHorizontalProjection:
    def test_empty_image(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        proj = horizontal_projection(img)
        assert len(proj) == 100
        assert np.all(proj == 0)

    def test_full_row(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        img[50, :] = 255
        proj = horizontal_projection(img)
        assert proj[50] == 200
        assert proj[0] == 0

    def test_output_length(self):
        img = np.zeros((150, 300), dtype=np.uint8)
        proj = horizontal_projection(img)
        assert len(proj) == 150


class TestFindLineBoundaries:
    def test_finds_correct_count(self):
        img = make_lined_image(num_lines=3)
        proj = horizontal_projection(img)
        boundaries = find_line_boundaries(proj)
        assert len(boundaries) == 3

    def test_finds_single_line(self):
        img = make_lined_image(num_lines=1)
        proj = horizontal_projection(img)
        boundaries = find_line_boundaries(proj)
        assert len(boundaries) == 1

    def test_empty_image_no_lines(self):
        proj = np.zeros(100)
        boundaries = find_line_boundaries(proj)
        assert len(boundaries) == 0

    def test_boundaries_are_ordered(self):
        img = make_lined_image(num_lines=4)
        proj = horizontal_projection(img)
        boundaries = find_line_boundaries(proj)
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] <= boundaries[i + 1][0]


class TestExtractLines:
    def test_extracts_correct_count(self):
        img = make_lined_image(num_lines=3)
        lines = extract_lines(img)
        assert len(lines) == 3

    def test_each_line_is_array(self):
        img = make_lined_image(num_lines=2)
        lines = extract_lines(img)
        for line in lines:
            assert isinstance(line, np.ndarray)
            assert line.ndim == 2

    def test_empty_image(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        lines = extract_lines(img)
        assert len(lines) == 0

    def test_line_dimensions_reasonable(self):
        img = make_lined_image(num_lines=2, line_height=30)
        lines = extract_lines(img, padding=5)
        for line in lines:
            assert line.shape[0] >= 30  # at least line_height
            assert line.shape[1] > 0
