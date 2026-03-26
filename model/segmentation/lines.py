import cv2
import numpy as np


def horizontal_projection(binary):
    """Compute horizontal projection profile (sum of white pixels per row)."""
    return np.sum(binary > 0, axis=1)


def find_line_boundaries(projection, min_gap=5, min_line_height=10):
    """
    Find start/end row indices for each text line from projection profile.

    Args:
        projection: 1D array of pixel counts per row
        min_gap: minimum rows of zero-projection to count as a gap between lines
        min_line_height: minimum height in pixels to count as a valid line

    Returns:
        list of (start_row, end_row) tuples
    """
    threshold = np.max(projection) * 0.02 if np.max(projection) > 0 else 0
    is_text = projection > threshold

    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(is_text):
        if val and not in_line:
            start = i
            in_line = True
        elif not val and in_line:
            if i - start >= min_line_height:
                lines.append((start, i))
            in_line = False

    # Handle line that extends to bottom of image
    if in_line and len(projection) - start >= min_line_height:
        lines.append((start, len(projection)))

    # Merge lines that are very close (within min_gap)
    if len(lines) <= 1:
        return lines

    merged = [lines[0]]
    for start, end in lines[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def extract_lines(binary, padding=5):
    """
    Extract individual line images from a binarized page.

    Args:
        binary: numpy array, binary image (text=white, bg=black)
        padding: pixels of padding to add above/below each line

    Returns:
        list of numpy arrays, each containing one text line
    """
    projection = horizontal_projection(binary)
    boundaries = find_line_boundaries(projection)

    h, w = binary.shape
    line_images = []

    for start, end in boundaries:
        # Add padding
        y_start = max(0, start - padding)
        y_end = min(h, end + padding)

        line_img = binary[y_start:y_end, :]

        # Crop horizontal whitespace
        col_sums = np.sum(line_img > 0, axis=0)
        nonzero_cols = np.where(col_sums > 0)[0]

        if len(nonzero_cols) == 0:
            continue

        x_start = max(0, nonzero_cols[0] - padding)
        x_end = min(w, nonzero_cols[-1] + padding)

        cropped = line_img[:, x_start:x_end]

        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            line_images.append(cropped)

    return line_images
