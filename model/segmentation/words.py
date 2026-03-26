import cv2
import numpy as np


def vertical_projection(binary):
    """Compute vertical projection profile (sum of white pixels per column)."""
    return np.sum(binary > 0, axis=0)


def find_word_gaps(projection, min_gap_width=None):
    """
    Find gaps between words in a vertical projection profile.

    Args:
        projection: 1D array of pixel counts per column
        min_gap_width: minimum gap width to consider as word separator.
                       If None, auto-detect using median gap analysis.

    Returns:
        list of (start_col, end_col) tuples for each word
    """
    threshold = np.max(projection) * 0.02 if np.max(projection) > 0 else 0
    is_ink = projection > threshold

    # Find runs of ink and gaps
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(is_ink):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(projection)))

    if len(segments) <= 1:
        return segments

    # Calculate gaps between segments
    gaps = []
    for i in range(1, len(segments)):
        gap_start = segments[i - 1][1]
        gap_end = segments[i][0]
        gap_width = gap_end - gap_start
        gaps.append((i, gap_width))

    if not gaps:
        return segments

    # Auto-detect gap threshold if not specified
    if min_gap_width is None:
        gap_widths = [g[1] for g in gaps]
        if len(gap_widths) > 1:
            median_gap = np.median(gap_widths)
            min_gap_width = max(median_gap * 0.6, 3)
        else:
            min_gap_width = gap_widths[0] * 0.5

    # Merge segments separated by small gaps (within-word gaps)
    words = [segments[0]]
    for i in range(1, len(segments)):
        gap_width = segments[i][0] - segments[i - 1][1]
        if gap_width < min_gap_width:
            # Merge with previous word
            words[-1] = (words[-1][0], segments[i][1])
        else:
            words.append(segments[i])

    return words


def extract_words(line_binary, padding=3):
    """
    Extract individual word images from a binary line image.

    Args:
        line_binary: numpy array, binary line image (text=white, bg=black)
        padding: pixels of padding around each word

    Returns:
        list of numpy arrays, each containing one word
    """
    projection = vertical_projection(line_binary)
    word_boundaries = find_word_gaps(projection)

    h, w = line_binary.shape
    word_images = []

    for col_start, col_end in word_boundaries:
        x_start = max(0, col_start - padding)
        x_end = min(w, col_end + padding)

        word_img = line_binary[:, x_start:x_end]

        # Crop vertical whitespace
        row_sums = np.sum(word_img > 0, axis=1)
        nonzero_rows = np.where(row_sums > 0)[0]

        if len(nonzero_rows) == 0:
            continue

        y_start = max(0, nonzero_rows[0] - padding)
        y_end = min(h, nonzero_rows[-1] + padding)

        cropped = word_img[y_start:y_end, :]

        if cropped.shape[0] > 3 and cropped.shape[1] > 3:
            word_images.append(cropped)

    return word_images
