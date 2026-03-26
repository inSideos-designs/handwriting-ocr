import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(img):
    """Convert PIL Image to OpenCV format (grayscale)."""
    if img.mode == "RGB" or img.mode == "RGBA":
        cv_img = np.array(img.convert("RGB"))
        return cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    return np.array(img)


def cv2_to_pil(img):
    """Convert OpenCV grayscale image to PIL Image."""
    return Image.fromarray(img, mode="L")


def detect_skew_angle(gray):
    """
    Detect the skew angle of text in an image using Hough line transform.
    Returns angle in degrees.
    """
    # Threshold and invert
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Filter outliers — keep angles within 45 degrees of median
    median_angle = np.median(angles)
    filtered = [a for a in angles if abs(a - median_angle) < 45]

    if not filtered:
        return 0.0

    return float(np.median(filtered))


def deskew(gray, angle=None):
    """
    Deskew a grayscale image. If angle is None, auto-detect.
    Returns the deskewed image.
    """
    if angle is None:
        angle = detect_skew_angle(gray)

    if abs(angle) < 0.5:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box size
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix for translation
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(gray, rotation_matrix, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)
    return rotated


def binarize(gray):
    """Apply adaptive binarization to a grayscale image."""
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (works well on lined paper, uneven lighting)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=10
    )
    return binary


def remove_lines(binary):
    """
    Remove horizontal and vertical ruled lines from a binary image.
    Preserves text strokes.
    """
    h, w = binary.shape

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 1), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 1)))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Subtract lines from original
    lines_mask = cv2.add(horizontal_lines, vertical_lines)
    cleaned = cv2.subtract(binary, lines_mask)

    # Small dilation to reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    return cleaned


def preprocess_page(img):
    """
    Full preprocessing pipeline for a page image.

    Args:
        img: PIL Image

    Returns:
        binary: numpy array (binary, text=white, bg=black)
        gray_deskewed: numpy array (grayscale, deskewed)
    """
    gray = pil_to_cv2(img)
    gray_deskewed = deskew(gray)
    binary = binarize(gray_deskewed)
    cleaned = remove_lines(binary)
    return cleaned, gray_deskewed
