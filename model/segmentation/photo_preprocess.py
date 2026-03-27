"""
Preprocessing for phone camera photos of handwriting.

Handles issues that don't exist in scanner/IAM-style images:
- Auto-rotation detection (0/90/180/270)
- Shadow removal
- Contrast enhancement for low-contrast ink (pencil, light pen)
"""

import cv2
import numpy as np
from PIL import Image, ImageOps


def remove_shadows(gray):
    """
    Remove shadows from a grayscale image using Gaussian background estimation.

    Uses a large Gaussian blur to estimate illumination, then normalizes.
    """
    # Estimate illumination with large Gaussian blur
    blur_size = min(gray.shape[0], gray.shape[1]) // 4
    blur_size = max(blur_size, 3)
    blur_size = min(blur_size, 255)
    if blur_size % 2 == 0:
        blur_size += 1

    bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Normalize: divide original by background, scale to 0-255
    # This removes uneven illumination while preserving text
    result = cv2.divide(gray, bg, scale=255)

    return result.astype(np.uint8)


def enhance_contrast(gray):
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def auto_rotate_with_model(img, predictor):
    """
    Auto-rotate by trying all 4 orientations with the OCR model
    and picking the one with highest confidence.

    More robust than heuristics for sparse text or unusual layouts.

    Args:
        img: PIL Image
        predictor: object with predict_with_confidence(img) method

    Returns:
        best rotated PIL Image, angle applied, (text, confidence)
    """
    # Try with and without EXIF transpose — phone photos may have
    # incorrect EXIF data that makes things worse
    candidates = [img]
    try:
        exif_img = ImageOps.exif_transpose(img)
        if exif_img.size != img.size:
            candidates.append(exif_img)
    except Exception:
        pass

    best_angle = 0
    best_conf = -1
    best_text = ""
    best_img = img

    for base_img in candidates:
        for angle in [0, 90, 180, 270]:
            rotated = base_img.rotate(angle, expand=True) if angle != 0 else base_img
            text, conf = predictor.predict_with_confidence(rotated)
            if conf > best_conf and text.strip():
                best_conf = conf
                best_angle = angle
                best_text = text
                best_img = rotated

    return best_img, best_angle, (best_text, best_conf)


def preprocess_photo(img):
    """
    Shadow removal and contrast enhancement for a phone camera photo.

    Does NOT handle rotation — use auto_rotate_with_model() for that,
    since model-based rotation is far more reliable than heuristics.

    Args:
        img: PIL Image (already correctly oriented)

    Returns:
        preprocessed PIL Image (RGB)
    """
    if img.mode != "L":
        gray = np.array(img.convert("L"))
    else:
        gray = np.array(img)

    gray = remove_shadows(gray)
    gray = enhance_contrast(gray)

    return Image.fromarray(gray, mode="L").convert("RGB")
