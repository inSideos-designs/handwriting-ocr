import numpy as np
import torch
from PIL import Image

IMG_HEIGHT = 32
IMG_WIDTH = 128


def convert_to_grayscale(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        return img.convert("L")
    return img


def resize_and_pad(
    img: Image.Image, target_height: int = IMG_HEIGHT, target_width: int = IMG_WIDTH
) -> Image.Image:
    w, h = img.size
    ratio = target_height / h
    new_w = min(int(w * ratio), target_width)
    img = img.resize((new_w, target_height), Image.BILINEAR)

    if new_w < target_width:
        padded = Image.new("L", (target_width, target_height), 0)
        padded.paste(img, (0, 0))
        return padded
    return img


def normalize_image(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def preprocess_image(img: Image.Image, target_width: int = IMG_WIDTH) -> torch.Tensor:
    img = convert_to_grayscale(img)
    img = resize_and_pad(img, target_width=target_width)
    return normalize_image(img)
