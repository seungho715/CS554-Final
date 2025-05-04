# image/utils.py
from PIL import Image
from io import BytesIO
import requests
import os

def img_from_disk(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def img_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

def load_image(src: str) -> Image.Image:
    if os.path.exists(src):
        return img_from_disk(src)
    if src.startswith(("http://", "https://")):
        return img_from_url(src)
    raise ValueError(f"Cannot load image from {src}")
