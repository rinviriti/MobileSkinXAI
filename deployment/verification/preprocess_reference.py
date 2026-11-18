"""Authoritative CPU image contract; no model or dataset changes.

Input may be a frozen prepared224 JPEG (read directly) or a new raw image.
Raw compatibility requires the historical JPEG quality95 / 4:4:4 round trip.
Do not encode an already prepared image a second time.
"""
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)

def prepared_rgb(path):
    with Image.open(path) as im:
        im.load()
        assert im.mode == 'RGB' and im.size == (224, 224)
        return np.array(im, dtype=np.uint8)

def raw_to_prepared_rgb(path):
    with Image.open(path) as im:
        im.load()
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        im = ImageOps.pad(rgb, (224, 224), method=Image.Resampling.LANCZOS,
                          color=(114, 114, 114), centering=(.5, .5))
        output = BytesIO()
        im.save(output, format='JPEG', quality=95, subsampling=0)
    output.seek(0)
    with Image.open(output) as im:
        return np.array(im, dtype=np.uint8)

def normalize_rgb(rgb):
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
    # Exactly ToTensor(float32) then Normalize; export-facing layout is NHWC.
    return np.ascontiguousarray(((rgb.astype(np.float32) / np.float32(255)) - MEAN) / STD)[None]

def preprocess(path, already_prepared=False):
    rgb = prepared_rgb(path) if already_prepared else raw_to_prepared_rgb(path)
    return normalize_rgb(rgb)
