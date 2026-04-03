"""
Image preprocessing for CT scan classification.
Handles standard JPG/PNG screenshots from Google Images.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2


TARGET_SIZE = (224, 224)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess a CT scan image for feature extraction and classification.
    Returns a normalized float32 numpy array (224, 224, 3).
    """
    # Convert to RGB
    img = img.convert("RGB")

    # Resize with high-quality resampling
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

    # Convert to numpy
    arr = np.array(img_resized, dtype=np.float32)

    # Normalize to [0, 1]
    arr = arr / 255.0

    # Apply CT-window-like normalization
    # Real CT: window center ~40 HU, width ~350 HU for soft tissue
    # On screenshots, we approximate by emphasizing mid-gray range
    arr = _apply_soft_tissue_window(arr)

    return arr


def validate_image(img: Image.Image) -> dict:
    """
    Basic sanity check — is this plausibly a CT scan?
    Returns {"valid": bool, "reason": str}
    """
    arr = np.array(img.convert("RGB"))

    # CT scans are typically dark (lots of black background)
    mean_brightness = arr.mean() / 255.0
    gray_fraction = _compute_gray_fraction(arr)

    checks = {
        "size_ok": min(img.size) >= 100,
        "not_too_bright": mean_brightness < 0.7,
        "grayscale_dominant": gray_fraction > 0.6,
    }

    if not checks["size_ok"]:
        return {"valid": False, "reason": "Image too small (min 100px)"}
    if not checks["not_too_bright"]:
        return {"valid": False, "reason": "Image too bright — may not be a CT scan"}

    return {"valid": True, "reason": "ok"}


def extract_visual_features(img: Image.Image) -> dict:
    """
    Extract human-interpretable visual features from the image.
    These approximate what a radiologist would assess.
    Used for the feature analysis panel in the UI.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)  # grayscale
    arr_norm = arr / 255.0

    h, w = arr.shape

    # Region of interest — central third (where pancreas typically appears)
    roi = arr_norm[h//3 : 2*h//3, w//3 : 2*w//3]

    # 1. Mean intensity in central ROI (high = hyperdense, low = hypodense)
    mean_intensity = float(roi.mean())

    # 2. Texture complexity (std dev as a proxy for heterogeneity)
    texture_std = float(roi.std())

    # 3. Dark region fraction — hypodense areas (possible necrosis, cysts)
    dark_fraction = float((roi < 0.25).mean())

    # 4. Bright spot fraction — calcifications, bones
    bright_fraction = float((roi > 0.75).mean())

    # 5. Edge density — structural complexity (irregular ducts, masses)
    edges = _compute_edge_density(arr)
    edge_density = float(edges)

    # 6. Symmetry — left-right symmetry of the image
    left  = arr_norm[:, :w//2]
    right = np.fliplr(arr_norm[:, w//2:])
    min_w = min(left.shape[1], right.shape[1])
    symmetry = 1.0 - float(np.abs(left[:, :min_w] - right[:, :min_w]).mean()) * 5
    symmetry = max(0.0, min(1.0, symmetry))

    return {
        "mean_intensity": mean_intensity,
        "texture_heterogeneity": texture_std,
        "dark_region_fraction": dark_fraction,
        "bright_spot_fraction": bright_fraction,
        "edge_density": edge_density,
        "bilateral_symmetry": symmetry,
    }


def _apply_soft_tissue_window(arr: np.ndarray) -> np.ndarray:
    """Enhance mid-range values to simulate CT soft-tissue window."""
    # Clip to soft-tissue range (roughly 0.1–0.7 on normalized scale)
    clipped = np.clip(arr, 0.1, 0.7)
    # Rescale to full [0, 1]
    rescaled = (clipped - 0.1) / 0.6
    return rescaled.astype(np.float32)


def _compute_gray_fraction(arr: np.ndarray) -> float:
    """Fraction of pixels where R≈G≈B (grayscale)."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    diff_rg = np.abs(r.astype(int) - g.astype(int))
    diff_rb = np.abs(r.astype(int) - b.astype(int))
    gray_mask = (diff_rg < 20) & (diff_rb < 20)
    return float(gray_mask.mean())


def _compute_edge_density(arr: np.ndarray) -> float:
    """Sobel edge density as a proxy for structural complexity."""
    try:
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        sobelx = cv2.Sobel(arr_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(arr_uint8, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return float(np.clip(magnitude.mean() / 50.0, 0, 1))
    except Exception:
        return 0.5
