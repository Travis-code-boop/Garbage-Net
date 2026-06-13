"""Shared image loading and preprocessing utilities.

Notebook block purpose:
    Image preprocessing. These functions convert a selected image path into
    the standard image versions used by later feature extraction code:
    RGB, HSV, grayscale, 128x128 grayscale, and a simple binary foreground mask.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ImageVersions:
    """Container for standardized image arrays used by feature extraction."""

    rgb: np.ndarray
    hsv: np.ndarray
    gray: np.ndarray
    gray_128: np.ndarray
    binary_mask: np.ndarray


def read_metadata_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV metadata file into a list of dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_one_train_sample_per_class(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select one deterministic train sample per class for preprocessing checks."""
    selected: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda item: (item["label"], item["image_id"])):
        if row.get("split") != "train":
            continue
        selected.setdefault(row["label"], row)
    return [selected[label] for label in sorted(selected)]


def load_rgb_image(relative_path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load an image as a 256x256 RGB uint8 array."""
    image_path = PROJECT_ROOT / relative_path
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array to HSV using PIL's 8-bit HSV representation."""
    image = Image.fromarray(rgb, mode="RGB")
    return np.asarray(image.convert("HSV"), dtype=np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using standard luminance weights."""
    rgb_float = rgb.astype(np.float32)
    gray = 0.299 * rgb_float[:, :, 0] + 0.587 * rgb_float[:, :, 1] + 0.114 * rgb_float[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def resize_gray(gray: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize a grayscale array to the requested size."""
    image = Image.fromarray(gray, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu's threshold for an 8-bit grayscale image."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_total = mu[-1]

    denominator = omega * (1.0 - omega)
    valid = denominator > 0
    sigma_between = np.zeros(256, dtype=np.float64)
    sigma_between[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denominator[valid]

    return int(np.argmax(sigma_between))


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Create a simple binary foreground mask with a light/dark foreground heuristic."""
    threshold = otsu_threshold(gray)
    masks = [gray > threshold, gray <= threshold]

    def score(mask: np.ndarray) -> float:
        ratio = float(mask.mean())
        if ratio < 0.02 or ratio > 0.90:
            return 10.0 + abs(ratio - 0.35)
        return abs(ratio - 0.35)

    chosen = min(masks, key=score)
    return chosen.astype(np.uint8)


def prepare_image_versions(relative_path: str | Path) -> ImageVersions:
    """Load an image and create all standard preprocessing versions."""
    rgb = load_rgb_image(relative_path)
    hsv = rgb_to_hsv(rgb)
    gray = rgb_to_gray(rgb)
    gray_128 = resize_gray(gray, size=(128, 128))
    mask = foreground_mask(gray)
    return ImageVersions(rgb=rgb, hsv=hsv, gray=gray, gray_128=gray_128, binary_mask=mask)


def summarize_versions(row: dict[str, str], versions: ImageVersions) -> dict[str, object]:
    """Return a compact row describing generated image versions."""
    return {
        "image_id": row["image_id"],
        "label": row["label"],
        "path": row["path"],
        "rgb_shape": "x".join(map(str, versions.rgb.shape)),
        "hsv_shape": "x".join(map(str, versions.hsv.shape)),
        "gray_shape": "x".join(map(str, versions.gray.shape)),
        "gray_128_shape": "x".join(map(str, versions.gray_128.shape)),
        "mask_shape": "x".join(map(str, versions.binary_mask.shape)),
        "mask_foreground_ratio": round(float(versions.binary_mask.mean()), 4),
        "rgb_dtype": str(versions.rgb.dtype),
        "gray_dtype": str(versions.gray.dtype),
    }


def array_to_uint8_image(array: np.ndarray) -> Image.Image:
    """Convert a grayscale or RGB array to a PIL image for preview grids."""
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def mask_to_preview(mask: np.ndarray) -> Image.Image:
    """Convert a 0/1 binary mask to a visible black-white preview image."""
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")
