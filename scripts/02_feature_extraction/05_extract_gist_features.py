"""Extract F5 simplified GIST features with the planned 64-dimensional design.

Notebook block purpose:
    F5 GIST feature extraction. For each image, this code applies four oriented
    Gabor filters to the 128x128 grayscale image, divides each response map into
    a 4x4 grid, and stores the mean absolute response per grid cell. The result
    is a compact 64-dimensional global structure descriptor.

Feature dimensions:
    Grid: 4 x 4 = 16 cells
    Orientations: 4
    Total: 16 x 4 = 64
"""

from __future__ import annotations

import json
import time

import numpy as np

from feature_io import (
    FEATURE_DIR,
    PROJECT_ROOT,
    label_mapping,
    read_metadata_with_split,
    rows_by_split,
    save_feature_cache,
    summarize_feature_cache,
)
from image_preprocessing import prepare_image_versions


OUTPUT_DIR = FEATURE_DIR / "F5_gist"
FEATURE_CACHE_PATH = OUTPUT_DIR / "F5_gist_64.npz"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "F5_gist_64_summary.json"

IMAGE_SIZE = 128
GRID_SIZE = 4
ORIENTATIONS_DEG = (0.0, 45.0, 90.0, 135.0)
KERNEL_SIZE = 15
SIGMA = 4.0
LAMBDA = 10.0
GAMMA = 0.5
PSI = 0.0


def gabor_kernel(
    theta_degrees: float,
    kernel_size: int = KERNEL_SIZE,
    sigma: float = SIGMA,
    wavelength: float = LAMBDA,
    gamma: float = GAMMA,
    psi: float = PSI,
) -> np.ndarray:
    """Create a real-valued zero-mean Gabor kernel."""
    radius = kernel_size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    theta = np.deg2rad(theta_degrees)
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    envelope = np.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2.0 * sigma**2))
    carrier = np.cos(2.0 * np.pi * x_theta / wavelength + psi)
    kernel = envelope * carrier
    kernel -= kernel.mean()
    norm = np.sqrt(np.sum(kernel**2))
    if norm > 0:
        kernel /= norm
    return kernel.astype(np.float32)


def convolve_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D image with a small kernel using reflect padding."""
    radius_y = kernel.shape[0] // 2
    radius_x = kernel.shape[1] // 2
    padded = np.pad(image, ((radius_y, radius_y), (radius_x, radius_x)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    response = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    return response.astype(np.float32)


def grid_mean_abs_response(response: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Pool a response map into mean absolute responses over a grid."""
    cell_h = response.shape[0] // grid_size
    cell_w = response.shape[1] // grid_size
    features = []

    abs_response = np.abs(response)
    for grid_y in range(grid_size):
        y0 = grid_y * cell_h
        y1 = y0 + cell_h
        for grid_x in range(grid_size):
            x0 = grid_x * cell_w
            x1 = x0 + cell_w
            features.append(float(abs_response[y0:y1, x0:x1].mean()))

    return np.asarray(features, dtype=np.float32)


def build_gabor_kernels() -> list[np.ndarray]:
    """Build the four default oriented Gabor kernels."""
    return [gabor_kernel(theta) for theta in ORIENTATIONS_DEG]


def extract_gist_from_gray(gray_128: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Extract the planned 64-dimensional simplified GIST vector."""
    image = gray_128.astype(np.float32) / 255.0
    image = image - float(image.mean())

    features = []
    for kernel in kernels:
        response = convolve_reflect(image, kernel)
        features.append(grid_mean_abs_response(response))

    return np.concatenate(features).astype(np.float32)


def extract_gist_features_for_image(relative_path: str, kernels: list[np.ndarray]) -> np.ndarray:
    """Extract simplified GIST features for one image."""
    versions = prepare_image_versions(relative_path)
    return extract_gist_from_gray(versions.gray_128, kernels)


def build_feature_names() -> tuple[list[str], list[str]]:
    """Build stable feature names and group labels for GIST features."""
    feature_names: list[str] = []
    feature_groups: list[str] = []

    for theta in ORIENTATIONS_DEG:
        theta_name = str(int(theta)).zfill(3)
        for grid_y in range(GRID_SIZE):
            for grid_x in range(GRID_SIZE):
                feature_names.append(f"gist_theta{theta_name}_grid_y{grid_y}_x{grid_x}")
                feature_groups.append("gist")

    return feature_names, feature_groups


def extract_split_features(split_rows: list[dict[str, str]], kernels: list[np.ndarray]) -> np.ndarray:
    """Extract GIST features for all rows in one split."""
    matrix = np.empty((len(split_rows), 64), dtype=np.float32)
    for index, row in enumerate(split_rows):
        matrix[index] = extract_gist_features_for_image(row["path"], kernels)
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(split_rows)} images for split")
    return matrix


def main() -> None:
    start_time = time.perf_counter()
    rows = read_metadata_with_split()
    split_rows = rows_by_split(rows)
    mapping = label_mapping(rows)
    feature_names, feature_groups = build_feature_names()
    kernels = build_gabor_kernels()

    split_features = {}
    for split_name, current_rows in split_rows.items():
        print(f"extracting {split_name}: {len(current_rows)} images")
        split_features[split_name] = extract_split_features(current_rows, kernels)

    params = {
        "feature_id": "F5_gist",
        "feature_dim": 64,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "grid_size": [GRID_SIZE, GRID_SIZE],
        "orientations_degrees": ORIENTATIONS_DEG,
        "kernel_size": KERNEL_SIZE,
        "sigma": SIGMA,
        "wavelength": LAMBDA,
        "gamma": GAMMA,
        "psi": PSI,
        "kernel_normalization": "zero-mean unit-l2",
        "pooling": "mean absolute response per grid cell",
        "source_metadata": str(
            (PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_with_split.csv").relative_to(
                PROJECT_ROOT
            )
        ),
    }

    save_feature_cache(
        FEATURE_CACHE_PATH,
        split_rows,
        split_features,
        feature_names,
        feature_groups,
        params,
        mapping,
    )

    summary = summarize_feature_cache(split_features, feature_names, feature_groups)
    summary["output_path"] = str(FEATURE_CACHE_PATH.relative_to(PROJECT_ROOT))
    summary["elapsed_seconds"] = round(time.perf_counter() - start_time, 3)
    summary["params"] = params

    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
