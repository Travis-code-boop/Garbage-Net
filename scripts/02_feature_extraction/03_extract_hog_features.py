"""Extract F3 HOG edge features with the planned 1764-dimensional design.

Notebook block purpose:
    F3 HOG feature extraction. For each image, this code uses the 128x128
    grayscale version to compute unsigned gradient orientation histograms,
    normalizes them over 2x2-cell blocks, and saves the resulting fixed-length
    feature matrices for later single-feature model screening.

Feature dimensions:
    Image size: 128 x 128
    Cell size: 16 x 16 -> 8 x 8 cells
    Block size: 2 x 2 cells -> 7 x 7 blocks
    Orientations: 9
    Total: 7 x 7 x 2 x 2 x 9 = 1764
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


OUTPUT_DIR = FEATURE_DIR / "F3_hog"
FEATURE_CACHE_PATH = OUTPUT_DIR / "F3_hog_1764.npz"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "F3_hog_1764_summary.json"

IMAGE_SIZE = 128
ORIENTATIONS = 9
PIXELS_PER_CELL = 16
CELLS_PER_BLOCK = 2
BLOCK_NORM_EPS = 1e-6
L2_HYS_CLIP = 0.2


def gradients(gray_128: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient magnitude and unsigned orientation in degrees."""
    image = gray_128.astype(np.float32) / 255.0
    gy, gx = np.gradient(image)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180.0).astype(np.float32)
    return magnitude, orientation


def cell_histograms(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    cell_size: int = PIXELS_PER_CELL,
    orientations: int = ORIENTATIONS,
) -> np.ndarray:
    """Accumulate gradient magnitudes into orientation histograms per cell."""
    num_cells_y = magnitude.shape[0] // cell_size
    num_cells_x = magnitude.shape[1] // cell_size
    histograms = np.zeros((num_cells_y, num_cells_x, orientations), dtype=np.float32)
    bin_width = 180.0 / orientations

    for cell_y in range(num_cells_y):
        y0 = cell_y * cell_size
        y1 = y0 + cell_size
        for cell_x in range(num_cells_x):
            x0 = cell_x * cell_size
            x1 = x0 + cell_size
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_ori = orientation[y0:y1, x0:x1].ravel()
            bins = np.floor(cell_ori / bin_width).astype(np.int16)
            bins = np.clip(bins, 0, orientations - 1)
            histograms[cell_y, cell_x] = np.bincount(
                bins,
                weights=cell_mag,
                minlength=orientations,
            ).astype(np.float32)

    return histograms


def normalize_block(block: np.ndarray) -> np.ndarray:
    """Apply L2-Hys normalization to one HOG block."""
    vector = block.ravel().astype(np.float32)
    norm = np.sqrt(np.sum(vector**2) + BLOCK_NORM_EPS**2)
    vector = vector / norm
    vector = np.minimum(vector, L2_HYS_CLIP)
    norm = np.sqrt(np.sum(vector**2) + BLOCK_NORM_EPS**2)
    return vector / norm


def extract_hog_from_gray(gray_128: np.ndarray) -> np.ndarray:
    """Extract the planned 1764-dimensional HOG vector from a 128x128 image."""
    magnitude, orientation = gradients(gray_128)
    histograms = cell_histograms(magnitude, orientation)
    num_cells_y, num_cells_x, _ = histograms.shape
    num_blocks_y = num_cells_y - CELLS_PER_BLOCK + 1
    num_blocks_x = num_cells_x - CELLS_PER_BLOCK + 1

    blocks = []
    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            block = histograms[
                block_y : block_y + CELLS_PER_BLOCK,
                block_x : block_x + CELLS_PER_BLOCK,
                :,
            ]
            blocks.append(normalize_block(block))

    return np.concatenate(blocks).astype(np.float32)


def build_feature_names() -> tuple[list[str], list[str]]:
    """Build stable feature names and group labels for HOG features."""
    feature_names: list[str] = []
    feature_groups: list[str] = []

    num_cells = IMAGE_SIZE // PIXELS_PER_CELL
    num_blocks = num_cells - CELLS_PER_BLOCK + 1
    for block_y in range(num_blocks):
        for block_x in range(num_blocks):
            for cell_y in range(CELLS_PER_BLOCK):
                for cell_x in range(CELLS_PER_BLOCK):
                    for orientation_bin in range(ORIENTATIONS):
                        feature_names.append(
                            "hog"
                            f"_block_y{block_y:02d}_x{block_x:02d}"
                            f"_cell_y{cell_y}_x{cell_x}"
                            f"_bin{orientation_bin:02d}"
                        )
                        feature_groups.append("hog")

    return feature_names, feature_groups


def extract_hog_features_for_image(relative_path: str) -> np.ndarray:
    """Extract the planned 1764-dimensional HOG vector for one image."""
    versions = prepare_image_versions(relative_path)
    return extract_hog_from_gray(versions.gray_128)


def extract_split_features(split_rows: list[dict[str, str]]) -> np.ndarray:
    """Extract HOG features for all rows in one split."""
    matrix = np.empty((len(split_rows), 1764), dtype=np.float32)
    for index, row in enumerate(split_rows):
        matrix[index] = extract_hog_features_for_image(row["path"])
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(split_rows)} images for split")
    return matrix


def main() -> None:
    start_time = time.perf_counter()
    rows = read_metadata_with_split()
    split_rows = rows_by_split(rows)
    mapping = label_mapping(rows)
    feature_names, feature_groups = build_feature_names()

    split_features = {}
    for split_name, current_rows in split_rows.items():
        print(f"extracting {split_name}: {len(current_rows)} images")
        split_features[split_name] = extract_split_features(current_rows)

    params = {
        "feature_id": "F3_hog",
        "feature_dim": 1764,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "orientations": ORIENTATIONS,
        "orientation_range": "unsigned 0-180 degrees",
        "pixels_per_cell": [PIXELS_PER_CELL, PIXELS_PER_CELL],
        "cells_per_block": [CELLS_PER_BLOCK, CELLS_PER_BLOCK],
        "num_cells": [IMAGE_SIZE // PIXELS_PER_CELL, IMAGE_SIZE // PIXELS_PER_CELL],
        "num_blocks": [
            IMAGE_SIZE // PIXELS_PER_CELL - CELLS_PER_BLOCK + 1,
            IMAGE_SIZE // PIXELS_PER_CELL - CELLS_PER_BLOCK + 1,
        ],
        "block_normalization": "L2-Hys",
        "l2_hys_clip": L2_HYS_CLIP,
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
