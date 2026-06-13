"""Extract F2 texture features with the planned 30-dimensional design.

Notebook block purpose:
    F2 texture feature extraction. For each image, this code uses the grayscale
    image to compute GLCM texture statistics and uniform LBP histograms. The
    resulting 30-dimensional vectors are saved by fixed train/validation/test
    split for later single-feature model screening.

Feature dimensions:
    GLCM: 4 directions x 5 statistics = 20
    LBP: uniform LBP with P=8, R=1 -> P + 2 = 10 bins
    Total: 30
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


OUTPUT_DIR = FEATURE_DIR / "F2_texture"
FEATURE_CACHE_PATH = OUTPUT_DIR / "F2_texture_30.npz"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "F2_texture_30_summary.json"

GRAY_LEVELS = 32
GLCM_DISTANCE = 1
GLCM_DIRECTIONS = {
    "0deg": (0, 1),
    "45deg": (-1, 1),
    "90deg": (-1, 0),
    "135deg": (-1, -1),
}
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2


def quantize_gray(gray: np.ndarray, levels: int = GRAY_LEVELS) -> np.ndarray:
    """Quantize an 8-bit grayscale image into integer levels [0, levels-1]."""
    quantized = (gray.astype(np.uint16) * levels) // 256
    return np.minimum(quantized, levels - 1).astype(np.uint8)


def shifted_pairs(image: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned source and neighbor pixels for a given offset."""
    if dy >= 0:
        src_y = slice(0, image.shape[0] - dy)
        dst_y = slice(dy, image.shape[0])
    else:
        src_y = slice(-dy, image.shape[0])
        dst_y = slice(0, image.shape[0] + dy)

    if dx >= 0:
        src_x = slice(0, image.shape[1] - dx)
        dst_x = slice(dx, image.shape[1])
    else:
        src_x = slice(-dx, image.shape[1])
        dst_x = slice(0, image.shape[1] + dx)

    return image[src_y, src_x].ravel(), image[dst_y, dst_x].ravel()


def glcm_matrix(quantized: np.ndarray, dy: int, dx: int, levels: int = GRAY_LEVELS) -> np.ndarray:
    """Build a normalized symmetric gray-level co-occurrence matrix."""
    source, neighbor = shifted_pairs(quantized, dy, dx)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (source, neighbor), 1.0)
    np.add.at(matrix, (neighbor, source), 1.0)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def glcm_properties(matrix: np.ndarray) -> list[float]:
    """Compute the five planned Haralick-style GLCM statistics."""
    levels = matrix.shape[0]
    i, j = np.indices((levels, levels))
    diff = i - j
    abs_diff = np.abs(diff)

    contrast = float(np.sum(matrix * (diff**2)))
    dissimilarity = float(np.sum(matrix * abs_diff))
    homogeneity = float(np.sum(matrix / (1.0 + abs_diff)))
    energy = float(np.sqrt(np.sum(matrix**2)))

    row_prob = matrix.sum(axis=1)
    col_prob = matrix.sum(axis=0)
    row_mean = float(np.sum(np.arange(levels) * row_prob))
    col_mean = float(np.sum(np.arange(levels) * col_prob))
    row_std = float(np.sqrt(np.sum(((np.arange(levels) - row_mean) ** 2) * row_prob)))
    col_std = float(np.sqrt(np.sum(((np.arange(levels) - col_mean) ** 2) * col_prob)))

    if row_std <= 1e-12 or col_std <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.sum(matrix * (i - row_mean) * (j - col_mean)) / (row_std * col_std))

    return [contrast, dissimilarity, homogeneity, energy, correlation]


def extract_glcm_features(gray: np.ndarray) -> list[float]:
    """Extract 20-dimensional GLCM features from one grayscale image."""
    quantized = quantize_gray(gray)
    features: list[float] = []
    for dy, dx in GLCM_DIRECTIONS.values():
        matrix = glcm_matrix(quantized, dy * GLCM_DISTANCE, dx * GLCM_DISTANCE)
        features.extend(glcm_properties(matrix))
    return features


def uniform_lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """Compute a normalized 10-bin uniform LBP histogram with P=8, R=1."""
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]

    bits = np.stack([(neighbor >= center).astype(np.uint8) for neighbor in neighbors], axis=0)
    transitions = np.sum(bits != np.roll(bits, shift=-1, axis=0), axis=0)
    ones = bits.sum(axis=0)
    bins = np.where(transitions <= 2, ones, LBP_POINTS + 1).astype(np.int16)
    hist = np.bincount(bins.ravel(), minlength=LBP_BINS).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def build_feature_names() -> tuple[list[str], list[str]]:
    """Build stable feature names and group labels for the 30 texture features."""
    feature_names: list[str] = []
    feature_groups: list[str] = []

    for direction in GLCM_DIRECTIONS:
        for prop in GLCM_PROPS:
            feature_names.append(f"texture_glcm_{prop}_{direction}")
            feature_groups.append("texture_glcm")

    for bin_index in range(LBP_BINS):
        feature_names.append(f"texture_lbp_uniform_bin_{bin_index:02d}")
        feature_groups.append("texture_lbp")

    return feature_names, feature_groups


def extract_texture_features_for_image(relative_path: str) -> np.ndarray:
    """Extract the planned 30-dimensional texture feature vector for one image."""
    versions = prepare_image_versions(relative_path)
    glcm_features = np.asarray(extract_glcm_features(versions.gray), dtype=np.float64)
    lbp_features = uniform_lbp_histogram(versions.gray)
    features = np.concatenate([glcm_features, lbp_features])
    return features.astype(np.float32)


def extract_split_features(split_rows: list[dict[str, str]]) -> np.ndarray:
    """Extract texture features for all rows in one split."""
    matrix = np.empty((len(split_rows), 30), dtype=np.float32)
    for index, row in enumerate(split_rows):
        matrix[index] = extract_texture_features_for_image(row["path"])
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
        "feature_id": "F2_texture",
        "feature_dim": 30,
        "gray_levels": GRAY_LEVELS,
        "glcm_distance": GLCM_DISTANCE,
        "glcm_directions": GLCM_DIRECTIONS,
        "glcm_properties": GLCM_PROPS,
        "glcm_symmetric": True,
        "lbp_points": LBP_POINTS,
        "lbp_radius": LBP_RADIUS,
        "lbp_bins": LBP_BINS,
        "lbp_mapping": "uniform patterns by number of ones; non-uniform patterns in final bin",
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
