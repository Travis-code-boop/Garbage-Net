"""Extract F1 color features with the planned 1039-dimensional design.

Notebook block purpose:
    F1 color feature extraction. For each image, this code creates a fixed
    feature vector made of HSV channel statistics, an HSV 3D histogram, and a
    BGR 3D histogram. The resulting split feature matrices are saved for later
    single-feature model screening.

Feature dimensions:
    HSV statistics: 3 channels x 5 statistics = 15
    HSV histogram: 8 x 8 x 8 = 512
    BGR histogram: 8 x 8 x 8 = 512
    Total: 1039
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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


OUTPUT_DIR = FEATURE_DIR / "F1_color"
FEATURE_CACHE_PATH = OUTPUT_DIR / "F1_color_1039.npz"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "F1_color_1039_summary.json"

HIST_BINS = 8
STAT_CHANNELS = ("H", "S", "V")
STAT_NAMES = ("mean", "std", "skewness", "kurtosis_excess", "entropy_norm")
MIN_STD_FOR_MOMENTS = 1e-12


def normalized_entropy_uint8(values: np.ndarray) -> float:
    """Compute entropy over 256 bins and normalize it to the [0, 1] range."""
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy / 8.0)


def channel_statistics(channel: np.ndarray) -> list[float]:
    """Compute mean, std, skewness, excess kurtosis, and normalized entropy."""
    values = channel.astype(np.float64) / 255.0
    mean = float(values.mean())
    std = float(values.std())

    if std < MIN_STD_FOR_MOMENTS:
        skewness = 0.0
        kurtosis_excess = 0.0
    else:
        centered = values - mean
        skewness = float(np.mean(centered**3) / (std**3))
        kurtosis_excess = float(np.mean(centered**4) / (std**4) - 3.0)

    entropy = normalized_entropy_uint8(channel)
    return [mean, std, skewness, kurtosis_excess, entropy]


def normalized_histogram_3d(image: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Compute a normalized 3D color histogram for an 8-bit 3-channel image."""
    hist, _ = np.histogramdd(
        image.reshape(-1, 3),
        bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.ravel()


def build_feature_names(bins: int = HIST_BINS) -> tuple[list[str], list[str]]:
    """Build stable feature names and group labels for the 1039 color features."""
    feature_names: list[str] = []
    feature_groups: list[str] = []

    for channel in STAT_CHANNELS:
        for stat_name in STAT_NAMES:
            feature_names.append(f"color_hsv_{channel}_{stat_name}")
            feature_groups.append("color_hsv_stats")

    for space in ("hsv", "bgr"):
        for first in range(bins):
            for second in range(bins):
                for third in range(bins):
                    if space == "hsv":
                        name = f"color_hsv_hist_h{first:02d}_s{second:02d}_v{third:02d}"
                    else:
                        name = f"color_bgr_hist_b{first:02d}_g{second:02d}_r{third:02d}"
                    feature_names.append(name)
                    feature_groups.append(f"color_{space}_hist")

    return feature_names, feature_groups


def extract_color_features_for_image(relative_path: str) -> np.ndarray:
    """Extract the planned 1039-dimensional color feature vector for one image."""
    versions = prepare_image_versions(relative_path)
    hsv = versions.hsv
    bgr = versions.rgb[:, :, ::-1]

    stats: list[float] = []
    for channel_index in range(3):
        stats.extend(channel_statistics(hsv[:, :, channel_index]))

    hsv_hist = normalized_histogram_3d(hsv)
    bgr_hist = normalized_histogram_3d(bgr)
    features = np.concatenate(
        [
            np.asarray(stats, dtype=np.float64),
            hsv_hist,
            bgr_hist,
        ]
    )
    return features.astype(np.float32)


def extract_split_features(split_rows: list[dict[str, str]]) -> np.ndarray:
    """Extract color features for all rows in one split."""
    matrix = np.empty((len(split_rows), 1039), dtype=np.float32)
    for index, row in enumerate(split_rows):
        matrix[index] = extract_color_features_for_image(row["path"])
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
        "feature_id": "F1_color",
        "feature_dim": 1039,
        "hsv_stat_channels": STAT_CHANNELS,
        "hsv_stat_names": STAT_NAMES,
        "hist_bins": [HIST_BINS, HIST_BINS, HIST_BINS],
        "hist_spaces": ["HSV", "BGR"],
        "hist_normalization": "sum_to_one_per_image",
        "stats_value_scale": "channel_values_divided_by_255_for moments",
        "min_std_for_skewness_kurtosis": "only zero-variance guard",
        "entropy": "256-bin entropy normalized by log2(256)",
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
