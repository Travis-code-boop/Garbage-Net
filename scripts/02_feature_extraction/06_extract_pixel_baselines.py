"""Extract F0 raw pixel baseline features.

Notebook block purpose:
    F0 raw pixel baseline extraction. This code creates simple downsampled
    pixel vectors that serve as the lowest-cost baseline before hand-crafted
    color, texture, HOG, shape, GIST, and BoF features are evaluated.

Feature dimensions:
    Gray baseline: 32 x 32 = 1024
    RGB baseline: 32 x 32 x 3 = 3072
"""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

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


OUTPUT_DIR = FEATURE_DIR / "F0_pixels"
GRAY_CACHE_PATH = OUTPUT_DIR / "F0_pixels_gray_1024.npz"
GRAY_SUMMARY_PATH = OUTPUT_DIR / "F0_pixels_gray_1024_summary.json"
RGB_CACHE_PATH = OUTPUT_DIR / "F0_pixels_rgb_3072.npz"
RGB_SUMMARY_PATH = OUTPUT_DIR / "F0_pixels_rgb_3072_summary.json"

PIXEL_SIZE = 32


def resize_array(array: np.ndarray, mode: str) -> np.ndarray:
    """Resize a grayscale or RGB uint8 array to 32x32."""
    image = Image.fromarray(array, mode=mode)
    resized = image.resize((PIXEL_SIZE, PIXEL_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def extract_pixel_features_for_image(relative_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract gray 1024-dimensional and RGB 3072-dimensional pixel baselines."""
    versions = prepare_image_versions(relative_path)
    gray_32 = resize_array(versions.gray, mode="L").reshape(-1)
    rgb_32 = resize_array(versions.rgb, mode="RGB").reshape(-1)
    return gray_32.astype(np.float32), rgb_32.astype(np.float32)


def build_feature_names() -> tuple[list[str], list[str], list[str], list[str]]:
    """Build feature names and groups for gray and RGB pixel baselines."""
    gray_names = [f"pixel_gray_y{y:02d}_x{x:02d}" for y in range(PIXEL_SIZE) for x in range(PIXEL_SIZE)]
    gray_groups = ["pixel_gray"] * len(gray_names)

    rgb_names = []
    rgb_groups = []
    for y in range(PIXEL_SIZE):
        for x in range(PIXEL_SIZE):
            for channel in ("R", "G", "B"):
                rgb_names.append(f"pixel_rgb_y{y:02d}_x{x:02d}_{channel}")
                rgb_groups.append("pixel_rgb")

    return gray_names, gray_groups, rgb_names, rgb_groups


def extract_split_features(split_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract gray and RGB pixel baselines for all rows in one split."""
    gray_matrix = np.empty((len(split_rows), 1024), dtype=np.float32)
    rgb_matrix = np.empty((len(split_rows), 3072), dtype=np.float32)

    for index, row in enumerate(split_rows):
        gray_matrix[index], rgb_matrix[index] = extract_pixel_features_for_image(row["path"])
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(split_rows)} images for split")

    return gray_matrix, rgb_matrix


def write_summary(path, summary: dict[str, object]) -> None:
    """Write a JSON summary file."""
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    rows = read_metadata_with_split()
    split_rows = rows_by_split(rows)
    mapping = label_mapping(rows)
    gray_names, gray_groups, rgb_names, rgb_groups = build_feature_names()

    gray_features = {}
    rgb_features = {}
    for split_name, current_rows in split_rows.items():
        print(f"extracting {split_name}: {len(current_rows)} images")
        gray_features[split_name], rgb_features[split_name] = extract_split_features(current_rows)

    common_params = {
        "feature_id": "F0_pixels",
        "pixel_size": [PIXEL_SIZE, PIXEL_SIZE],
        "value_scale": "uint8 divided by 255",
        "source_metadata": str(
            (PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_with_split.csv").relative_to(
                PROJECT_ROOT
            )
        ),
    }

    gray_params = {
        **common_params,
        "variant": "gray",
        "feature_dim": 1024,
        "input": "grayscale image resized to 32x32",
    }
    save_feature_cache(
        GRAY_CACHE_PATH,
        split_rows,
        gray_features,
        gray_names,
        gray_groups,
        gray_params,
        mapping,
    )
    gray_summary = summarize_feature_cache(gray_features, gray_names, gray_groups)
    gray_summary["output_path"] = str(GRAY_CACHE_PATH.relative_to(PROJECT_ROOT))
    gray_summary["elapsed_seconds_total_script"] = round(time.perf_counter() - start_time, 3)
    gray_summary["params"] = gray_params
    write_summary(GRAY_SUMMARY_PATH, gray_summary)

    rgb_params = {
        **common_params,
        "variant": "rgb",
        "feature_dim": 3072,
        "input": "RGB image resized to 32x32",
    }
    save_feature_cache(
        RGB_CACHE_PATH,
        split_rows,
        rgb_features,
        rgb_names,
        rgb_groups,
        rgb_params,
        mapping,
    )
    rgb_summary = summarize_feature_cache(rgb_features, rgb_names, rgb_groups)
    rgb_summary["output_path"] = str(RGB_CACHE_PATH.relative_to(PROJECT_ROOT))
    rgb_summary["elapsed_seconds_total_script"] = round(time.perf_counter() - start_time, 3)
    rgb_summary["params"] = rgb_params
    write_summary(RGB_SUMMARY_PATH, rgb_summary)

    print(
        json.dumps(
            {
                "gray": gray_summary,
                "rgb": rgb_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
