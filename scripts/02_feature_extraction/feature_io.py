"""Shared feature matrix input/output helpers.

Notebook block purpose:
    Feature cache helpers. These utilities keep the split order, label mapping,
    feature names, and feature group names consistent when saving extracted
    feature matrices for later model screening.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_WITH_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_with_split.csv"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"

SPLITS = ("train", "validation", "test")


def read_metadata_with_split(path: Path = METADATA_WITH_SPLIT_PATH) -> list[dict[str, str]]:
    """Read metadata with fixed split assignments."""
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: row["image_id"])


def rows_by_split(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group metadata rows by split with stable image_id ordering."""
    grouped = {split: [] for split in SPLITS}
    for row in rows:
        grouped[row["split"]].append(row)
    for split in SPLITS:
        grouped[split].sort(key=lambda row: row["image_id"])
    return grouped


def label_mapping(rows: list[dict[str, str]]) -> dict[str, int]:
    """Create a stable alphabetical label-to-index mapping."""
    labels = sorted({row["label"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def encode_labels(rows: list[dict[str, str]], mapping: dict[str, int]) -> np.ndarray:
    """Encode string labels as integer labels."""
    return np.asarray([mapping[row["label"]] for row in rows], dtype=np.int64)


def row_values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    """Extract a string metadata column into a NumPy array."""
    return np.asarray([row[key] for row in rows])


def save_feature_cache(
    output_path: Path,
    split_rows: dict[str, list[dict[str, str]]],
    split_features: dict[str, np.ndarray],
    feature_names: list[str],
    feature_groups: list[str],
    params: dict[str, object],
    mapping: dict[str, int],
) -> None:
    """Save split feature matrices plus metadata to a compressed npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "feature_names": np.asarray(feature_names),
        "feature_groups": np.asarray(feature_groups),
        "label_names": np.asarray([label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]),
        "label_to_index_json": json.dumps(mapping, ensure_ascii=False),
        "params_json": json.dumps(params, ensure_ascii=False, indent=2),
    }

    for split in SPLITS:
        rows = split_rows[split]
        payload[f"X_{split}"] = split_features[split].astype(np.float32, copy=False)
        payload[f"y_{split}"] = encode_labels(rows, mapping)
        payload[f"labels_{split}"] = row_values(rows, "label")
        payload[f"image_ids_{split}"] = row_values(rows, "image_id")
        payload[f"paths_{split}"] = row_values(rows, "path")

    np.savez_compressed(output_path, **payload)


def summarize_feature_cache(
    split_features: dict[str, np.ndarray],
    feature_names: list[str],
    feature_groups: list[str],
) -> dict[str, object]:
    """Return shape and numeric quality checks for extracted features."""
    summary: dict[str, object] = {
        "num_features": len(feature_names),
        "num_feature_group_entries": len(feature_groups),
        "feature_groups": sorted(set(feature_groups)),
    }

    for split, matrix in split_features.items():
        summary[split] = {
            "shape": list(matrix.shape),
            "nan_count": int(np.isnan(matrix).sum()),
            "inf_count": int(np.isinf(matrix).sum()),
            "min": float(np.nanmin(matrix)),
            "max": float(np.nanmax(matrix)),
        }

    return summary
