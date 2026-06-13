"""Merge SIFT-BoF and ORB-BoF into the planned F6 256-dimensional feature set.

Notebook block purpose:
    F6 final BoF feature construction. This code concatenates the already
    extracted SIFT-BoF and ORB-BoF matrices along the feature axis, preserving
    the fixed train/validation/test split and metadata arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_io import FEATURE_DIR, PROJECT_ROOT


OUTPUT_DIR = FEATURE_DIR / "F6_bof"
SIFT_CACHE_PATH = OUTPUT_DIR / "F6_sift_bof_128.npz"
ORB_CACHE_PATH = OUTPUT_DIR / "F6_orb_bof_128.npz"
MERGED_CACHE_PATH = OUTPUT_DIR / "F6_bof_256.npz"
MERGED_SUMMARY_PATH = OUTPUT_DIR / "F6_bof_256_summary.json"

SPLITS = ("train", "validation", "test")


def assert_matching_metadata(sift_cache: np.lib.npyio.NpzFile, orb_cache: np.lib.npyio.NpzFile) -> None:
    """Ensure both feature caches use identical split ordering and labels."""
    for split in SPLITS:
        for key_prefix in ("image_ids", "labels", "paths", "y"):
            key = f"{key_prefix}_{split}"
            if not np.array_equal(sift_cache[key], orb_cache[key]):
                raise ValueError(f"Metadata mismatch for {key}")
    if not np.array_equal(sift_cache["label_names"], orb_cache["label_names"]):
        raise ValueError("label_names mismatch")


def summarize(split_features: dict[str, np.ndarray], feature_names: np.ndarray, feature_groups: np.ndarray) -> dict:
    """Summarize merged feature cache quality."""
    summary = {
        "num_features": int(len(feature_names)),
        "num_feature_group_entries": int(len(feature_groups)),
        "feature_groups": sorted(set(feature_groups.tolist())),
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


def main() -> None:
    sift_cache = np.load(SIFT_CACHE_PATH)
    orb_cache = np.load(ORB_CACHE_PATH)
    assert_matching_metadata(sift_cache, orb_cache)

    split_features = {}
    payload: dict[str, object] = {}
    for split in SPLITS:
        matrix = np.concatenate(
            [sift_cache[f"X_{split}"], orb_cache[f"X_{split}"]],
            axis=1,
        ).astype(np.float32)
        split_features[split] = matrix
        payload[f"X_{split}"] = matrix
        payload[f"y_{split}"] = sift_cache[f"y_{split}"]
        payload[f"labels_{split}"] = sift_cache[f"labels_{split}"]
        payload[f"image_ids_{split}"] = sift_cache[f"image_ids_{split}"]
        payload[f"paths_{split}"] = sift_cache[f"paths_{split}"]

    feature_names = np.concatenate([sift_cache["feature_names"], orb_cache["feature_names"]])
    feature_groups = np.concatenate([sift_cache["feature_groups"], orb_cache["feature_groups"]])
    params = {
        "feature_id": "F6_bof",
        "feature_dim": 256,
        "parts": [
            {
                "name": "sift_bof",
                "dim": 128,
                "cache": str(SIFT_CACHE_PATH.relative_to(PROJECT_ROOT)),
                "vocab": str((OUTPUT_DIR / "vocab_sift_k128.npy").relative_to(PROJECT_ROOT)),
            },
            {
                "name": "orb_bof",
                "dim": 128,
                "cache": str(ORB_CACHE_PATH.relative_to(PROJECT_ROOT)),
                "vocab": str((OUTPUT_DIR / "vocab_orb_k128.npy").relative_to(PROJECT_ROOT)),
            },
        ],
        "merge": "column-wise concatenation [SIFT-BoF, ORB-BoF]",
    }

    payload["feature_names"] = feature_names
    payload["feature_groups"] = feature_groups
    payload["label_names"] = sift_cache["label_names"]
    payload["label_to_index_json"] = sift_cache["label_to_index_json"]
    payload["params_json"] = json.dumps(params, ensure_ascii=False, indent=2)

    np.savez_compressed(MERGED_CACHE_PATH, **payload)

    summary = summarize(split_features, feature_names, feature_groups)
    summary["output_path"] = str(MERGED_CACHE_PATH.relative_to(PROJECT_ROOT))
    summary["params"] = params
    MERGED_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
