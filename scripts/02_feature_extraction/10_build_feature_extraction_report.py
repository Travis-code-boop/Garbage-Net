"""Build a consolidated report for all extracted feature caches.

Notebook block purpose:
    Stage 1 feature extraction report. This code scans the feature cache files,
    verifies dimensions and split shapes, and writes summary tables used before
    moving into Stage 2 single-feature model screening.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "feature_reports"
REPORT_CSV_PATH = REPORT_DIR / "feature_extraction_report.csv"
REPORT_JSON_PATH = REPORT_DIR / "feature_extraction_report.json"


FEATURE_CACHES = [
    {
        "feature_id": "F0_pixels_gray",
        "planned_role": "C0 primary raw-pixel baseline",
        "path": FEATURE_DIR / "F0_pixels" / "F0_pixels_gray_1024.npz",
        "expected_dim": 1024,
    },
    {
        "feature_id": "F0_pixels_rgb",
        "planned_role": "optional raw-pixel baseline",
        "path": FEATURE_DIR / "F0_pixels" / "F0_pixels_rgb_3072.npz",
        "expected_dim": 3072,
    },
    {
        "feature_id": "F1_color",
        "planned_role": "C1 color feature group",
        "path": FEATURE_DIR / "F1_color" / "F1_color_1039.npz",
        "expected_dim": 1039,
    },
    {
        "feature_id": "F2_texture",
        "planned_role": "C2 texture feature group",
        "path": FEATURE_DIR / "F2_texture" / "F2_texture_30.npz",
        "expected_dim": 30,
    },
    {
        "feature_id": "F3_hog",
        "planned_role": "C3 HOG edge feature group",
        "path": FEATURE_DIR / "F3_hog" / "F3_hog_1764.npz",
        "expected_dim": 1764,
    },
    {
        "feature_id": "F4_shape",
        "planned_role": "C4 shape feature group",
        "path": FEATURE_DIR / "F4_shape" / "F4_shape_12.npz",
        "expected_dim": 12,
    },
    {
        "feature_id": "F5_gist",
        "planned_role": "C5 GIST global structure feature group",
        "path": FEATURE_DIR / "F5_gist" / "F5_gist_64.npz",
        "expected_dim": 64,
    },
    {
        "feature_id": "F6_sift_bof",
        "planned_role": "SIFT part of local BoF feature group",
        "path": FEATURE_DIR / "F6_bof" / "F6_sift_bof_128.npz",
        "expected_dim": 128,
    },
    {
        "feature_id": "F6_orb_bof",
        "planned_role": "ORB part of local BoF feature group",
        "path": FEATURE_DIR / "F6_bof" / "F6_orb_bof_128.npz",
        "expected_dim": 128,
    },
    {
        "feature_id": "F6_bof",
        "planned_role": "C6 full local BoF feature group",
        "path": FEATURE_DIR / "F6_bof" / "F6_bof_256.npz",
        "expected_dim": 256,
    },
]


def load_params(cache: np.lib.npyio.NpzFile) -> dict[str, object]:
    """Read params_json from a feature cache."""
    if "params_json" not in cache.files:
        return {}
    return json.loads(str(cache["params_json"]))


def cache_row(config: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Summarize one feature cache as CSV row plus full JSON details."""
    path = Path(config["path"])
    if not path.exists():
        missing_row = {
            "feature_id": config["feature_id"],
            "planned_role": config["planned_role"],
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": False,
            "expected_dim": config["expected_dim"],
            "actual_dim": "",
            "dim_ok": False,
            "train_shape": "",
            "validation_shape": "",
            "test_shape": "",
            "nan_count_total": "",
            "inf_count_total": "",
            "feature_groups": "",
            "label_names": "",
            "status": "missing",
        }
        return missing_row, {"config": config, "status": "missing"}

    cache = np.load(path, allow_pickle=False)
    params = load_params(cache)
    feature_names = cache["feature_names"]
    feature_groups = cache["feature_groups"]
    label_names = cache["label_names"]

    split_details = {}
    nan_total = 0
    inf_total = 0
    for split in ("train", "validation", "test"):
        matrix = cache[f"X_{split}"]
        nan_count = int(np.isnan(matrix).sum())
        inf_count = int(np.isinf(matrix).sum())
        nan_total += nan_count
        inf_total += inf_count
        split_details[split] = {
            "shape": list(matrix.shape),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "min": float(np.nanmin(matrix)),
            "max": float(np.nanmax(matrix)),
        }

    actual_dim = int(len(feature_names))
    expected_dim = int(config["expected_dim"])
    dim_ok = actual_dim == expected_dim
    split_dim_ok = all(split_details[split]["shape"][1] == expected_dim for split in split_details)
    status = "ok" if dim_ok and split_dim_ok and nan_total == 0 and inf_total == 0 else "check"

    row = {
        "feature_id": config["feature_id"],
        "planned_role": config["planned_role"],
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "expected_dim": expected_dim,
        "actual_dim": actual_dim,
        "dim_ok": dim_ok and split_dim_ok,
        "train_shape": "x".join(map(str, split_details["train"]["shape"])),
        "validation_shape": "x".join(map(str, split_details["validation"]["shape"])),
        "test_shape": "x".join(map(str, split_details["test"]["shape"])),
        "nan_count_total": nan_total,
        "inf_count_total": inf_total,
        "feature_groups": "|".join(sorted(set(feature_groups.tolist()))),
        "label_names": "|".join(label_names.tolist()),
        "status": status,
    }

    detail = {
        "feature_id": config["feature_id"],
        "planned_role": config["planned_role"],
        "path": str(path.relative_to(PROJECT_ROOT)),
        "expected_dim": expected_dim,
        "actual_dim": actual_dim,
        "dim_ok": dim_ok,
        "split_dim_ok": split_dim_ok,
        "feature_groups": sorted(set(feature_groups.tolist())),
        "label_names": label_names.tolist(),
        "split_details": split_details,
        "params": params,
        "status": status,
    }
    return row, detail


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write report rows to CSV."""
    fieldnames = [
        "feature_id",
        "planned_role",
        "path",
        "exists",
        "expected_dim",
        "actual_dim",
        "dim_ok",
        "train_shape",
        "validation_shape",
        "test_shape",
        "nan_count_total",
        "inf_count_total",
        "feature_groups",
        "label_names",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    for config in FEATURE_CACHES:
        row, detail = cache_row(config)
        rows.append(row)
        details.append(detail)

    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    report = {
        "status_counts": status_counts,
        "num_feature_caches": len(rows),
        "report_csv": str(REPORT_CSV_PATH.relative_to(PROJECT_ROOT)),
        "features": details,
    }

    write_csv(REPORT_CSV_PATH, rows)
    REPORT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
