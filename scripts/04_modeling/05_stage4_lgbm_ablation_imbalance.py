"""Stage 4: LightGBM ablation and class-imbalance comparison.

Notebook block purpose:
    Stage 4 LightGBM ablation and imbalance analysis. This block keeps the
    Stage 3 LightGBM parameter anchor fixed, compares selected feature-group
    removals, and then compares no weighting, mild sqrt-balanced weighting, and
    full balanced sample weighting on the two strongest feature candidates.
    The test split is intentionally not used.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATH = PROJECT_ROOT / "vendor" / "python"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))

import numpy as np  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402


RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
RESULT_CSV = REPORT_DIR / "stage4_lgbm_ablation_imbalance.csv"
ABLATION_SUMMARY_CSV = REPORT_DIR / "stage4_lgbm_ablation_summary.csv"
IMBALANCE_SUMMARY_CSV = REPORT_DIR / "stage4_lgbm_imbalance_summary.csv"
SUMMARY_JSON = REPORT_DIR / "stage4_lgbm_ablation_imbalance_summary.json"
DEPENDENCY_JSON = REPORT_DIR / "stage4_dependency_versions.json"

FEATURE_CONFIGS = {
    "F1_color": PROJECT_ROOT / "data" / "features" / "F1_color" / "F1_color_1039.npz",
    "F2_texture": PROJECT_ROOT / "data" / "features" / "F2_texture" / "F2_texture_30.npz",
    "F3_hog": PROJECT_ROOT / "data" / "features" / "F3_hog" / "F3_hog_1764.npz",
    "F4_shape": PROJECT_ROOT / "data" / "features" / "F4_shape" / "F4_shape_12.npz",
    "F5_gist": PROJECT_ROOT / "data" / "features" / "F5_gist" / "F5_gist_64.npz",
    "F6_bof": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_bof_256.npz",
}

COMBO_CONFIGS = {
    "C11_full": {
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog", "F6_bof"],
        "stage3_macro_f1": 0.8096497497,
        "stage3_minority_recall_mean": 0.8157689034,
        "reason": "Full handcrafted C11 reference and Stage 3 best.",
    },
    "no_shape": {
        "feature_ids": ["F3_hog", "F1_color", "F2_texture", "F5_gist", "F6_bof"],
        "stage3_macro_f1": 0.8071305522,
        "stage3_minority_recall_mean": 0.8108669426,
        "reason": "Drop F4_shape from full adaptive candidate.",
    },
    "no_bof": {
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog"],
        "stage3_macro_f1": 0.8019770252,
        "stage3_minority_recall_mean": 0.7961176801,
        "reason": "Drop F6_bof from C11.",
    },
    "no_gist": {
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F3_hog", "F6_bof"],
        "stage3_macro_f1": "",
        "stage3_minority_recall_mean": "",
        "reason": "Drop F5_gist from C11; not directly run in Stage 3.",
    },
    "no_shape_no_bof": {
        "feature_ids": ["F3_hog", "F1_color", "F2_texture", "F5_gist"],
        "stage3_macro_f1": 0.7947198122,
        "stage3_minority_recall_mean": 0.7963977922,
        "reason": "Compact adaptive candidate without F4_shape and F6_bof.",
    },
    "hog_color_texture": {
        "feature_ids": ["F3_hog", "F1_color", "F2_texture"],
        "stage3_macro_f1": 0.8011968050,
        "stage3_minority_recall_mean": 0.8005701396,
        "reason": "Tight HOG+Color+Texture main candidate.",
    },
    "lowdim_color_texture_gist": {
        "feature_ids": ["F1_color", "F2_texture", "F5_gist"],
        "stage3_macro_f1": 0.7844679925,
        "stage3_minority_recall_mean": 0.8103740611,
        "reason": "Low-dimensional interpretable candidate.",
    },
}

RUN_CONFIGS = [
    {"run_id": "AB_C11_full_balanced", "experiment_type": "ablation", "combo_id": "C11_full", "sample_weight_mode": "balanced"},
    {"run_id": "AB_no_shape_balanced", "experiment_type": "ablation", "combo_id": "no_shape", "sample_weight_mode": "balanced"},
    {"run_id": "AB_no_bof_balanced", "experiment_type": "ablation", "combo_id": "no_bof", "sample_weight_mode": "balanced"},
    {"run_id": "AB_no_gist_balanced", "experiment_type": "ablation", "combo_id": "no_gist", "sample_weight_mode": "balanced"},
    {"run_id": "AB_no_shape_no_bof_balanced", "experiment_type": "ablation", "combo_id": "no_shape_no_bof", "sample_weight_mode": "balanced"},
    {"run_id": "AB_hog_color_texture_balanced", "experiment_type": "ablation", "combo_id": "hog_color_texture", "sample_weight_mode": "balanced"},
    {"run_id": "AB_lowdim_color_texture_gist_balanced", "experiment_type": "ablation", "combo_id": "lowdim_color_texture_gist", "sample_weight_mode": "balanced"},
    {"run_id": "IMB_C11_full_none", "experiment_type": "imbalance", "combo_id": "C11_full", "sample_weight_mode": "none"},
    {"run_id": "IMB_C11_full_sqrt_balanced", "experiment_type": "imbalance", "combo_id": "C11_full", "sample_weight_mode": "sqrt_balanced"},
    {"run_id": "IMB_no_shape_none", "experiment_type": "imbalance", "combo_id": "no_shape", "sample_weight_mode": "none"},
    {"run_id": "IMB_no_shape_sqrt_balanced", "experiment_type": "imbalance", "combo_id": "no_shape", "sample_weight_mode": "sqrt_balanced"},
]

RESULT_FIELDNAMES = [
    "run_id",
    "experiment_type",
    "combo_id",
    "feature_ids",
    "feature_dim_raw",
    "feature_dim_after_variance",
    "sample_weight_mode",
    "stage3_macro_f1",
    "stage3_minority_recall_mean",
    "validation_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "validation_weighted_f1",
    "minority_recall_mean",
    "delta_macro_f1_vs_C11_none",
    "delta_minority_recall_vs_C11_none",
    "delta_macro_f1_vs_stage3",
    "delta_minority_recall_vs_stage3",
    "fit_seconds",
    "predict_seconds",
    "total_seconds",
    "params",
    "warnings",
    "recall_battery",
    "f1_battery",
    "recall_biological",
    "f1_biological",
    "recall_trash",
    "f1_trash",
]


def log(message: str) -> None:
    """Print a flush-safe progress message for long runs."""
    print(message, flush=True)


def module_version(module_name: str) -> str:
    """Return an importable module version string."""
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def write_dependency_versions() -> None:
    """Save dependency versions used by Stage 4."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": module_version("numpy"),
        "scikit_learn": module_version("sklearn"),
        "lightgbm": module_version("lightgbm"),
    }
    DEPENDENCY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with DEPENDENCY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(versions, handle, ensure_ascii=False, indent=2)


def jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-friendly values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value.__class__.__name__


def params_text(params: dict[str, Any]) -> str:
    """Compact parameter summary for CSV readability."""
    return json.dumps(jsonable(params), ensure_ascii=False, sort_keys=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows when the file exists."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append a single checkpoint row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_feature_cache(path: Path) -> dict[str, Any]:
    """Load train and validation data from one feature cache."""
    cache = np.load(path, allow_pickle=False)
    return {
        "X_train": cache["X_train"],
        "y_train": cache["y_train"],
        "X_validation": cache["X_validation"],
        "y_validation": cache["y_validation"],
        "label_names": cache["label_names"].tolist(),
        "image_ids_train": cache["image_ids_train"],
        "image_ids_validation": cache["image_ids_validation"],
    }


def load_all_features() -> dict[str, dict[str, Any]]:
    """Load all needed feature blocks and validate split alignment."""
    store: dict[str, dict[str, Any]] = {}
    for feature_id, path in FEATURE_CONFIGS.items():
        store[feature_id] = load_feature_cache(path)
        log(
            f"[load] {feature_id}: train {store[feature_id]['X_train'].shape}, "
            f"validation {store[feature_id]['X_validation'].shape}"
        )

    reference = store["F1_color"]
    for feature_id, data in store.items():
        for split in ("train", "validation"):
            if not np.array_equal(reference[f"y_{split}"], data[f"y_{split}"]):
                raise ValueError(f"Label mismatch for {feature_id} / {split}")
            if not np.array_equal(reference[f"image_ids_{split}"], data[f"image_ids_{split}"]):
                raise ValueError(f"Image-id mismatch for {feature_id} / {split}")
        if reference["label_names"] != data["label_names"]:
            raise ValueError(f"Label-name mismatch for {feature_id}")
    log("[check] Feature split, image-id, and label alignment passed.")
    return store


def build_combo_matrix(
    feature_store: dict[str, dict[str, Any]],
    feature_ids: list[str],
    split: str,
) -> np.ndarray:
    """Concatenate feature groups for one split."""
    pieces = [feature_store[feature_id][f"X_{split}"] for feature_id in feature_ids]
    matrix = pieces[0] if len(pieces) == 1 else np.hstack(pieces)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {feature_ids} / {split}")
    return matrix.astype(np.float32, copy=False)


def lightgbm_stage3_params() -> dict[str, Any]:
    """Return fixed Stage 3 LightGBM anchor parameters."""
    return {
        "objective": "multiclass",
        "n_estimators": 460,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
    }


def sample_weight_for_mode(y: np.ndarray, mode: str) -> np.ndarray | None:
    """Return sample weights for one imbalance setting."""
    if mode == "none":
        return None
    balanced = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    if mode == "balanced":
        return balanced
    if mode == "sqrt_balanced":
        weights = np.sqrt(balanced)
        return weights / np.mean(weights)
    raise ValueError(f"Unsupported sample_weight_mode: {mode}")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    """Compute validation metrics and selected minority-class metrics."""
    labels = list(range(len(label_names)))
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    metrics = {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "validation_macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    for label in MINORITY_LABELS:
        label_index = label_names.index(label)
        metrics[f"recall_{label}"] = float(per_class_recall[label_index])
        metrics[f"f1_{label}"] = float(per_class_f1[label_index])
    metrics["minority_recall_mean"] = float(
        np.mean([metrics[f"recall_{label}"] for label in MINORITY_LABELS])
    )
    return metrics


def warning_summary(caught: list[warnings.WarningMessage]) -> str:
    """Return compact warning text."""
    messages: list[str] = []
    for item in caught:
        message = str(item.message).replace("\n", " ")
        if message not in messages:
            messages.append(message)
    return " | ".join(messages[:5])


def as_float(value: Any, default: float = float("nan")) -> float:
    """Convert a possibly empty value to float."""
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def run_one(
    config: dict[str, str],
    feature_store: dict[str, dict[str, Any]],
    c11_reference: dict[str, float] | None,
) -> dict[str, Any]:
    """Run one fixed LightGBM ablation or imbalance experiment."""
    combo = COMBO_CONFIGS[config["combo_id"]]
    feature_ids = list(combo["feature_ids"])
    X_train_raw = build_combo_matrix(feature_store, feature_ids, "train")
    X_validation_raw = build_combo_matrix(feature_store, feature_ids, "validation")
    y_train = feature_store[feature_ids[0]]["y_train"]
    y_validation = feature_store[feature_ids[0]]["y_validation"]
    label_names = feature_store[feature_ids[0]]["label_names"]

    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train_raw)
    X_validation = selector.transform(X_validation_raw)

    params = lightgbm_stage3_params()
    params["num_class"] = len(label_names)
    sample_weight = sample_weight_for_mode(y_train, config["sample_weight_mode"])
    model = LGBMClassifier(**params)

    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as predict_warnings:
        warnings.simplefilter("always")
        y_pred = np.asarray(model.predict(X_validation), dtype=np.int64)
    predict_seconds = time.perf_counter() - predict_start
    caught.extend(predict_warnings)

    metrics = evaluate_predictions(y_validation, y_pred, label_names)
    stage3_macro = combo["stage3_macro_f1"]
    stage3_minority = combo["stage3_minority_recall_mean"]
    row: dict[str, Any] = {
        "run_id": config["run_id"],
        "experiment_type": config["experiment_type"],
        "combo_id": config["combo_id"],
        "feature_ids": "+".join(feature_ids),
        "feature_dim_raw": int(X_train_raw.shape[1]),
        "feature_dim_after_variance": int(X_train.shape[1]),
        "sample_weight_mode": config["sample_weight_mode"],
        "stage3_macro_f1": stage3_macro,
        "stage3_minority_recall_mean": stage3_minority,
        "fit_seconds": round(float(fit_seconds), 4),
        "predict_seconds": round(float(predict_seconds), 4),
        "total_seconds": round(float(fit_seconds + predict_seconds), 4),
        "params": params_text(params),
        "warnings": warning_summary(caught),
    }
    row.update({key: round(value, 10) for key, value in metrics.items()})

    if c11_reference is None:
        row["delta_macro_f1_vs_C11_none"] = 0.0
        row["delta_minority_recall_vs_C11_none"] = 0.0
    else:
        row["delta_macro_f1_vs_C11_none"] = round(
            metrics["validation_macro_f1"] - c11_reference["validation_macro_f1"],
            10,
        )
        row["delta_minority_recall_vs_C11_none"] = round(
            metrics["minority_recall_mean"] - c11_reference["minority_recall_mean"],
            10,
        )

    if stage3_macro == "":
        row["delta_macro_f1_vs_stage3"] = ""
    else:
        row["delta_macro_f1_vs_stage3"] = round(
            metrics["validation_macro_f1"] - float(stage3_macro),
            10,
        )
    if stage3_minority == "":
        row["delta_minority_recall_vs_stage3"] = ""
    else:
        row["delta_minority_recall_vs_stage3"] = round(
            metrics["minority_recall_mean"] - float(stage3_minority),
            10,
        )
    return row


def summarize(rows: list[dict[str, Any]]) -> None:
    """Write ablation and imbalance summaries."""
    if not rows:
        return
    ablation_rows = [
        row for row in rows if row["experiment_type"] == "ablation" and row["sample_weight_mode"] == "balanced"
    ]
    ablation_rows = sorted(
        ablation_rows,
        key=lambda row: as_float(row["validation_macro_f1"]),
        reverse=True,
    )
    write_csv(ABLATION_SUMMARY_CSV, ablation_rows, RESULT_FIELDNAMES)

    imbalance_rows = [
        row
        for row in rows
        if row["combo_id"] in {"C11_full", "no_shape"}
    ]
    imbalance_rows = sorted(
        imbalance_rows,
        key=lambda row: (row["combo_id"], as_float(row["validation_macro_f1"]) * -1),
    )
    write_csv(IMBALANCE_SUMMARY_CSV, imbalance_rows, RESULT_FIELDNAMES)

    best_macro = max(rows, key=lambda row: as_float(row["validation_macro_f1"]))
    best_minority = max(rows, key=lambda row: as_float(row["minority_recall_mean"]))
    summary = {
        "num_runs": len(rows),
        "best_macro_run": best_macro,
        "best_minority_recall_run": best_minority,
        "best_ablation_rows": ablation_rows[:5],
        "outputs": {
            "result_csv": str(RESULT_CSV.relative_to(PROJECT_ROOT)),
            "ablation_summary_csv": str(ABLATION_SUMMARY_CSV.relative_to(PROJECT_ROOT)),
            "imbalance_summary_csv": str(IMBALANCE_SUMMARY_CSV.relative_to(PROJECT_ROOT)),
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Stage 4 LightGBM ablation and imbalance comparison.")
    parser.add_argument("--resume", action="store_true", help="Skip completed run IDs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit new runs for smoke tests.")
    parser.add_argument(
        "--runs",
        default="",
        help="Comma-separated run IDs to execute. Empty means all runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run fixed LightGBM ablation and imbalance experiments."""
    args = parse_args()
    write_dependency_versions()
    feature_store = load_all_features()

    selected = {item.strip() for item in args.runs.split(",") if item.strip()}
    configs = [config for config in RUN_CONFIGS if not selected or config["run_id"] in selected]
    if selected:
        found = {config["run_id"] for config in configs}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"Unknown run IDs: {missing}")
    existing = {row["run_id"]: row for row in read_csv_rows(RESULT_CSV)} if args.resume else {}
    pending = [config for config in configs if not args.resume or config["run_id"] not in existing]
    if args.max_runs is not None:
        pending = pending[: args.max_runs]

    if not args.resume:
        write_csv(RESULT_CSV, [], RESULT_FIELDNAMES)

    log(
        f"[plan] Stage 4 LightGBM ablation/imbalance: {len(configs)} selected runs, "
        f"{len(pending)} pending runs."
    )
    log(
        "[estimate] Full run is expected to take roughly 30-45 minutes. "
        "Each fit writes a checkpoint row immediately."
    )

    c11_reference: dict[str, float] | None = None
    if args.resume:
        for row in completed_rows:
            if row.get("run_id") == "AB_C11_full_balanced":
                c11_reference = {
                    "validation_macro_f1": float(row["validation_macro_f1"]),
                    "minority_recall_mean": float(row["minority_recall_mean"]),
                }
                break
    completed_rows = list(read_csv_rows(RESULT_CSV)) if args.resume else []
    for index, config in enumerate(pending, start=1):
        combo = COMBO_CONFIGS[config["combo_id"]]
        raw_dim = sum(
            feature_store[feature_id]["X_train"].shape[1]
            for feature_id in combo["feature_ids"]
        )
        log(
            f"[run {index}/{len(pending)}] {config['run_id']} | "
            f"combo={config['combo_id']} | raw_dim={raw_dim} | "
            f"weight={config['sample_weight_mode']}"
        )
        run_start = time.perf_counter()
        row = run_one(config, feature_store, c11_reference)
        append_csv_row(RESULT_CSV, row, RESULT_FIELDNAMES)
        completed_rows.append(row)
        if row["run_id"] == "AB_C11_full_balanced":
            c11_reference = {
                "validation_macro_f1": float(row["validation_macro_f1"]),
                "minority_recall_mean": float(row["minority_recall_mean"]),
            }
        log(
            f"[done] {config['run_id']} | macro-F1={float(row['validation_macro_f1']):.4f} | "
            f"minority={float(row['minority_recall_mean']):.4f} | "
            f"time={time.perf_counter() - run_start:.1f}s"
        )

    all_rows = read_csv_rows(RESULT_CSV)
    summarize(all_rows)
    log("[complete] Stage 4 LightGBM ablation/imbalance summaries written.")


if __name__ == "__main__":
    main()
