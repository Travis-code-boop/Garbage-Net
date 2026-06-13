"""Final fixed-candidate evaluation on the held-out test split.

Notebook block purpose:
    Final evaluation after model selection is complete. This block freezes the
    selected Stage 4 stacking design, rebuilds out-of-fold meta features on the
    combined train+validation split, refits each base learner on the full
    train+validation split, and evaluates once on the held-out test split.

Important protocol:
    1. Model and hyperparameter choices are fixed before this script touches
       test labels.
    2. Train+validation is used for final fitting.
    3. Test is used only for final metrics, confusion matrices, and error
       analysis outputs.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATH = PROJECT_ROOT / "vendor" / "python"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))

import numpy as np  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402


RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
FINAL_BASE_FOLDS_CSV = REPORT_DIR / "final_base_folds.csv"
FINAL_BASE_RESULTS_CSV = REPORT_DIR / "final_base_results.csv"
FINAL_MODEL_RESULTS_CSV = REPORT_DIR / "final_model_results.csv"
FINAL_PER_CLASS_CSV = REPORT_DIR / "final_per_class_metrics.csv"
FINAL_CONFUSION_CSV = REPORT_DIR / "final_confusion_matrices.csv"
FINAL_PREDICTIONS_CSV = REPORT_DIR / "final_test_predictions.csv"
FINAL_ERROR_PAIRS_CSV = REPORT_DIR / "final_error_pairs.csv"
FINAL_ERROR_EXAMPLES_CSV = REPORT_DIR / "final_error_examples.csv"
FINAL_SUMMARY_JSON = REPORT_DIR / "final_test_evaluation_summary.json"
FINAL_DEPENDENCY_JSON = REPORT_DIR / "final_dependency_versions.json"
FINAL_CACHE_DIR = REPORT_DIR / "final_stacking_cache"

FEATURE_CONFIGS = {
    "F1_color": PROJECT_ROOT / "data" / "features" / "F1_color" / "F1_color_1039.npz",
    "F2_texture": PROJECT_ROOT / "data" / "features" / "F2_texture" / "F2_texture_30.npz",
    "F3_hog": PROJECT_ROOT / "data" / "features" / "F3_hog" / "F3_hog_1764.npz",
    "F4_shape": PROJECT_ROOT / "data" / "features" / "F4_shape" / "F4_shape_12.npz",
    "F5_gist": PROJECT_ROOT / "data" / "features" / "F5_gist" / "F5_gist_64.npz",
    "F6_bof": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_bof_256.npz",
}

COMBO_CONFIGS = {
    "C11_full": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog", "F6_bof"],
    "hog_color_texture": ["F3_hog", "F1_color", "F2_texture"],
    "lowdim_color_texture_gist": ["F1_color", "F2_texture", "F5_gist"],
    "gist_texture_shape": ["F5_gist", "F2_texture", "F4_shape"],
}

BASE_CONFIGS = [
    {
        "base_id": "lr_c11_balanced",
        "group": "small",
        "model_type": "logreg",
        "combo_id": "C11_full",
        "output_type": "proba",
        "params": {"C": 0.8, "max_iter": 700, "class_weight": "balanced"},
    },
    {
        "base_id": "sgd_svm_hct_balanced",
        "group": "small",
        "model_type": "sgd_svm",
        "combo_id": "hog_color_texture",
        "output_type": "decision",
        "params": {
            "alpha": 0.00008,
            "max_iter": 3000,
            "tol": 0.001,
            "class_weight": "balanced",
        },
    },
    {
        "base_id": "rf_lowdim_balanced",
        "group": "small",
        "model_type": "random_forest",
        "combo_id": "lowdim_color_texture_gist",
        "output_type": "proba",
        "params": {
            "n_estimators": 240,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
        },
    },
    {
        "base_id": "knn_compact_gist_texture_shape",
        "group": "small",
        "model_type": "knn",
        "combo_id": "gist_texture_shape",
        "output_type": "proba",
        "params": {"n_neighbors": 11, "weights": "distance", "p": 2},
    },
    {
        "base_id": "lgbm_small_hct_sqrt",
        "group": "small",
        "model_type": "lightgbm",
        "combo_id": "hog_color_texture",
        "output_type": "proba",
        "sample_weight_mode": "sqrt_balanced",
        "params": {
            "n_estimators": 220,
            "num_leaves": 31,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 1.0,
        },
    },
    {
        "base_id": "lgbm_anchor_c11_sqrt",
        "group": "anchor",
        "model_type": "lightgbm",
        "combo_id": "C11_full",
        "output_type": "proba",
        "sample_weight_mode": "sqrt_balanced",
        "params": {
            "n_estimators": 460,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 1.0,
        },
    },
]

FINAL_STACK_BASE_IDS = [
    "lr_c11_balanced",
    "sgd_svm_hct_balanced",
    "rf_lowdim_balanced",
    "knn_compact_gist_texture_shape",
    "lgbm_small_hct_sqrt",
    "lgbm_anchor_c11_sqrt",
]
SMALL_STACK_BASE_IDS = [base_id for base_id in FINAL_STACK_BASE_IDS if base_id != "lgbm_anchor_c11_sqrt"]
PROBA_SMALL_BASE_IDS = [
    "lr_c11_balanced",
    "rf_lowdim_balanced",
    "knn_compact_gist_texture_shape",
    "lgbm_small_hct_sqrt",
]
PROBA_WITH_ANCHOR_BASE_IDS = PROBA_SMALL_BASE_IDS + ["lgbm_anchor_c11_sqrt"]

BASE_FOLD_FIELDNAMES = [
    "base_id",
    "fold_index",
    "combo_id",
    "feature_ids",
    "output_type",
    "fit_samples",
    "holdout_samples",
    "feature_dim_raw",
    "fit_seconds",
    "predict_seconds",
    "holdout_macro_f1",
    "holdout_minority_recall_mean",
    "warnings",
]

BASE_RESULT_FIELDNAMES = [
    "model_id",
    "group",
    "model_type",
    "combo_id",
    "feature_ids",
    "output_type",
    "feature_dim_raw",
    "trainval_oof_accuracy",
    "trainval_oof_macro_precision",
    "trainval_oof_macro_recall",
    "trainval_oof_macro_f1",
    "trainval_oof_weighted_f1",
    "trainval_oof_minority_recall_mean",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_minority_recall_mean",
    "fold_fit_seconds_total",
    "full_fit_seconds",
    "full_predict_seconds",
    "params",
    "warnings",
]

MODEL_RESULT_FIELDNAMES = [
    "model_id",
    "model_type",
    "base_ids",
    "num_base_models",
    "num_meta_features",
    "meta_model",
    "meta_C",
    "meta_weight_mode",
    "trainval_meta_accuracy",
    "trainval_meta_macro_precision",
    "trainval_meta_macro_recall",
    "trainval_meta_macro_f1",
    "trainval_meta_weighted_f1",
    "trainval_meta_minority_recall_mean",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_minority_recall_mean",
    "fit_seconds",
    "predict_seconds",
    "params",
    "warnings",
]

PER_CLASS_FIELDNAMES = [
    "model_id",
    "split",
    "class_name",
    "support",
    "precision",
    "recall",
    "f1",
]

CONFUSION_FIELDNAMES = [
    "model_id",
    "split",
    "true_label",
    "predicted_label",
    "count",
]

PREDICTION_FIELDNAMES = [
    "image_id",
    "path",
    "true_label",
    "predicted_label",
    "correct",
    "confidence",
    "true_label_probability",
    "top1_label",
    "top1_probability",
    "top2_label",
    "top2_probability",
    "top3_label",
    "top3_probability",
]

ERROR_PAIR_FIELDNAMES = [
    "true_label",
    "predicted_label",
    "count",
]

ERROR_EXAMPLE_FIELDNAMES = PREDICTION_FIELDNAMES


@dataclass
class BaseOutput:
    """Final base-model outputs."""

    summary: dict[str, Any]
    oof_meta: np.ndarray
    test_meta: np.ndarray


def log(message: str) -> None:
    """Print a flush-safe progress message."""
    print(message, flush=True)


def module_version(module_name: str) -> str:
    """Return an importable module version string."""
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


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


def warning_summary(caught: list[warnings.WarningMessage]) -> str:
    """Return compact warning text."""
    messages: list[str] = []
    for item in caught:
        message = str(item.message).replace("\n", " ")
        if message not in messages:
            messages.append(message)
    return " | ".join(messages[:5])


def write_dependency_versions() -> None:
    """Save dependency versions used by final evaluation."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": module_version("numpy"),
        "scikit_learn": module_version("sklearn"),
        "lightgbm": module_version("lightgbm"),
    }
    FINAL_DEPENDENCY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_DEPENDENCY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(versions, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one checkpoint row to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_feature_cache(path: Path) -> dict[str, Any]:
    """Load one feature cache."""
    cache = np.load(path, allow_pickle=False)
    return {
        "X_train": cache["X_train"],
        "y_train": cache["y_train"],
        "image_ids_train": cache["image_ids_train"],
        "paths_train": cache["paths_train"],
        "X_validation": cache["X_validation"],
        "y_validation": cache["y_validation"],
        "image_ids_validation": cache["image_ids_validation"],
        "paths_validation": cache["paths_validation"],
        "X_test": cache["X_test"],
        "y_test": cache["y_test"],
        "image_ids_test": cache["image_ids_test"],
        "paths_test": cache["paths_test"],
        "label_names": cache["label_names"].tolist(),
    }


def load_all_features() -> dict[str, dict[str, Any]]:
    """Load feature blocks and validate split alignment."""
    store: dict[str, dict[str, Any]] = {}
    for feature_id, path in FEATURE_CONFIGS.items():
        store[feature_id] = load_feature_cache(path)
        log(
            f"[load] {feature_id}: train {store[feature_id]['X_train'].shape}, "
            f"validation {store[feature_id]['X_validation'].shape}, "
            f"test {store[feature_id]['X_test'].shape}"
        )

    reference = store["F1_color"]
    for feature_id, data in store.items():
        for split in ("train", "validation", "test"):
            if not np.array_equal(reference[f"y_{split}"], data[f"y_{split}"]):
                raise ValueError(f"Label mismatch for {feature_id} / {split}")
            if not np.array_equal(reference[f"image_ids_{split}"], data[f"image_ids_{split}"]):
                raise ValueError(f"Image-id mismatch for {feature_id} / {split}")
        if reference["label_names"] != data["label_names"]:
            raise ValueError(f"Label-name mismatch for {feature_id}")
    log("[check] Feature split, image-id, and label alignment passed.")
    return store


def stratified_limited_indices(
    y: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    """Return all indices or a reproducible stratified subset."""
    indices = np.arange(len(y), dtype=np.int64)
    if max_samples is None or len(indices) <= max_samples:
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )
    return np.asarray(np.sort(selected), dtype=np.int64)


def reference_arrays(
    feature_store: dict[str, dict[str, Any]],
    trainval_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    """Return labels and metadata after optional subsampling."""
    reference = feature_store["F1_color"]
    y_trainval_full = np.concatenate([reference["y_train"], reference["y_validation"]])
    image_ids_trainval_full = np.concatenate(
        [reference["image_ids_train"], reference["image_ids_validation"]]
    )
    paths_trainval_full = np.concatenate([reference["paths_train"], reference["paths_validation"]])
    return {
        "y_trainval": y_trainval_full[trainval_indices],
        "image_ids_trainval": image_ids_trainval_full[trainval_indices],
        "paths_trainval": paths_trainval_full[trainval_indices],
        "y_test": reference["y_test"][test_indices],
        "image_ids_test": reference["image_ids_test"][test_indices],
        "paths_test": reference["paths_test"][test_indices],
        "label_names": reference["label_names"],
    }


def build_combo_matrix(
    feature_store: dict[str, dict[str, Any]],
    feature_ids: list[str],
    split: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Concatenate feature groups for trainval or test."""
    pieces = []
    for feature_id in feature_ids:
        data = feature_store[feature_id]
        if split == "trainval":
            matrix = np.vstack([data["X_train"], data["X_validation"]])
        elif split == "test":
            matrix = data["X_test"]
        else:
            raise ValueError(f"Unsupported split: {split}")
        pieces.append(matrix[indices])
    combo = pieces[0] if len(pieces) == 1 else np.hstack(pieces)
    if not np.isfinite(combo).all():
        raise ValueError(f"Non-finite values in {feature_ids} / {split}")
    return combo.astype(np.float32, copy=False)


def sample_weight_for_mode(y: np.ndarray, mode: str | None) -> np.ndarray | None:
    """Return sample weights for a class-imbalance mode."""
    if not mode or mode == "none":
        return None
    balanced = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    if mode == "balanced":
        return balanced
    if mode == "sqrt_balanced":
        weights = np.sqrt(balanced)
        return weights / np.mean(weights)
    raise ValueError(f"Unsupported sample_weight_mode: {mode}")


def make_estimator(config: dict[str, Any], num_classes: int, random_state: int) -> BaseEstimator:
    """Create one base estimator pipeline."""
    model_type = str(config["model_type"])
    params = dict(config.get("params", {}))
    if model_type == "logreg":
        model = LogisticRegression(
            solver="lbfgs",
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 700)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
        )
        return Pipeline(
            [("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", model)]
        )
    if model_type == "sgd_svm":
        model = SGDClassifier(
            loss="hinge",
            alpha=float(params.get("alpha", 0.0001)),
            max_iter=int(params.get("max_iter", 3000)),
            tol=float(params.get("tol", 0.001)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", model)]
        )
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 240)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("variance", VarianceThreshold(0.0)), ("model", model)])
    if model_type == "knn":
        model = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 11)),
            weights=params.get("weights", "distance"),
            p=int(params.get("p", 2)),
            n_jobs=-1,
        )
        return Pipeline(
            [("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", model)]
        )
    if model_type == "lightgbm":
        model = LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
            **params,
        )
        return Pipeline([("variance", VarianceThreshold(0.0)), ("model", model)])
    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_estimator(estimator: BaseEstimator, config: dict[str, Any], X: np.ndarray, y: np.ndarray) -> None:
    """Fit one estimator with optional sample weights."""
    weight = sample_weight_for_mode(y, config.get("sample_weight_mode"))
    if weight is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, model__sample_weight=weight)


def meta_output(estimator: BaseEstimator, X: np.ndarray, output_type: str) -> np.ndarray:
    """Return class-wise base outputs used as meta features."""
    if output_type == "proba":
        values = estimator.predict_proba(X)
    elif output_type == "decision":
        values = estimator.decision_function(X)
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = np.column_stack([-values, values]).astype(np.float32)
    return values


def labels_from_meta_features(values: np.ndarray) -> np.ndarray:
    """Convert class-wise meta values to hard predictions."""
    return np.asarray(np.argmax(values, axis=1), dtype=np.int64)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    prefix: str,
) -> dict[str, float]:
    """Compute aggregate metrics and selected minority recall."""
    labels = list(range(len(label_names)))
    metrics = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for label in MINORITY_LABELS:
        metrics[f"{prefix}recall_{label}"] = float(per_class_recall[label_names.index(label)])
    metrics[f"{prefix}minority_recall_mean"] = float(
        np.mean([metrics[f"{prefix}recall_{label}"] for label in MINORITY_LABELS])
    )
    return metrics


def rounded_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metric values for stable CSV output."""
    return {key: round(float(value), 10) for key, value in metrics.items()}


def per_class_rows(
    model_id: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Build per-class metric rows."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    return [
        {
            "model_id": model_id,
            "split": split,
            "class_name": label_names[index],
            "support": int(support[index]),
            "precision": round(float(precision[index]), 10),
            "recall": round(float(recall[index]), 10),
            "f1": round(float(f1[index]), 10),
        }
        for index in range(len(label_names))
    ]


def confusion_rows(
    model_id: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Build long-form confusion matrix rows."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    rows = []
    for true_index, true_label in enumerate(label_names):
        for pred_index, pred_label in enumerate(label_names):
            rows.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[true_index, pred_index]),
                }
            )
    return rows


def cache_is_complete(cache_path: Path, n_trainval: int, n_test: int, n_classes: int) -> bool:
    """Check whether a base cache can be reused."""
    if not cache_path.exists():
        return False
    try:
        cache = np.load(cache_path, allow_pickle=False)
        return (
            cache["oof_meta"].shape == (n_trainval, n_classes)
            and cache["test_meta"].shape == (n_test, n_classes)
            and np.isfinite(cache["oof_meta"]).all()
            and np.isfinite(cache["test_meta"]).all()
            and bool(np.all(cache["fold_done"]))
        )
    except (KeyError, OSError, ValueError):
        return False


def save_base_cache(
    cache_path: Path,
    oof_meta: np.ndarray,
    test_meta: np.ndarray,
    fold_done: np.ndarray,
) -> None:
    """Save base meta-feature cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        oof_meta=oof_meta.astype(np.float32, copy=False),
        test_meta=test_meta.astype(np.float32, copy=False),
        fold_done=fold_done.astype(bool, copy=False),
    )


def load_or_init_base_cache(
    cache_path: Path,
    n_trainval: int,
    n_test: int,
    n_classes: int,
    n_folds: int,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a partial cache or initialize empty meta arrays."""
    if resume and cache_path.exists():
        try:
            cache = np.load(cache_path, allow_pickle=False)
            oof = cache["oof_meta"]
            test = cache["test_meta"]
            fold_done = cache["fold_done"]
            if (
                oof.shape == (n_trainval, n_classes)
                and test.shape == (n_test, n_classes)
                and fold_done.shape == (n_folds,)
            ):
                return oof.astype(np.float32), test.astype(np.float32), fold_done.astype(bool)
        except (KeyError, OSError, ValueError):
            pass
    return (
        np.full((n_trainval, n_classes), np.nan, dtype=np.float32),
        np.full((n_test, n_classes), np.nan, dtype=np.float32),
        np.zeros(n_folds, dtype=bool),
    )


def run_base_model(
    config: dict[str, Any],
    feature_store: dict[str, dict[str, Any]],
    y_trainval: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    trainval_indices: np.ndarray,
    test_indices: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    resume: bool,
) -> BaseOutput:
    """Build trainval OOF and test meta features for one base model."""
    base_id = str(config["base_id"])
    combo_id = str(config["combo_id"])
    feature_ids = list(COMBO_CONFIGS[combo_id])
    output_type = str(config["output_type"])
    X_trainval = build_combo_matrix(feature_store, feature_ids, "trainval", trainval_indices)
    X_test = build_combo_matrix(feature_store, feature_ids, "test", test_indices)
    n_classes = len(label_names)
    cache_path = FINAL_CACHE_DIR / f"{base_id}.npz"

    if resume and cache_is_complete(cache_path, len(y_trainval), len(y_test), n_classes):
        log(f"[base skip] {base_id}: complete final cache found.")
        cache = np.load(cache_path, allow_pickle=False)
        oof_meta = cache["oof_meta"].astype(np.float32)
        test_meta = cache["test_meta"].astype(np.float32)
        summary = summarize_base(
            config,
            feature_ids,
            X_trainval.shape[1],
            y_trainval,
            y_test,
            label_names,
            oof_meta,
            test_meta,
            0.0,
            0.0,
            0.0,
            "loaded from cache",
        )
        return BaseOutput(summary=summary, oof_meta=oof_meta, test_meta=test_meta)

    oof_meta, test_meta, fold_done = load_or_init_base_cache(
        cache_path,
        len(y_trainval),
        len(y_test),
        n_classes,
        len(folds),
        resume,
    )
    fold_fit_seconds_total = 0.0
    all_warnings: list[str] = []

    for fold_index, (fit_idx, holdout_idx) in enumerate(folds):
        if resume and fold_done[fold_index] and np.isfinite(oof_meta[holdout_idx]).all():
            log(f"[base fold skip] {base_id} fold {fold_index + 1}/{len(folds)}")
            continue
        estimator = make_estimator(config, n_classes, RANDOM_SEED + fold_index)
        log(
            f"[base fold] {base_id} fold {fold_index + 1}/{len(folds)} | "
            f"fit={len(fit_idx)} holdout={len(holdout_idx)} raw_dim={X_trainval.shape[1]}"
        )
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_estimator(estimator, config, X_trainval[fit_idx], y_trainval[fit_idx])
        fit_seconds = time.perf_counter() - fit_start
        fold_fit_seconds_total += fit_seconds
        warn_text = warning_summary(caught)
        if warn_text:
            all_warnings.append(warn_text)

        predict_start = time.perf_counter()
        fold_meta = meta_output(estimator, X_trainval[holdout_idx], output_type)
        predict_seconds = time.perf_counter() - predict_start
        oof_meta[holdout_idx] = fold_meta
        fold_done[fold_index] = True
        save_base_cache(cache_path, oof_meta, test_meta, fold_done)

        fold_pred = labels_from_meta_features(fold_meta)
        fold_metrics = evaluate_predictions(
            y_trainval[holdout_idx],
            fold_pred,
            label_names,
            prefix="holdout_",
        )
        append_csv_row(
            FINAL_BASE_FOLDS_CSV,
            {
                "base_id": base_id,
                "fold_index": fold_index + 1,
                "combo_id": combo_id,
                "feature_ids": "+".join(feature_ids),
                "output_type": output_type,
                "fit_samples": len(fit_idx),
                "holdout_samples": len(holdout_idx),
                "feature_dim_raw": X_trainval.shape[1],
                "fit_seconds": round(float(fit_seconds), 4),
                "predict_seconds": round(float(predict_seconds), 4),
                "holdout_macro_f1": round(fold_metrics["holdout_macro_f1"], 10),
                "holdout_minority_recall_mean": round(
                    fold_metrics["holdout_minority_recall_mean"],
                    10,
                ),
                "warnings": warn_text,
            },
            BASE_FOLD_FIELDNAMES,
        )
        log(
            f"[base fold done] {base_id} fold {fold_index + 1} | "
            f"macro-F1={fold_metrics['holdout_macro_f1']:.4f} | "
            f"time={fit_seconds + predict_seconds:.1f}s"
        )

    if not np.isfinite(oof_meta).all():
        raise ValueError(f"OOF meta features are incomplete for {base_id}")

    log(f"[base full] {base_id}: refit on train+validation and predict test.")
    estimator = make_estimator(config, n_classes, RANDOM_SEED)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_estimator(estimator, config, X_trainval, y_trainval)
    full_fit_seconds = time.perf_counter() - fit_start
    warn_text = warning_summary(caught)
    if warn_text:
        all_warnings.append(warn_text)

    predict_start = time.perf_counter()
    test_meta = meta_output(estimator, X_test, output_type)
    full_predict_seconds = time.perf_counter() - predict_start
    save_base_cache(cache_path, oof_meta, test_meta, np.ones(len(folds), dtype=bool))

    summary = summarize_base(
        config,
        feature_ids,
        X_trainval.shape[1],
        y_trainval,
        y_test,
        label_names,
        oof_meta,
        test_meta,
        fold_fit_seconds_total,
        full_fit_seconds,
        full_predict_seconds,
        " | ".join(dict.fromkeys(all_warnings)),
    )
    log(
        f"[base done] {base_id} | test macro-F1={summary['test_macro_f1']:.4f} | "
        f"minority={summary['test_minority_recall_mean']:.4f}"
    )
    return BaseOutput(summary=summary, oof_meta=oof_meta, test_meta=test_meta)


def summarize_base(
    config: dict[str, Any],
    feature_ids: list[str],
    feature_dim_raw: int,
    y_trainval: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    oof_meta: np.ndarray,
    test_meta: np.ndarray,
    fold_fit_seconds_total: float,
    full_fit_seconds: float,
    full_predict_seconds: float,
    warning_text: str,
) -> dict[str, Any]:
    """Create one base-model final summary row."""
    trainval_pred = labels_from_meta_features(oof_meta)
    test_pred = labels_from_meta_features(test_meta)
    row: dict[str, Any] = {
        "model_id": config["base_id"],
        "group": config["group"],
        "model_type": config["model_type"],
        "combo_id": config["combo_id"],
        "feature_ids": "+".join(feature_ids),
        "output_type": config["output_type"],
        "feature_dim_raw": feature_dim_raw,
        "fold_fit_seconds_total": round(float(fold_fit_seconds_total), 4),
        "full_fit_seconds": round(float(full_fit_seconds), 4),
        "full_predict_seconds": round(float(full_predict_seconds), 4),
        "params": params_text(config),
        "warnings": warning_text,
    }
    row.update(
        rounded_metrics(
            evaluate_predictions(y_trainval, trainval_pred, label_names, prefix="trainval_oof_")
        )
    )
    row.update(rounded_metrics(evaluate_predictions(y_test, test_pred, label_names, prefix="test_")))
    return row


def make_meta_estimator(C: float, weight_mode: str | None, random_state: int) -> Pipeline:
    """Create logistic stacking meta learner."""
    class_weight = "balanced" if weight_mode == "balanced_class_weight" else None
    model = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def fit_meta_estimator(estimator: Pipeline, X: np.ndarray, y: np.ndarray, weight_mode: str | None) -> None:
    """Fit meta estimator with optional sample weights."""
    if weight_mode == "sqrt_sample_weight":
        estimator.fit(X, y, model__sample_weight=sample_weight_for_mode(y, "sqrt_balanced"))
    else:
        estimator.fit(X, y)


def evaluate_stacking_model(
    model_id: str,
    base_ids: list[str],
    base_outputs: dict[str, BaseOutput],
    y_trainval: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    meta_C: float,
    meta_weight_mode: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Train fixed logistic meta learner and evaluate on test."""
    X_meta_trainval = np.hstack([base_outputs[base_id].oof_meta for base_id in base_ids]).astype(np.float32)
    X_meta_test = np.hstack([base_outputs[base_id].test_meta for base_id in base_ids]).astype(np.float32)
    estimator = make_meta_estimator(meta_C, meta_weight_mode, RANDOM_SEED)

    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_meta_estimator(estimator, X_meta_trainval, y_trainval, meta_weight_mode)
    fit_seconds = time.perf_counter() - fit_start

    trainval_pred = np.asarray(estimator.predict(X_meta_trainval), dtype=np.int64)
    predict_start = time.perf_counter()
    test_pred = np.asarray(estimator.predict(X_meta_test), dtype=np.int64)
    test_proba = np.asarray(estimator.predict_proba(X_meta_test), dtype=np.float32)
    predict_seconds = time.perf_counter() - predict_start

    row: dict[str, Any] = {
        "model_id": model_id,
        "model_type": "stacking",
        "base_ids": "+".join(base_ids),
        "num_base_models": len(base_ids),
        "num_meta_features": X_meta_trainval.shape[1],
        "meta_model": "LogisticRegression",
        "meta_C": meta_C,
        "meta_weight_mode": meta_weight_mode,
        "fit_seconds": round(float(fit_seconds), 4),
        "predict_seconds": round(float(predict_seconds), 4),
        "params": params_text({"C": meta_C, "weight_mode": meta_weight_mode}),
        "warnings": warning_summary(caught),
    }
    row.update(
        rounded_metrics(
            evaluate_predictions(
                y_trainval,
                trainval_pred,
                label_names,
                prefix="trainval_meta_",
            )
        )
    )
    row.update(rounded_metrics(evaluate_predictions(y_test, test_pred, label_names, prefix="test_")))
    return row, trainval_pred, test_pred, test_proba


def evaluate_soft_average(
    model_id: str,
    base_ids: list[str],
    base_outputs: dict[str, BaseOutput],
    y_trainval: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate simple average of probability base outputs."""
    trainval_values = np.mean([base_outputs[base_id].oof_meta for base_id in base_ids], axis=0)
    test_values = np.mean([base_outputs[base_id].test_meta for base_id in base_ids], axis=0)
    trainval_pred = labels_from_meta_features(trainval_values)
    test_pred = labels_from_meta_features(test_values)
    row: dict[str, Any] = {
        "model_id": model_id,
        "model_type": "soft_average",
        "base_ids": "+".join(base_ids),
        "num_base_models": len(base_ids),
        "num_meta_features": len(base_ids) * len(label_names),
        "meta_model": "",
        "meta_C": "",
        "meta_weight_mode": "",
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "params": params_text({"method": "unweighted mean of class probabilities"}),
        "warnings": "",
    }
    row.update(
        rounded_metrics(
            evaluate_predictions(y_trainval, trainval_pred, label_names, prefix="trainval_meta_")
        )
    )
    row.update(rounded_metrics(evaluate_predictions(y_test, test_pred, label_names, prefix="test_")))
    return row, trainval_pred, test_pred, test_values


def prediction_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    image_ids: np.ndarray,
    paths: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Build detailed prediction rows for the final model."""
    rows = []
    for index in range(len(y_true)):
        ranking = np.argsort(proba[index])[::-1][:3]
        true_index = int(y_true[index])
        pred_index = int(y_pred[index])
        row = {
            "image_id": str(image_ids[index]),
            "path": str(paths[index]),
            "true_label": label_names[true_index],
            "predicted_label": label_names[pred_index],
            "correct": bool(true_index == pred_index),
            "confidence": round(float(proba[index, pred_index]), 10),
            "true_label_probability": round(float(proba[index, true_index]), 10),
        }
        for rank, label_index in enumerate(ranking, start=1):
            row[f"top{rank}_label"] = label_names[int(label_index)]
            row[f"top{rank}_probability"] = round(float(proba[index, label_index]), 10)
        rows.append(row)
    return rows


def error_pair_rows(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> list[dict[str, Any]]:
    """Summarize final-model error pairs."""
    counts: dict[tuple[str, str], int] = {}
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        if int(true_value) == int(pred_value):
            continue
        key = (label_names[int(true_value)], label_names[int(pred_value)])
        counts[key] = counts.get(key, 0) + 1
    rows = [
        {"true_label": true_label, "predicted_label": pred_label, "count": count}
        for (true_label, pred_label), count in counts.items()
    ]
    return sorted(rows, key=lambda row: int(row["count"]), reverse=True)


def prepare_outputs(force: bool, resume: bool) -> None:
    """Prepare output files and optional cache reset."""
    if force and resume:
        raise ValueError("--force and --resume should not be used together.")
    FINAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        FINAL_BASE_FOLDS_CSV,
        FINAL_BASE_RESULTS_CSV,
        FINAL_MODEL_RESULTS_CSV,
        FINAL_PER_CLASS_CSV,
        FINAL_CONFUSION_CSV,
        FINAL_PREDICTIONS_CSV,
        FINAL_ERROR_PAIRS_CSV,
        FINAL_ERROR_EXAMPLES_CSV,
        FINAL_SUMMARY_JSON,
    ]
    if force:
        for path in outputs:
            if path.exists():
                path.unlink()
        for cache_path in FINAL_CACHE_DIR.glob("*.npz"):
            cache_path.unlink()
    if not resume:
        write_csv(FINAL_BASE_FOLDS_CSV, [], BASE_FOLD_FIELDNAMES)
        write_csv(FINAL_BASE_RESULTS_CSV, [], BASE_RESULT_FIELDNAMES)
        write_csv(FINAL_MODEL_RESULTS_CSV, [], MODEL_RESULT_FIELDNAMES)
        write_csv(FINAL_PER_CLASS_CSV, [], PER_CLASS_FIELDNAMES)
        write_csv(FINAL_CONFUSION_CSV, [], CONFUSION_FIELDNAMES)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run final fixed-candidate test evaluation.")
    parser.add_argument("--n-folds", type=int, default=5, help="OOF folds on train+validation.")
    parser.add_argument("--max-trainval-samples", type=int, default=None, help="Stratified trainval subsample for smoke tests.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Stratified test subsample for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed final base caches.")
    parser.add_argument("--force", action="store_true", help="Overwrite final outputs and caches.")
    return parser.parse_args()


def main() -> None:
    """Run final evaluation."""
    args = parse_args()
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    prepare_outputs(force=args.force, resume=args.resume)
    write_dependency_versions()

    feature_store = load_all_features()
    reference = feature_store["F1_color"]
    full_y_trainval = np.concatenate([reference["y_train"], reference["y_validation"]])
    trainval_indices = stratified_limited_indices(
        full_y_trainval,
        args.max_trainval_samples,
        RANDOM_SEED + 10,
    )
    test_indices = stratified_limited_indices(
        reference["y_test"],
        args.max_test_samples,
        RANDOM_SEED + 11,
    )
    arrays = reference_arrays(feature_store, trainval_indices, test_indices)
    y_trainval = arrays["y_trainval"]
    y_test = arrays["y_test"]
    label_names = arrays["label_names"]
    folds = list(
        StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_SEED).split(
            np.zeros(len(y_trainval)),
            y_trainval,
        )
    )

    log(
        f"[plan] Final test evaluation | trainval={len(y_trainval)} | "
        f"test={len(y_test)} | folds={args.n_folds} | bases={len(BASE_CONFIGS)}"
    )
    estimated = "60-90 minutes"
    if args.max_trainval_samples is not None or args.max_test_samples is not None:
        estimated = "2-8 minutes"
    log(
        f"[estimate] Expected wall time: {estimated}. "
        "Each base fold and final refit writes checkpoints."
    )

    base_outputs: dict[str, BaseOutput] = {}
    base_rows: list[dict[str, Any]] = []
    for index, config in enumerate(BASE_CONFIGS, start=1):
        log(f"[base {index}/{len(BASE_CONFIGS)}] {config['base_id']}")
        output = run_base_model(
            config,
            feature_store,
            y_trainval,
            y_test,
            label_names,
            trainval_indices,
            test_indices,
            folds,
            resume=args.resume,
        )
        base_outputs[str(config["base_id"])] = output
        base_rows.append(output.summary)
    write_csv(FINAL_BASE_RESULTS_CSV, base_rows, BASE_RESULT_FIELDNAMES)

    model_rows: list[dict[str, Any]] = []
    per_class_all: list[dict[str, Any]] = []
    confusion_all: list[dict[str, Any]] = []

    for base_id, output in base_outputs.items():
        test_pred = labels_from_meta_features(output.test_meta)
        per_class_all.extend(per_class_rows(base_id, "test", y_test, test_pred, label_names))
        confusion_all.extend(confusion_rows(base_id, "test", y_test, test_pred, label_names))

    final_row, final_trainval_pred, final_test_pred, final_test_proba = evaluate_stacking_model(
        "final_stack_small_plus_lgbm_anchor",
        FINAL_STACK_BASE_IDS,
        base_outputs,
        y_trainval,
        y_test,
        label_names,
        meta_C=0.1,
        meta_weight_mode="none",
    )
    model_rows.append(final_row)
    per_class_all.extend(
        per_class_rows(
            "final_stack_small_plus_lgbm_anchor",
            "test",
            y_test,
            final_test_pred,
            label_names,
        )
    )
    confusion_all.extend(
        confusion_rows(
            "final_stack_small_plus_lgbm_anchor",
            "test",
            y_test,
            final_test_pred,
            label_names,
        )
    )

    small_row, _, small_test_pred, _ = evaluate_stacking_model(
        "final_stack_small_only",
        SMALL_STACK_BASE_IDS,
        base_outputs,
        y_trainval,
        y_test,
        label_names,
        meta_C=0.1,
        meta_weight_mode="sqrt_sample_weight",
    )
    model_rows.append(small_row)
    per_class_all.extend(per_class_rows("final_stack_small_only", "test", y_test, small_test_pred, label_names))
    confusion_all.extend(confusion_rows("final_stack_small_only", "test", y_test, small_test_pred, label_names))

    avg_anchor_row, _, avg_anchor_pred, _ = evaluate_soft_average(
        "final_avg_small_plus_lgbm_anchor_proba",
        PROBA_WITH_ANCHOR_BASE_IDS,
        base_outputs,
        y_trainval,
        y_test,
        label_names,
    )
    model_rows.append(avg_anchor_row)
    per_class_all.extend(
        per_class_rows(
            "final_avg_small_plus_lgbm_anchor_proba",
            "test",
            y_test,
            avg_anchor_pred,
            label_names,
        )
    )
    confusion_all.extend(
        confusion_rows(
            "final_avg_small_plus_lgbm_anchor_proba",
            "test",
            y_test,
            avg_anchor_pred,
            label_names,
        )
    )

    avg_small_row, _, avg_small_pred, _ = evaluate_soft_average(
        "final_avg_small_proba",
        PROBA_SMALL_BASE_IDS,
        base_outputs,
        y_trainval,
        y_test,
        label_names,
    )
    model_rows.append(avg_small_row)
    per_class_all.extend(per_class_rows("final_avg_small_proba", "test", y_test, avg_small_pred, label_names))
    confusion_all.extend(confusion_rows("final_avg_small_proba", "test", y_test, avg_small_pred, label_names))

    model_rows = sorted(model_rows, key=lambda row: float(row["test_macro_f1"]), reverse=True)
    write_csv(FINAL_MODEL_RESULTS_CSV, model_rows, MODEL_RESULT_FIELDNAMES)
    write_csv(FINAL_PER_CLASS_CSV, per_class_all, PER_CLASS_FIELDNAMES)
    write_csv(FINAL_CONFUSION_CSV, confusion_all, CONFUSION_FIELDNAMES)

    pred_rows = prediction_rows(
        y_test,
        final_test_pred,
        final_test_proba,
        arrays["image_ids_test"],
        arrays["paths_test"],
        label_names,
    )
    write_csv(FINAL_PREDICTIONS_CSV, pred_rows, PREDICTION_FIELDNAMES)
    error_pairs = error_pair_rows(y_test, final_test_pred, label_names)
    write_csv(FINAL_ERROR_PAIRS_CSV, error_pairs, ERROR_PAIR_FIELDNAMES)
    error_examples = [
        row for row in sorted(pred_rows, key=lambda item: float(item["confidence"]), reverse=True)
        if not row["correct"]
    ][:80]
    write_csv(FINAL_ERROR_EXAMPLES_CSV, error_examples, ERROR_EXAMPLE_FIELDNAMES)

    summary = {
        "protocol": "Fixed final model selected on train/validation; test used once for final evaluation.",
        "trainval_samples": len(y_trainval),
        "test_samples": len(y_test),
        "n_folds": args.n_folds,
        "best_final_model": model_rows[0],
        "base_results": base_rows,
        "model_results": model_rows,
        "top_error_pairs": error_pairs[:12],
    }
    with FINAL_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, ensure_ascii=False, indent=2)

    best = model_rows[0]
    log(
        f"[complete] Best final test model: {best['model_id']} | "
        f"macro-F1={float(best['test_macro_f1']):.4f} | "
        f"minority={float(best['test_minority_recall_mean']):.4f}"
    )


if __name__ == "__main__":
    main()
