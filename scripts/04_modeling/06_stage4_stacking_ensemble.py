"""Stage 4: stacking ensemble with small and diverse base learners.

Notebook block purpose:
    Stage 4 stacking ensemble. This block builds out-of-fold meta features
    from small/diverse traditional learners, optionally adds the current
    LightGBM C11 sqrt-balanced anchor, trains a lightweight logistic meta
    learner using train-only cross-validation, and evaluates once on the
    validation split. The test split is intentionally not used.

Design notes:
    1. Base learners are fitted by stratified OOF folds on the train split.
    2. Validation predictions are produced only after refitting each base
       learner on the full train split.
    3. Every base fold writes a checkpoint row and cache file for resume.
    4. Meta-learner hyperparameters are selected by CV on OOF meta features.
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
    f1_score,
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
STAGE4_BEST_MACRO_F1 = 0.8131372761
STAGE4_BEST_MINORITY_RECALL = 0.8157689034

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"

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
        "description": "Balanced multinomial logistic regression on C11 full.",
        "params": {"C": 0.8, "max_iter": 700, "class_weight": "balanced"},
    },
    {
        "base_id": "sgd_svm_hct_balanced",
        "group": "small",
        "model_type": "sgd_svm",
        "combo_id": "hog_color_texture",
        "output_type": "decision",
        "description": "Linear-kernel SVM via SGD on HOG+Color+Texture.",
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
        "description": "Small random forest on low-dimensional Color+Texture+GIST.",
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
        "description": "Distance-based local model on compact GIST+Texture+Shape.",
        "params": {"n_neighbors": 11, "weights": "distance", "p": 2},
    },
    {
        "base_id": "lgbm_small_hct_sqrt",
        "group": "small",
        "model_type": "lightgbm",
        "combo_id": "hog_color_texture",
        "output_type": "proba",
        "description": "Compact LightGBM on HOG+Color+Texture with sqrt-balanced weights.",
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
        "description": "Current Stage 4 best LightGBM C11 sqrt-balanced anchor.",
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

BASE_FOLD_FIELDNAMES = [
    "base_id",
    "fold_index",
    "combo_id",
    "feature_ids",
    "output_type",
    "train_samples",
    "holdout_samples",
    "feature_dim_raw",
    "fit_seconds",
    "predict_seconds",
    "holdout_macro_f1",
    "holdout_minority_recall_mean",
    "params",
    "warnings",
]

BASE_SUMMARY_FIELDNAMES = [
    "base_id",
    "group",
    "model_type",
    "combo_id",
    "feature_ids",
    "output_type",
    "feature_dim_raw",
    "oof_accuracy",
    "oof_macro_precision",
    "oof_macro_recall",
    "oof_macro_f1",
    "oof_weighted_f1",
    "oof_minority_recall_mean",
    "validation_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "validation_weighted_f1",
    "validation_minority_recall_mean",
    "delta_macro_f1_vs_stage4_best",
    "delta_minority_recall_vs_stage4_best",
    "fold_fit_seconds_total",
    "full_fit_seconds",
    "full_predict_seconds",
    "params",
    "warnings",
    "validation_recall_battery",
    "validation_f1_battery",
    "validation_recall_biological",
    "validation_f1_biological",
    "validation_recall_trash",
    "validation_f1_trash",
]

META_SEARCH_FIELDNAMES = [
    "ensemble_id",
    "ensemble_type",
    "base_ids",
    "num_base_models",
    "num_meta_features",
    "meta_model",
    "meta_C",
    "meta_weight_mode",
    "inner_cv_macro_f1",
    "inner_cv_minority_recall_mean",
    "inner_cv_weighted_f1",
    "fit_seconds_total",
    "params",
    "warnings",
]

ENSEMBLE_FIELDNAMES = [
    "ensemble_id",
    "ensemble_type",
    "base_ids",
    "num_base_models",
    "num_meta_features",
    "meta_model",
    "meta_C",
    "meta_weight_mode",
    "inner_cv_macro_f1",
    "inner_cv_minority_recall_mean",
    "validation_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "validation_weighted_f1",
    "validation_minority_recall_mean",
    "delta_macro_f1_vs_stage4_best",
    "delta_minority_recall_vs_stage4_best",
    "fit_seconds",
    "predict_seconds",
    "params",
    "warnings",
    "validation_recall_battery",
    "validation_f1_battery",
    "validation_recall_biological",
    "validation_f1_biological",
    "validation_recall_trash",
    "validation_f1_trash",
]


@dataclass
class ReportPaths:
    """Output paths for one run tag."""

    base_folds_csv: Path
    base_summary_csv: Path
    meta_search_csv: Path
    ensemble_results_csv: Path
    best_csv: Path
    summary_json: Path
    dependency_json: Path
    cache_dir: Path


def log(message: str) -> None:
    """Print a flush-safe progress message for long runs."""
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


def tagged_name(stem: str, run_tag: str, suffix: str) -> str:
    """Build stable output names while keeping the full run clean."""
    if run_tag == "full":
        return f"{stem}{suffix}"
    return f"{stem}_{run_tag}{suffix}"


def build_report_paths(run_tag: str) -> ReportPaths:
    """Build all report paths for one run tag."""
    cache_name = "stage4_stacking_cache" if run_tag == "full" else f"stage4_stacking_{run_tag}_cache"
    return ReportPaths(
        base_folds_csv=REPORT_DIR / tagged_name("stage4_stacking_base_folds", run_tag, ".csv"),
        base_summary_csv=REPORT_DIR / tagged_name("stage4_stacking_base_summary", run_tag, ".csv"),
        meta_search_csv=REPORT_DIR / tagged_name("stage4_stacking_meta_search", run_tag, ".csv"),
        ensemble_results_csv=REPORT_DIR / tagged_name("stage4_stacking_results", run_tag, ".csv"),
        best_csv=REPORT_DIR / tagged_name("stage4_stacking_best", run_tag, ".csv"),
        summary_json=REPORT_DIR / tagged_name("stage4_stacking_summary", run_tag, ".json"),
        dependency_json=REPORT_DIR / tagged_name("stage4_stacking_dependency_versions", run_tag, ".json"),
        cache_dir=REPORT_DIR / cache_name,
    )


def write_dependency_versions(path: Path) -> None:
    """Save dependency versions used by Stage 4 stacking."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": module_version("numpy"),
        "scikit_learn": module_version("sklearn"),
        "lightgbm": module_version("lightgbm"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(versions, handle, ensure_ascii=False, indent=2)


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
    """Append one checkpoint row to a CSV file."""
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
    """Load all feature blocks and validate split alignment."""
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


def build_combo_matrix(
    feature_store: dict[str, dict[str, Any]],
    feature_ids: list[str],
    split: str,
    indices: np.ndarray | None,
) -> np.ndarray:
    """Concatenate feature groups for one split and optional subset."""
    pieces = []
    for feature_id in feature_ids:
        matrix = feature_store[feature_id][f"X_{split}"]
        if indices is not None:
            matrix = matrix[indices]
        pieces.append(matrix)
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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    prefix: str = "",
) -> dict[str, float]:
    """Compute macro metrics and selected minority-class metrics."""
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
    for label in MINORITY_LABELS:
        label_index = label_names.index(label)
        metrics[f"{prefix}recall_{label}"] = float(per_class_recall[label_index])
        metrics[f"{prefix}f1_{label}"] = float(per_class_f1[label_index])
    metrics[f"{prefix}minority_recall_mean"] = float(
        np.mean([metrics[f"{prefix}recall_{label}"] for label in MINORITY_LABELS])
    )
    return metrics


def rounded_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metric values for stable CSV output."""
    return {key: round(float(value), 10) for key, value in metrics.items()}


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
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
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
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
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
        return Pipeline(
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("model", model),
            ]
        )
    if model_type == "knn":
        model = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 11)),
            weights=params.get("weights", "distance"),
            p=int(params.get("p", 2)),
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
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
        return Pipeline(
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("model", model),
            ]
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_estimator(
    estimator: BaseEstimator,
    config: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> None:
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


def cache_is_complete(
    cache_path: Path,
    n_train: int,
    n_validation: int,
    n_classes: int,
) -> bool:
    """Check whether a base cache can be reused."""
    if not cache_path.exists():
        return False
    try:
        cache = np.load(cache_path, allow_pickle=False)
        oof = cache["oof_meta"]
        validation = cache["validation_meta"]
        fold_done = cache["fold_done"]
        return (
            oof.shape == (n_train, n_classes)
            and validation.shape == (n_validation, n_classes)
            and bool(np.all(fold_done))
            and np.isfinite(oof).all()
            and np.isfinite(validation).all()
        )
    except (KeyError, OSError, ValueError):
        return False


def save_base_cache(
    cache_path: Path,
    oof_meta: np.ndarray,
    validation_meta: np.ndarray,
    fold_done: np.ndarray,
) -> None:
    """Save base meta-feature cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        oof_meta=oof_meta.astype(np.float32, copy=False),
        validation_meta=validation_meta.astype(np.float32, copy=False),
        fold_done=fold_done.astype(bool, copy=False),
    )


def load_or_init_base_cache(
    cache_path: Path,
    n_train: int,
    n_validation: int,
    n_classes: int,
    n_folds: int,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a partial cache or initialize empty meta arrays."""
    if resume and cache_path.exists():
        try:
            cache = np.load(cache_path, allow_pickle=False)
            oof = cache["oof_meta"]
            validation = cache["validation_meta"]
            fold_done = cache["fold_done"]
            if oof.shape == (n_train, n_classes) and validation.shape == (n_validation, n_classes):
                if fold_done.shape == (n_folds,):
                    return oof.astype(np.float32), validation.astype(np.float32), fold_done.astype(bool)
        except (KeyError, OSError, ValueError):
            pass
    oof_meta = np.full((n_train, n_classes), np.nan, dtype=np.float32)
    validation_meta = np.full((n_validation, n_classes), np.nan, dtype=np.float32)
    fold_done = np.zeros(n_folds, dtype=bool)
    return oof_meta, validation_meta, fold_done


def run_base_model(
    config: dict[str, Any],
    feature_store: dict[str, dict[str, Any]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    label_names: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    paths: ReportPaths,
    resume: bool,
    refresh_validation: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build OOF and validation meta features for one base model."""
    base_id = str(config["base_id"])
    combo_id = str(config["combo_id"])
    feature_ids = list(COMBO_CONFIGS[combo_id])
    output_type = str(config["output_type"])
    X_train = build_combo_matrix(feature_store, feature_ids, "train", train_indices)
    X_validation = build_combo_matrix(feature_store, feature_ids, "validation", validation_indices)
    n_classes = len(label_names)
    cache_path = paths.cache_dir / f"{base_id}.npz"

    if (
        resume
        and not refresh_validation
        and cache_is_complete(cache_path, len(y_train), len(y_validation), n_classes)
    ):
        log(f"[base skip] {base_id}: complete cache found.")
        cache = np.load(cache_path, allow_pickle=False)
        oof_meta = cache["oof_meta"].astype(np.float32)
        validation_meta = cache["validation_meta"].astype(np.float32)
        summary = summarize_base(
            config,
            feature_ids,
            X_train.shape[1],
            y_train,
            y_validation,
            label_names,
            oof_meta,
            validation_meta,
            0.0,
            0.0,
            0.0,
            "loaded from cache",
        )
        return summary, oof_meta, validation_meta

    oof_meta, validation_meta, fold_done = load_or_init_base_cache(
        cache_path,
        len(y_train),
        len(y_validation),
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
            f"train={len(fit_idx)} holdout={len(holdout_idx)} raw_dim={X_train.shape[1]}"
        )
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_estimator(estimator, config, X_train[fit_idx], y_train[fit_idx])
        fit_seconds = time.perf_counter() - fit_start
        fold_fit_seconds_total += fit_seconds
        warn_text = warning_summary(caught)
        if warn_text:
            all_warnings.append(warn_text)

        predict_start = time.perf_counter()
        fold_meta = meta_output(estimator, X_train[holdout_idx], output_type)
        predict_seconds = time.perf_counter() - predict_start
        oof_meta[holdout_idx] = fold_meta
        fold_done[fold_index] = True
        save_base_cache(cache_path, oof_meta, validation_meta, fold_done)

        fold_pred = labels_from_meta_features(fold_meta)
        fold_metrics = evaluate_predictions(y_train[holdout_idx], fold_pred, label_names, prefix="holdout_")
        append_csv_row(
            paths.base_folds_csv,
            {
                "base_id": base_id,
                "fold_index": fold_index + 1,
                "combo_id": combo_id,
                "feature_ids": "+".join(feature_ids),
                "output_type": output_type,
                "train_samples": len(fit_idx),
                "holdout_samples": len(holdout_idx),
                "feature_dim_raw": X_train.shape[1],
                "fit_seconds": round(float(fit_seconds), 4),
                "predict_seconds": round(float(predict_seconds), 4),
                "holdout_macro_f1": round(fold_metrics["holdout_macro_f1"], 10),
                "holdout_minority_recall_mean": round(
                    fold_metrics["holdout_minority_recall_mean"],
                    10,
                ),
                "params": params_text(config),
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

    log(f"[base full] {base_id}: refit on full train and predict validation.")
    estimator = make_estimator(config, n_classes, RANDOM_SEED)
    full_fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_estimator(estimator, config, X_train, y_train)
    full_fit_seconds = time.perf_counter() - full_fit_start
    warn_text = warning_summary(caught)
    if warn_text:
        all_warnings.append(warn_text)

    full_predict_start = time.perf_counter()
    validation_meta = meta_output(estimator, X_validation, output_type)
    full_predict_seconds = time.perf_counter() - full_predict_start
    save_base_cache(cache_path, oof_meta, validation_meta, np.ones(len(folds), dtype=bool))

    summary = summarize_base(
        config,
        feature_ids,
        X_train.shape[1],
        y_train,
        y_validation,
        label_names,
        oof_meta,
        validation_meta,
        fold_fit_seconds_total,
        full_fit_seconds,
        full_predict_seconds,
        " | ".join(dict.fromkeys(all_warnings)),
    )
    log(
        f"[base done] {base_id} | validation macro-F1="
        f"{summary['validation_macro_f1']:.4f} | minority="
        f"{summary['validation_minority_recall_mean']:.4f}"
    )
    return summary, oof_meta, validation_meta


def summarize_base(
    config: dict[str, Any],
    feature_ids: list[str],
    feature_dim_raw: int,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    label_names: list[str],
    oof_meta: np.ndarray,
    validation_meta: np.ndarray,
    fold_fit_seconds_total: float,
    full_fit_seconds: float,
    full_predict_seconds: float,
    warning_text: str,
) -> dict[str, Any]:
    """Create one base-model summary row."""
    oof_pred = labels_from_meta_features(oof_meta)
    validation_pred = labels_from_meta_features(validation_meta)
    oof_metrics = rounded_metrics(evaluate_predictions(y_train, oof_pred, label_names, prefix="oof_"))
    validation_metrics = rounded_metrics(
        evaluate_predictions(y_validation, validation_pred, label_names, prefix="validation_")
    )
    row: dict[str, Any] = {
        "base_id": config["base_id"],
        "group": config["group"],
        "model_type": config["model_type"],
        "combo_id": config["combo_id"],
        "feature_ids": "+".join(feature_ids),
        "output_type": config["output_type"],
        "feature_dim_raw": feature_dim_raw,
        "delta_macro_f1_vs_stage4_best": round(
            validation_metrics["validation_macro_f1"] - STAGE4_BEST_MACRO_F1,
            10,
        ),
        "delta_minority_recall_vs_stage4_best": round(
            validation_metrics["validation_minority_recall_mean"] - STAGE4_BEST_MINORITY_RECALL,
            10,
        ),
        "fold_fit_seconds_total": round(float(fold_fit_seconds_total), 4),
        "full_fit_seconds": round(float(full_fit_seconds), 4),
        "full_predict_seconds": round(float(full_predict_seconds), 4),
        "params": params_text(config),
        "warnings": warning_text,
    }
    row.update(oof_metrics)
    row.update(validation_metrics)
    return row


def make_meta_estimator(C: float, weight_mode: str | None, random_state: int) -> Pipeline:
    """Create the logistic stacking meta learner."""
    class_weight = "balanced" if weight_mode == "balanced_class_weight" else None
    model = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def fit_meta_estimator(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    weight_mode: str | None,
) -> None:
    """Fit meta estimator with optional sqrt-balanced sample weights."""
    if weight_mode == "sqrt_sample_weight":
        estimator.fit(X, y, model__sample_weight=sample_weight_for_mode(y, "sqrt_balanced"))
    else:
        estimator.fit(X, y)


def meta_param_grid() -> list[dict[str, Any]]:
    """Return compact meta-learner search grid."""
    rows: list[dict[str, Any]] = []
    for C in (0.1, 0.3, 1.0, 3.0):
        for weight_mode in ("none", "balanced_class_weight", "sqrt_sample_weight"):
            rows.append({"C": C, "weight_mode": weight_mode})
    return rows


def run_meta_search(
    ensemble_id: str,
    ensemble_type: str,
    base_ids: list[str],
    X_meta_train: np.ndarray,
    y_train: np.ndarray,
    label_names: list[str],
    n_splits: int,
    paths: ReportPaths,
) -> dict[str, Any]:
    """Select meta-learner hyperparameters using train-only CV."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED + 77)
    search_rows: list[dict[str, Any]] = []
    for params in meta_param_grid():
        fold_preds = np.full(len(y_train), -1, dtype=np.int64)
        fit_seconds_total = 0.0
        warnings_seen: list[str] = []
        for fold_index, (fit_idx, holdout_idx) in enumerate(splitter.split(X_meta_train, y_train)):
            estimator = make_meta_estimator(
                C=float(params["C"]),
                weight_mode=str(params["weight_mode"]),
                random_state=RANDOM_SEED + fold_index,
            )
            fit_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_meta_estimator(
                    estimator,
                    X_meta_train[fit_idx],
                    y_train[fit_idx],
                    str(params["weight_mode"]),
                )
            fit_seconds_total += time.perf_counter() - fit_start
            warn_text = warning_summary(caught)
            if warn_text:
                warnings_seen.append(warn_text)
            fold_preds[holdout_idx] = estimator.predict(X_meta_train[holdout_idx])
        if np.any(fold_preds < 0):
            raise ValueError(f"Meta CV predictions incomplete for {ensemble_id}")
        metrics = evaluate_predictions(y_train, fold_preds, label_names, prefix="inner_cv_")
        row = {
            "ensemble_id": ensemble_id,
            "ensemble_type": ensemble_type,
            "base_ids": "+".join(base_ids),
            "num_base_models": len(base_ids),
            "num_meta_features": X_meta_train.shape[1],
            "meta_model": "LogisticRegression",
            "meta_C": params["C"],
            "meta_weight_mode": params["weight_mode"],
            "inner_cv_macro_f1": round(metrics["inner_cv_macro_f1"], 10),
            "inner_cv_minority_recall_mean": round(
                metrics["inner_cv_minority_recall_mean"],
                10,
            ),
            "inner_cv_weighted_f1": round(metrics["inner_cv_weighted_f1"], 10),
            "fit_seconds_total": round(float(fit_seconds_total), 4),
            "params": params_text(params),
            "warnings": " | ".join(dict.fromkeys(warnings_seen)),
        }
        append_csv_row(paths.meta_search_csv, row, META_SEARCH_FIELDNAMES)
        search_rows.append(row)
    return max(
        search_rows,
        key=lambda row: (
            float(row["inner_cv_macro_f1"]),
            float(row["inner_cv_minority_recall_mean"]),
        ),
    )


def evaluate_stacking_ensemble(
    ensemble_id: str,
    base_ids: list[str],
    base_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    label_names: list[str],
    paths: ReportPaths,
    meta_cv_folds: int,
) -> dict[str, Any]:
    """Train and validate one stacking ensemble."""
    X_meta_train = np.hstack([base_outputs[base_id][0] for base_id in base_ids]).astype(np.float32)
    X_meta_validation = np.hstack([base_outputs[base_id][1] for base_id in base_ids]).astype(np.float32)
    if not np.isfinite(X_meta_train).all() or not np.isfinite(X_meta_validation).all():
        raise ValueError(f"Non-finite meta features in {ensemble_id}")

    log(
        f"[meta search] {ensemble_id}: {len(base_ids)} bases, "
        f"{X_meta_train.shape[1]} meta features, {meta_cv_folds}-fold CV."
    )
    best_meta = run_meta_search(
        ensemble_id,
        "stacking",
        base_ids,
        X_meta_train,
        y_train,
        label_names,
        meta_cv_folds,
        paths,
    )
    estimator = make_meta_estimator(
        C=float(best_meta["meta_C"]),
        weight_mode=str(best_meta["meta_weight_mode"]),
        random_state=RANDOM_SEED + 400,
    )
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_meta_estimator(
            estimator,
            X_meta_train,
            y_train,
            str(best_meta["meta_weight_mode"]),
        )
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    y_pred = np.asarray(estimator.predict(X_meta_validation), dtype=np.int64)
    predict_seconds = time.perf_counter() - predict_start
    metrics = rounded_metrics(evaluate_predictions(y_validation, y_pred, label_names, prefix="validation_"))
    row = {
        "ensemble_id": ensemble_id,
        "ensemble_type": "stacking",
        "base_ids": "+".join(base_ids),
        "num_base_models": len(base_ids),
        "num_meta_features": X_meta_train.shape[1],
        "meta_model": "LogisticRegression",
        "meta_C": best_meta["meta_C"],
        "meta_weight_mode": best_meta["meta_weight_mode"],
        "inner_cv_macro_f1": best_meta["inner_cv_macro_f1"],
        "inner_cv_minority_recall_mean": best_meta["inner_cv_minority_recall_mean"],
        "delta_macro_f1_vs_stage4_best": round(
            metrics["validation_macro_f1"] - STAGE4_BEST_MACRO_F1,
            10,
        ),
        "delta_minority_recall_vs_stage4_best": round(
            metrics["validation_minority_recall_mean"] - STAGE4_BEST_MINORITY_RECALL,
            10,
        ),
        "fit_seconds": round(float(fit_seconds), 4),
        "predict_seconds": round(float(predict_seconds), 4),
        "params": best_meta["params"],
        "warnings": warning_summary(caught),
    }
    row.update(metrics)
    log(
        f"[ensemble done] {ensemble_id} | macro-F1="
        f"{row['validation_macro_f1']:.4f} | minority="
        f"{row['validation_minority_recall_mean']:.4f}"
    )
    return row


def evaluate_average_ensemble(
    ensemble_id: str,
    base_ids: list[str],
    base_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    y_validation: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Evaluate a simple soft average of probability-producing bases."""
    validation_values = np.mean([base_outputs[base_id][1] for base_id in base_ids], axis=0)
    y_pred = labels_from_meta_features(validation_values)
    metrics = rounded_metrics(evaluate_predictions(y_validation, y_pred, label_names, prefix="validation_"))
    row = {
        "ensemble_id": ensemble_id,
        "ensemble_type": "soft_average",
        "base_ids": "+".join(base_ids),
        "num_base_models": len(base_ids),
        "num_meta_features": len(base_ids) * len(label_names),
        "meta_model": "",
        "meta_C": "",
        "meta_weight_mode": "",
        "inner_cv_macro_f1": "",
        "inner_cv_minority_recall_mean": "",
        "delta_macro_f1_vs_stage4_best": round(
            metrics["validation_macro_f1"] - STAGE4_BEST_MACRO_F1,
            10,
        ),
        "delta_minority_recall_vs_stage4_best": round(
            metrics["validation_minority_recall_mean"] - STAGE4_BEST_MINORITY_RECALL,
            10,
        ),
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "params": params_text({"method": "unweighted mean of class probabilities"}),
        "warnings": "",
    }
    row.update(metrics)
    log(
        f"[ensemble done] {ensemble_id} | macro-F1="
        f"{row['validation_macro_f1']:.4f} | minority="
        f"{row['validation_minority_recall_mean']:.4f}"
    )
    return row


def selected_base_configs(base_ids: set[str], include_anchor: bool) -> list[dict[str, Any]]:
    """Filter base configs from CLI options."""
    configs = [dict(config) for config in BASE_CONFIGS if include_anchor or config["group"] != "anchor"]
    if base_ids:
        configs = [config for config in configs if config["base_id"] in base_ids]
        found = {config["base_id"] for config in configs}
        missing = sorted(base_ids - found)
        if missing:
            raise ValueError(f"Unknown base IDs or excluded anchor IDs: {missing}")
    return configs


def build_ensemble_plan(configs: list[dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    """Return ensemble IDs, types, and base IDs to evaluate."""
    available = {str(config["base_id"]): config for config in configs}
    small_ids = [str(config["base_id"]) for config in configs if config["group"] == "small"]
    anchor_ids = [str(config["base_id"]) for config in configs if config["group"] == "anchor"]
    plan: list[tuple[str, str, list[str]]] = []
    if len(small_ids) >= 2:
        plan.append(("stack_small_only", "stacking", small_ids))
    if "lgbm_anchor_c11_sqrt" in available and len(small_ids) >= 2:
        plan.append(("stack_small_plus_lgbm_anchor", "stacking", small_ids + anchor_ids))

    proba_small = [
        str(config["base_id"])
        for config in configs
        if config["group"] == "small" and config["output_type"] == "proba"
    ]
    proba_with_anchor = [
        str(config["base_id"])
        for config in configs
        if config["output_type"] == "proba"
    ]
    if len(proba_small) >= 2:
        plan.append(("avg_small_proba", "soft_average", proba_small))
    if "lgbm_anchor_c11_sqrt" in available and len(proba_with_anchor) >= 2:
        plan.append(("avg_small_plus_lgbm_anchor_proba", "soft_average", proba_with_anchor))
    return plan


def summarize_outputs(
    paths: ReportPaths,
    base_rows: list[dict[str, Any]],
    ensemble_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write summary JSON and best CSV."""
    sorted_ensembles = sorted(
        ensemble_rows,
        key=lambda row: float(row["validation_macro_f1"]),
        reverse=True,
    )
    write_csv(paths.best_csv, sorted_ensembles[:5], ENSEMBLE_FIELDNAMES)
    best = sorted_ensembles[0] if sorted_ensembles else None
    summary = {
        "run_tag": args.run_tag,
        "n_folds": args.n_folds,
        "meta_cv_folds": args.meta_cv_folds,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "stage4_best_reference": {
            "model": "C11 full + LightGBM + sqrt_balanced",
            "validation_macro_f1": STAGE4_BEST_MACRO_F1,
            "minority_recall_mean": STAGE4_BEST_MINORITY_RECALL,
        },
        "num_base_models": len(base_rows),
        "num_ensembles": len(ensemble_rows),
        "best_ensemble": best,
        "base_rows": base_rows,
        "outputs": {
            "base_folds_csv": str(paths.base_folds_csv.relative_to(PROJECT_ROOT)),
            "base_summary_csv": str(paths.base_summary_csv.relative_to(PROJECT_ROOT)),
            "meta_search_csv": str(paths.meta_search_csv.relative_to(PROJECT_ROOT)),
            "ensemble_results_csv": str(paths.ensemble_results_csv.relative_to(PROJECT_ROOT)),
            "best_csv": str(paths.best_csv.relative_to(PROJECT_ROOT)),
        },
    }
    with paths.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, ensure_ascii=False, indent=2)


def prepare_output_files(
    paths: ReportPaths,
    force: bool,
    resume: bool,
    refresh_validation: bool,
) -> None:
    """Initialize output files for a fresh or resumed run."""
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    if force and resume:
        raise ValueError("--force and --resume should not be used together.")
    if force:
        for path in (
            paths.base_folds_csv,
            paths.base_summary_csv,
            paths.meta_search_csv,
            paths.ensemble_results_csv,
            paths.best_csv,
            paths.summary_json,
        ):
            if path.exists():
                path.unlink()
        for cache_path in paths.cache_dir.glob("*.npz"):
            cache_path.unlink()
    if not resume:
        write_csv(paths.base_folds_csv, [], BASE_FOLD_FIELDNAMES)
        write_csv(paths.base_summary_csv, [], BASE_SUMMARY_FIELDNAMES)
        write_csv(paths.meta_search_csv, [], META_SEARCH_FIELDNAMES)
        write_csv(paths.ensemble_results_csv, [], ENSEMBLE_FIELDNAMES)
    elif refresh_validation:
        write_csv(paths.base_summary_csv, [], BASE_SUMMARY_FIELDNAMES)
        write_csv(paths.meta_search_csv, [], META_SEARCH_FIELDNAMES)
        write_csv(paths.ensemble_results_csv, [], ENSEMBLE_FIELDNAMES)
        write_csv(paths.best_csv, [], ENSEMBLE_FIELDNAMES)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Stage 4 stacking ensemble.")
    parser.add_argument("--run-tag", default="full", help="Output tag; default writes final Stage 4 names.")
    parser.add_argument("--n-folds", type=int, default=5, help="OOF folds for base learners.")
    parser.add_argument("--meta-cv-folds", type=int, default=5, help="CV folds for meta-learner search.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Stratified train subsample for smoke tests.")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="Stratified validation subsample for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed base caches when available.")
    parser.add_argument(
        "--refresh-validation",
        action="store_true",
        help="With --resume, reuse OOF folds but refit full-train validation predictions.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite this run tag's prior outputs and caches.")
    parser.add_argument("--no-anchor", action="store_true", help="Do not run the LightGBM anchor base learner.")
    parser.add_argument(
        "--bases",
        default="",
        help="Comma-separated base IDs to run. Empty means all selected bases.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Stage 4 stacking."""
    args = parse_args()
    if args.n_folds < 2 or args.meta_cv_folds < 2:
        raise ValueError("Both --n-folds and --meta-cv-folds must be at least 2.")
    selected_ids = {item.strip() for item in args.bases.split(",") if item.strip()}
    include_anchor = not args.no_anchor
    configs = selected_base_configs(selected_ids, include_anchor)
    if len(configs) < 2:
        raise ValueError("At least two base learners are needed for stacking.")

    paths = build_report_paths(args.run_tag)
    prepare_output_files(
        paths,
        force=args.force,
        resume=args.resume,
        refresh_validation=args.refresh_validation,
    )
    write_dependency_versions(paths.dependency_json)
    feature_store = load_all_features()
    reference = feature_store["F1_color"]
    full_y_train = reference["y_train"]
    full_y_validation = reference["y_validation"]
    label_names = reference["label_names"]
    train_indices = stratified_limited_indices(
        full_y_train,
        args.max_train_samples,
        RANDOM_SEED + 10,
    )
    validation_indices = stratified_limited_indices(
        full_y_validation,
        args.max_validation_samples,
        RANDOM_SEED + 11,
    )
    y_train = full_y_train[train_indices]
    y_validation = full_y_validation[validation_indices]
    folds = list(
        StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_SEED).split(
            np.zeros(len(y_train)),
            y_train,
        )
    )

    log(
        f"[plan] Stage 4 stacking run_tag={args.run_tag} | bases={len(configs)} | "
        f"train={len(y_train)} validation={len(y_validation)} | folds={args.n_folds}"
    )
    estimated = "15-45 minutes"
    if args.max_train_samples is not None or args.max_validation_samples is not None:
        estimated = "1-5 minutes"
    elif include_anchor:
        estimated = "30-70 minutes"
    log(
        f"[estimate] Expected wall time: {estimated}. "
        "Each base fold and full-base refit writes progress/checkpoints."
    )

    base_rows: list[dict[str, Any]] = []
    base_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, config in enumerate(configs, start=1):
        log(f"[base {index}/{len(configs)}] {config['base_id']} | {config['description']}")
        summary, oof_meta, validation_meta = run_base_model(
            config,
            feature_store,
            y_train,
            y_validation,
            label_names,
            train_indices,
            validation_indices,
            folds,
            paths,
            args.resume,
            args.refresh_validation,
        )
        base_rows.append(summary)
        base_outputs[str(config["base_id"])] = (oof_meta, validation_meta)

    write_csv(paths.base_summary_csv, base_rows, BASE_SUMMARY_FIELDNAMES)

    ensemble_rows: list[dict[str, Any]] = []
    ensemble_plan = build_ensemble_plan(configs)
    for ensemble_id, ensemble_type, base_ids in ensemble_plan:
        if not all(base_id in base_outputs for base_id in base_ids):
            continue
        if ensemble_type == "stacking":
            row = evaluate_stacking_ensemble(
                ensemble_id,
                base_ids,
                base_outputs,
                y_train,
                y_validation,
                label_names,
                paths,
                args.meta_cv_folds,
            )
        elif ensemble_type == "soft_average":
            row = evaluate_average_ensemble(
                ensemble_id,
                base_ids,
                base_outputs,
                y_validation,
                label_names,
            )
        else:
            raise ValueError(f"Unsupported ensemble_type: {ensemble_type}")
        append_csv_row(paths.ensemble_results_csv, row, ENSEMBLE_FIELDNAMES)
        ensemble_rows.append(row)

    summarize_outputs(paths, base_rows, ensemble_rows, args)
    if ensemble_rows:
        best = max(ensemble_rows, key=lambda row: float(row["validation_macro_f1"]))
        log(
            f"[complete] Best ensemble: {best['ensemble_id']} | "
            f"macro-F1={float(best['validation_macro_f1']):.4f} | "
            f"delta_vs_stage4_best={float(best['delta_macro_f1_vs_stage4_best']):+.4f}"
        )
    else:
        log("[complete] No ensemble rows were produced.")


if __name__ == "__main__":
    main()
