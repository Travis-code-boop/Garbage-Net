"""Stage 3: adaptive feature fusion and broad model screening.

Notebook block purpose:
    Stage 3 adaptive feature fusion screening. This block starts from the best
    tuned single feature group, builds research-informed feature fusion
    candidates, screens baseline and ensemble classifiers with fixed compact
    parameters, and writes checkpointed validation reports. The test split is
    intentionally not used here.

Screening strategy:
    1. Load already extracted F1-F6 feature caches.
    2. Validate split and label alignment before any concatenation.
    3. Build adaptive fusion combinations from Stage 2 results.
    4. Train fixed-parameter models on train and evaluate once on validation.
    5. Save per-run metrics immediately so long runs leave usable checkpoints.
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
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import Normalizer, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

try:
    from xgboost import XGBClassifier  # noqa: E402
except Exception:  # pragma: no cover - handled at runtime for clearer logs.
    XGBClassifier = None  # type: ignore[assignment]

try:
    from lightgbm import LGBMClassifier  # noqa: E402
except Exception:  # pragma: no cover - handled at runtime for clearer logs.
    LGBMClassifier = None  # type: ignore[assignment]


RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
STAGE3_COMBO_PLAN_CSV = REPORT_DIR / "stage3_feature_combo_plan.csv"
STAGE3_RESULTS_CSV = REPORT_DIR / "stage3_model_screening.csv"
STAGE3_BEST_BY_COMBO_CSV = REPORT_DIR / "stage3_best_by_combo.csv"
STAGE3_BEST_BY_MODEL_CSV = REPORT_DIR / "stage3_best_by_model.csv"
STAGE3_FEATURE_ADDITION_CSV = REPORT_DIR / "stage3_feature_addition_summary.csv"
STAGE3_SHORTLIST_CSV = REPORT_DIR / "stage3_candidate_shortlist.csv"
STAGE3_PREDICTIONS_NPZ = REPORT_DIR / "stage3_validation_predictions.npz"
STAGE3_RUN_SUMMARY_JSON = REPORT_DIR / "stage3_run_summary.json"
STAGE3_DEPENDENCY_JSON = REPORT_DIR / "stage3_dependency_versions.json"


FEATURE_CONFIGS = {
    "F1_color": {
        "description": "color features 1039d",
        "path": PROJECT_ROOT / "data" / "features" / "F1_color" / "F1_color_1039.npz",
    },
    "F2_texture": {
        "description": "texture features 30d",
        "path": PROJECT_ROOT / "data" / "features" / "F2_texture" / "F2_texture_30.npz",
    },
    "F3_hog": {
        "description": "HOG features 1764d",
        "path": PROJECT_ROOT / "data" / "features" / "F3_hog" / "F3_hog_1764.npz",
    },
    "F4_shape": {
        "description": "shape features 12d",
        "path": PROJECT_ROOT / "data" / "features" / "F4_shape" / "F4_shape_12.npz",
    },
    "F5_gist": {
        "description": "GIST features 64d",
        "path": PROJECT_ROOT / "data" / "features" / "F5_gist" / "F5_gist_64.npz",
    },
    "F6_bof": {
        "description": "SIFT+ORB BoF features 256d",
        "path": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_bof_256.npz",
    },
}


COMBO_CONFIGS = [
    {
        "combo_id": "S3_00_F3_hog",
        "feature_ids": ["F3_hog"],
        "role": "best_single_anchor",
        "reason": "Stage 2 tuned best single feature; validation macro-F1 0.5612.",
    },
    {
        "combo_id": "S3_01_F3_hog__F1_color",
        "feature_ids": ["F3_hog", "F1_color"],
        "role": "one_step_addition",
        "reason": "Color had the best minority recall mean in Stage 2.",
    },
    {
        "combo_id": "S3_02_F3_hog__F2_texture",
        "feature_ids": ["F3_hog", "F2_texture"],
        "role": "one_step_addition",
        "reason": "Texture is low-dimensional and stable, with good macro-F1.",
    },
    {
        "combo_id": "S3_03_F3_hog__F5_gist",
        "feature_ids": ["F3_hog", "F5_gist"],
        "role": "one_step_addition",
        "reason": "GIST adds compact global structure information.",
    },
    {
        "combo_id": "S3_04_F3_hog__F6_bof",
        "feature_ids": ["F3_hog", "F6_bof"],
        "role": "one_step_addition",
        "reason": "BoF is research-backed and helped battery/biological recall.",
    },
    {
        "combo_id": "S3_05_F3_hog__F1_color__F2_texture",
        "feature_ids": ["F3_hog", "F1_color", "F2_texture"],
        "role": "adaptive_candidate",
        "reason": "Combines strongest edge feature with color and texture complements.",
    },
    {
        "combo_id": "S3_06_F3_hog__F1_color__F5_gist",
        "feature_ids": ["F3_hog", "F1_color", "F5_gist"],
        "role": "adaptive_candidate",
        "reason": "Tests color plus global layout without the texture block.",
    },
    {
        "combo_id": "S3_07_F3_hog__F2_texture__F5_gist",
        "feature_ids": ["F3_hog", "F2_texture", "F5_gist"],
        "role": "adaptive_candidate",
        "reason": "Tests low-dimensional texture and global structure complements.",
    },
    {
        "combo_id": "S3_08_F3_hog__F1_color__F2_texture__F5_gist",
        "feature_ids": ["F3_hog", "F1_color", "F2_texture", "F5_gist"],
        "role": "adaptive_candidate",
        "reason": "Main compact fusion candidate before adding BoF.",
    },
    {
        "combo_id": "S3_09_F3_hog__F1_color__F2_texture__F5_gist__F6_bof",
        "feature_ids": ["F3_hog", "F1_color", "F2_texture", "F5_gist", "F6_bof"],
        "role": "adaptive_candidate",
        "reason": "Adds local keypoint histogram to the compact fusion candidate.",
    },
    {
        "combo_id": "S3_10_F1_color__F2_texture",
        "feature_ids": ["F1_color", "F2_texture"],
        "role": "reference_combo",
        "reason": "Lightweight color+texture reference from the original C7 idea.",
    },
    {
        "combo_id": "S3_11_F1_color__F2_texture__F5_gist",
        "feature_ids": ["F1_color", "F2_texture", "F5_gist"],
        "role": "reference_combo",
        "reason": "Lightweight color+texture+GIST reference without HOG.",
    },
    {
        "combo_id": "S3_12_C10_reference_no_bof",
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog"],
        "role": "reference_combo",
        "reason": "Original C10-style all handcrafted features except BoF.",
    },
    {
        "combo_id": "S3_13_C11_reference_full_handcrafted",
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog", "F6_bof"],
        "role": "reference_combo",
        "reason": "Original C11-style full handcrafted feature reference.",
    },
]

MODEL_NAMES = [
    "LogisticRegression",
    "KNN",
    "LinearSVM_SGD",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "LightGBM",
]


def log(message: str) -> None:
    """Print a flush-safe progress message for long runs."""
    print(message, flush=True)


def module_version(module_name: str) -> str:
    """Return an importable module version string."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}: {exc}"
    return str(getattr(module, "__version__", "unknown"))


def write_dependency_versions() -> dict[str, str]:
    """Save dependency versions used by Stage 3."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": module_version("numpy"),
        "scipy": module_version("scipy"),
        "scikit_learn": module_version("sklearn"),
        "xgboost": module_version("xgboost"),
        "lightgbm": module_version("lightgbm"),
    }
    STAGE3_DEPENDENCY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with STAGE3_DEPENDENCY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(versions, handle, ensure_ascii=False, indent=2)
    return versions


def jsonable(value: Any) -> Any:
    """Convert numpy and estimator values to JSON-friendly values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def read_existing_run_ids(path: Path) -> set[str]:
    """Read checkpointed run IDs when resuming."""
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["run_id"] for row in csv.DictReader(handle) if row.get("run_id")}


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
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
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
        "feature_names": cache["feature_names"],
        "feature_groups": cache["feature_groups"],
        "image_ids_train": cache["image_ids_train"],
        "image_ids_validation": cache["image_ids_validation"],
    }


def load_all_features() -> dict[str, dict[str, Any]]:
    """Load all feature blocks needed for Stage 3."""
    feature_store: dict[str, dict[str, Any]] = {}
    for feature_id, config in FEATURE_CONFIGS.items():
        path = Path(config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing feature cache: {path}")
        feature_store[feature_id] = load_feature_cache(path)
        log(
            f"[load] {feature_id}: train {feature_store[feature_id]['X_train'].shape}, "
            f"validation {feature_store[feature_id]['X_validation'].shape}"
        )
    validate_feature_alignment(feature_store)
    return feature_store


def validate_feature_alignment(feature_store: dict[str, dict[str, Any]]) -> None:
    """Ensure feature blocks align before concatenation."""
    first_key = next(iter(feature_store))
    reference = feature_store[first_key]
    for feature_id, data in feature_store.items():
        for split in ("train", "validation"):
            y_key = f"y_{split}"
            id_key = f"image_ids_{split}"
            if not np.array_equal(reference[y_key], data[y_key]):
                raise ValueError(f"Label alignment mismatch for {feature_id} / {split}")
            if not np.array_equal(reference[id_key], data[id_key]):
                raise ValueError(f"Image-id alignment mismatch for {feature_id} / {split}")
        if reference["label_names"] != data["label_names"]:
            raise ValueError(f"Label-name alignment mismatch for {feature_id}")
    log("[check] Feature split, image-id, and label alignment passed.")


def combo_feature_dim(feature_store: dict[str, dict[str, Any]], feature_ids: list[str]) -> int:
    """Return concatenated dimension for a feature combination."""
    return int(sum(feature_store[feature_id]["X_train"].shape[1] for feature_id in feature_ids))


def build_combo_matrix(
    feature_store: dict[str, dict[str, Any]],
    feature_ids: list[str],
    split: str,
) -> np.ndarray:
    """Concatenate feature groups for one split."""
    pieces = [feature_store[feature_id][f"X_{split}"] for feature_id in feature_ids]
    if len(pieces) == 1:
        matrix = pieces[0]
    else:
        matrix = np.hstack(pieces)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values found in combo {feature_ids} / {split}")
    return matrix.astype(np.float32, copy=False)


def combo_plan_rows(feature_store: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build and save the Stage 3 feature-combination plan."""
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(COMBO_CONFIGS, start=1):
        feature_ids = list(config["feature_ids"])
        rows.append(
            {
                "plan_order": index,
                "combo_id": config["combo_id"],
                "role": config["role"],
                "feature_ids": "+".join(feature_ids),
                "feature_count": len(feature_ids),
                "feature_dim": combo_feature_dim(feature_store, feature_ids),
                "reason": config["reason"],
            }
        )
    write_csv(
        STAGE3_COMBO_PLAN_CSV,
        rows,
        [
            "plan_order",
            "combo_id",
            "role",
            "feature_ids",
            "feature_count",
            "feature_dim",
            "reason",
        ],
    )
    return rows


def make_estimator(
    model_name: str,
    num_classes: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Create one fixed-parameter Stage 3 screening estimator."""
    variance_filter = VarianceThreshold(threshold=0.0)

    if model_name == "LogisticRegression":
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.3,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2500,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        return estimator, {}, estimator.get_params(deep=False)

    if model_name == "KNN":
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                ("normalizer", Normalizer(norm="l2")),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=9,
                        weights="distance",
                        metric="manhattan",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return estimator, {}, estimator.get_params(deep=False)

    if model_name == "LinearSVM_SGD":
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=3e-4,
                        class_weight="balanced",
                        average=True,
                        max_iter=2500,
                        tol=1e-3,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return estimator, {}, estimator.get_params(deep=False)

    if model_name == "DecisionTree":
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                (
                    "model",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=16,
                        min_samples_leaf=10,
                        max_features=0.6,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        return estimator, {}, estimator.get_params(deep=False)

    if model_name == "RandomForest":
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=260,
                        criterion="gini",
                        max_depth=None,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return estimator, {}, estimator.get_params(deep=False)

    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not importable. Install xgboost before Stage 3.")
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=num_classes,
                        n_estimators=360,
                        max_depth=5,
                        learning_rate=0.055,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_lambda=2.0,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )
        return estimator, {"model__sample_weight": "balanced"}, estimator.get_params(deep=False)

    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not importable. Install lightgbm before Stage 3.")
        estimator = Pipeline(
            steps=[
                ("variance", variance_filter),
                (
                    "model",
                    LGBMClassifier(
                        objective="multiclass",
                        num_class=num_classes,
                        n_estimators=460,
                        num_leaves=31,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_lambda=1.0,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                        verbose=-1,
                        force_col_wise=True,
                    ),
                ),
            ]
        )
        return estimator, {"model__sample_weight": "balanced"}, estimator.get_params(deep=False)

    raise ValueError(f"Unsupported model: {model_name}")


def estimator_params_summary(estimator: Pipeline) -> str:
    """Return compact model-step parameters for reporting."""
    model = estimator.named_steps["model"]
    params = model.get_params(deep=False)
    keep_keys = [
        "C",
        "class_weight",
        "n_neighbors",
        "weights",
        "metric",
        "alpha",
        "average",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "n_estimators",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "num_leaves",
        "reg_lambda",
    ]
    return params_text({key: params[key] for key in keep_keys if key in params})


def unique_warning_texts(caught_warnings: list[warnings.WarningMessage]) -> str:
    """Return a short de-duplicated warning summary."""
    messages: list[str] = []
    for item in caught_warnings:
        message = str(item.message).replace("\n", " ")
        if message not in messages:
            messages.append(message)
    return " | ".join(messages[:5])


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    """Compute validation metrics and per-class recalls/F1s."""
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

    metrics: dict[str, float] = {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "validation_macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "validation_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "validation_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    for label in label_names:
        label_index = label_names.index(label)
        metrics[f"recall_{label}"] = float(per_class_recall[label_index])
        metrics[f"f1_{label}"] = float(per_class_f1[label_index])

    metrics["minority_recall_mean"] = float(
        np.mean([metrics[f"recall_{label}"] for label in MINORITY_LABELS])
    )
    return metrics


def result_fieldnames(label_names: list[str]) -> list[str]:
    """Return stable Stage 3 result columns."""
    per_class_fields: list[str] = []
    for label in label_names:
        per_class_fields.extend([f"recall_{label}", f"f1_{label}"])
    return [
        "run_id",
        "combo_id",
        "combo_role",
        "feature_ids",
        "feature_dim",
        "model",
        "train_samples",
        "validation_samples",
        "validation_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_weighted_f1",
        "minority_recall_mean",
        "fit_seconds",
        "predict_seconds",
        "total_seconds",
        "model_params",
        "warnings",
        *per_class_fields,
    ]


def run_one_screening(
    combo_config: dict[str, Any],
    model_name: str,
    feature_store: dict[str, dict[str, Any]],
    sample_weight: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Train and evaluate one Stage 3 combo/model run."""
    feature_ids = list(combo_config["feature_ids"])
    X_train = build_combo_matrix(feature_store, feature_ids, "train")
    X_validation = build_combo_matrix(feature_store, feature_ids, "validation")
    y_train = feature_store[feature_ids[0]]["y_train"]
    y_validation = feature_store[feature_ids[0]]["y_validation"]
    label_names = feature_store[feature_ids[0]]["label_names"]

    estimator, fit_params_spec, _ = make_estimator(model_name, len(label_names))
    fit_params: dict[str, Any] = {}
    for key, value in fit_params_spec.items():
        if value == "balanced":
            fit_params[key] = sample_weight
        else:
            fit_params[key] = value

    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X_train, y_train, **fit_params)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = estimator.predict(X_validation)
    predict_seconds = time.perf_counter() - predict_start

    metrics = evaluate_predictions(y_validation, y_pred, label_names)
    row: dict[str, Any] = {
        "run_id": f"{combo_config['combo_id']}__{model_name}",
        "combo_id": combo_config["combo_id"],
        "combo_role": combo_config["role"],
        "feature_ids": "+".join(feature_ids),
        "feature_dim": int(X_train.shape[1]),
        "model": model_name,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_validation)),
        "fit_seconds": round(float(fit_seconds), 4),
        "predict_seconds": round(float(predict_seconds), 4),
        "total_seconds": round(float(fit_seconds + predict_seconds), 4),
        "model_params": estimator_params_summary(estimator),
        "warnings": unique_warning_texts(caught),
    }
    row.update({key: round(value, 10) for key, value in metrics.items()})
    return row, np.asarray(y_pred, dtype=np.int64)


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    """Read Stage 3 result rows from CSV."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    """Read a CSV numeric field safely."""
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_results(rows: list[dict[str, Any]], label_names: list[str]) -> None:
    """Write best-by reports and shortlist after screening."""
    if not rows:
        return

    sorted_rows = sorted(rows, key=lambda row: as_float(row, "validation_macro_f1"), reverse=True)

    best_by_combo: list[dict[str, Any]] = []
    for combo_id in sorted({row["combo_id"] for row in rows}):
        combo_rows = [row for row in rows if row["combo_id"] == combo_id]
        best_by_combo.append(
            max(combo_rows, key=lambda row: as_float(row, "validation_macro_f1"))
        )
    best_by_combo = sorted(
        best_by_combo,
        key=lambda row: as_float(row, "validation_macro_f1"),
        reverse=True,
    )

    best_by_model: list[dict[str, Any]] = []
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        best_by_model.append(
            max(model_rows, key=lambda row: as_float(row, "validation_macro_f1"))
        )
    best_by_model = sorted(
        best_by_model,
        key=lambda row: as_float(row, "validation_macro_f1"),
        reverse=True,
    )

    result_fields = result_fieldnames(label_names)
    write_csv(STAGE3_BEST_BY_COMBO_CSV, best_by_combo, result_fields)
    write_csv(STAGE3_BEST_BY_MODEL_CSV, best_by_model, result_fields)

    base_best = next(
        row for row in best_by_combo if row["combo_id"] == "S3_00_F3_hog"
    )
    addition_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(best_by_combo, start=1):
        addition_rows.append(
            {
                "rank": rank,
                "combo_id": row["combo_id"],
                "feature_ids": row["feature_ids"],
                "best_model": row["model"],
                "feature_dim": row["feature_dim"],
                "validation_macro_f1": row["validation_macro_f1"],
                "minority_recall_mean": row["minority_recall_mean"],
                "macro_f1_delta_vs_F3_anchor": round(
                    as_float(row, "validation_macro_f1")
                    - as_float(base_best, "validation_macro_f1"),
                    10,
                ),
                "minority_recall_delta_vs_F3_anchor": round(
                    as_float(row, "minority_recall_mean")
                    - as_float(base_best, "minority_recall_mean"),
                    10,
                ),
                "total_seconds": row["total_seconds"],
            }
        )
    write_csv(
        STAGE3_FEATURE_ADDITION_CSV,
        addition_rows,
        [
            "rank",
            "combo_id",
            "feature_ids",
            "best_model",
            "feature_dim",
            "validation_macro_f1",
            "minority_recall_mean",
            "macro_f1_delta_vs_F3_anchor",
            "minority_recall_delta_vs_F3_anchor",
            "total_seconds",
        ],
    )

    best_macro = as_float(sorted_rows[0], "validation_macro_f1")
    best_minority = max(as_float(row, "minority_recall_mean") for row in rows)
    best_baseline = max(
        (
            row
            for row in rows
            if row["model"] in {"LogisticRegression", "KNN", "LinearSVM_SGD", "DecisionTree"}
        ),
        key=lambda row: as_float(row, "validation_macro_f1"),
    )

    shortlist: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, row in enumerate(sorted_rows, start=1):
        reasons: list[str] = []
        macro = as_float(row, "validation_macro_f1")
        minority = as_float(row, "minority_recall_mean")
        if rank <= 8:
            reasons.append("top8_macro_f1")
        if macro >= best_macro - 0.015:
            reasons.append("within_0.015_of_best_macro_f1")
        if minority >= best_minority - 0.02 and macro >= 0.50:
            reasons.append("minority_recall_competitive")
        if row["run_id"] == best_baseline["run_id"]:
            reasons.append("best_four_baseline_model")
        if row["run_id"] in {item["run_id"] for item in best_by_model[:3]}:
            reasons.append("best_model_family_representative")
        if reasons and row["run_id"] not in seen:
            copied = dict(row)
            copied["overall_rank"] = rank
            copied["shortlist_reasons"] = "; ".join(reasons)
            shortlist.append(copied)
            seen.add(row["run_id"])
        if len(shortlist) >= 15:
            break

    write_csv(
        STAGE3_SHORTLIST_CSV,
        shortlist,
        ["overall_rank", "shortlist_reasons", *result_fields],
    )

    summary = {
        "num_completed_runs": len(rows),
        "best_run": sorted_rows[0],
        "best_by_combo": best_by_combo[:5],
        "best_by_model": best_by_model,
        "shortlist_size": len(shortlist),
        "outputs": {
            "combo_plan_csv": str(STAGE3_COMBO_PLAN_CSV.relative_to(PROJECT_ROOT)),
            "model_screening_csv": str(STAGE3_RESULTS_CSV.relative_to(PROJECT_ROOT)),
            "best_by_combo_csv": str(STAGE3_BEST_BY_COMBO_CSV.relative_to(PROJECT_ROOT)),
            "best_by_model_csv": str(STAGE3_BEST_BY_MODEL_CSV.relative_to(PROJECT_ROOT)),
            "feature_addition_csv": str(STAGE3_FEATURE_ADDITION_CSV.relative_to(PROJECT_ROOT)),
            "shortlist_csv": str(STAGE3_SHORTLIST_CSV.relative_to(PROJECT_ROOT)),
            "predictions_npz": str(STAGE3_PREDICTIONS_NPZ.relative_to(PROJECT_ROOT)),
        },
    }
    with STAGE3_RUN_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse Stage 3 CLI options."""
    parser = argparse.ArgumentParser(description="Run Stage 3 feature fusion screening.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip run IDs already present in stage3_model_screening.csv.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of new runs, useful for smoke tests.",
    )
    parser.add_argument(
        "--combos",
        default="",
        help="Comma-separated combo IDs to run. Empty means all planned combos.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to run. Empty means all Stage 3 models.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Stage 3 adaptive feature fusion screening."""
    args = parse_args()
    versions = write_dependency_versions()
    log(f"[env] Dependency versions saved: {versions}")

    if XGBClassifier is None:
        raise RuntimeError("XGBoost is required for Stage 3 but is unavailable.")
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is required for Stage 3 but is unavailable.")

    feature_store = load_all_features()
    label_names = feature_store["F3_hog"]["label_names"]
    y_train = feature_store["F3_hog"]["y_train"]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    plan_rows = combo_plan_rows(feature_store)

    selected_combo_ids = {
        item.strip() for item in args.combos.split(",") if item.strip()
    }
    selected_model_names = {
        item.strip() for item in args.models.split(",") if item.strip()
    }
    combo_configs = [
        combo
        for combo in COMBO_CONFIGS
        if not selected_combo_ids or combo["combo_id"] in selected_combo_ids
    ]
    model_names = [
        model
        for model in MODEL_NAMES
        if not selected_model_names or model in selected_model_names
    ]

    if selected_combo_ids and len(combo_configs) != len(selected_combo_ids):
        found = {combo["combo_id"] for combo in combo_configs}
        missing = sorted(selected_combo_ids - found)
        raise ValueError(f"Unknown combo IDs: {missing}")
    if selected_model_names and len(model_names) != len(selected_model_names):
        found_models = set(model_names)
        missing_models = sorted(selected_model_names - found_models)
        raise ValueError(f"Unknown model names: {missing_models}")

    run_plan = [
        (combo, model_name)
        for combo in combo_configs
        for model_name in model_names
    ]
    existing_run_ids = read_existing_run_ids(STAGE3_RESULTS_CSV) if args.resume else set()
    pending_plan = [
        (combo, model_name)
        for combo, model_name in run_plan
        if f"{combo['combo_id']}__{model_name}" not in existing_run_ids
    ]
    if args.max_runs is not None:
        pending_plan = pending_plan[: args.max_runs]

    log(
        "[plan] Stage 3 broad screening: "
        f"{len(plan_rows)} planned combos, {len(model_names)} models, "
        f"{len(run_plan)} total possible runs, {len(pending_plan)} pending runs."
    )
    log(
        "[estimate] Full broad screening is expected to take roughly 30-90 minutes "
        "on this data. RF/XGBoost/LightGBM will dominate runtime; each run is "
        "checkpointed immediately after validation."
    )

    result_fields = result_fieldnames(label_names)
    if not args.resume:
        write_csv(STAGE3_RESULTS_CSV, [], result_fields)

    prediction_rows: list[np.ndarray] = []
    prediction_run_ids: list[str] = []
    start_all = time.perf_counter()
    for index, (combo, model_name) in enumerate(pending_plan, start=1):
        run_id = f"{combo['combo_id']}__{model_name}"
        dim = combo_feature_dim(feature_store, list(combo["feature_ids"]))
        log(
            f"[run {index}/{len(pending_plan)}] {run_id} | dim={dim} | "
            "starting fit+validation"
        )
        run_start = time.perf_counter()
        row, y_pred = run_one_screening(combo, model_name, feature_store, sample_weight)
        append_csv_row(STAGE3_RESULTS_CSV, row, result_fields)
        prediction_rows.append(y_pred)
        prediction_run_ids.append(run_id)
        log(
            f"[done] {run_id} | macro-F1={row['validation_macro_f1']:.4f} | "
            f"minority-recall={row['minority_recall_mean']:.4f} | "
            f"time={time.perf_counter() - run_start:.1f}s"
        )

    if prediction_rows:
        np.savez_compressed(
            STAGE3_PREDICTIONS_NPZ,
            run_ids=np.asarray(prediction_run_ids),
            y_pred_validation=np.vstack(prediction_rows),
            y_validation=feature_store["F3_hog"]["y_validation"],
            label_names=np.asarray(label_names),
        )

    rows = read_result_rows(STAGE3_RESULTS_CSV)
    summarize_results(rows, label_names)
    log(
        f"[complete] Stage 3 screening summary written after "
        f"{time.perf_counter() - start_all:.1f}s for this invocation."
    )


if __name__ == "__main__":
    main()
