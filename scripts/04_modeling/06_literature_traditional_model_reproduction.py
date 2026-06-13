"""Literature-inspired traditional ML reproductions on the local waste dataset.

Notebook block purpose:
    This block applies selected traditional, non-neural waste-classification
    research pipelines to the fixed local train/validation split. It is kept
    separate from the core modeling stages because the main model is still
    under development. By default the test split remains sealed.

Progress and runtime policy:
    1. Load each candidate's feature matrix.
    2. Run a small stratified pilot fit before full training.
    3. Print estimated full training/prediction time before fitting.
    4. Print progress and checkpoint reports after each completed candidate.
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
from sklearn.base import clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

try:
    from lightgbm import LGBMClassifier  # noqa: E402
except Exception:  # pragma: no cover - handled at runtime.
    LGBMClassifier = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier  # noqa: E402
except Exception:  # pragma: no cover - handled at runtime.
    XGBClassifier = None  # type: ignore[assignment]


RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
RESULTS_CSV = REPORT_DIR / "literature_traditional_results.csv"
PER_CLASS_CSV = REPORT_DIR / "literature_traditional_per_class.csv"
PREDICTIONS_CSV = REPORT_DIR / "literature_traditional_validation_predictions.csv"
CONFUSION_CSV = REPORT_DIR / "literature_traditional_confusion_matrices.csv"
SUMMARY_JSON = REPORT_DIR / "literature_traditional_summary.json"
PREDICTIONS_NPZ = REPORT_DIR / "literature_traditional_validation_predictions.npz"
DEPENDENCY_JSON = REPORT_DIR / "literature_traditional_dependency_versions.json"

FEATURE_CONFIGS = {
    "F0_pixels_rgb": {
        "description": "RGB pixel baseline 3072d",
        "path": PROJECT_ROOT / "data" / "features" / "F0_pixels" / "F0_pixels_rgb_3072.npz",
    },
    "F1_color": {
        "description": "HSV/BGR color histograms and HSV statistics 1039d",
        "path": PROJECT_ROOT / "data" / "features" / "F1_color" / "F1_color_1039.npz",
    },
    "F2_texture": {
        "description": "GLCM + LBP texture features 30d",
        "path": PROJECT_ROOT / "data" / "features" / "F2_texture" / "F2_texture_30.npz",
    },
    "F3_hog": {
        "description": "HOG edge/shape features 1764d",
        "path": PROJECT_ROOT / "data" / "features" / "F3_hog" / "F3_hog_1764.npz",
    },
    "F4_shape": {
        "description": "contour geometry + Hu moments 12d",
        "path": PROJECT_ROOT / "data" / "features" / "F4_shape" / "F4_shape_12.npz",
    },
    "F5_gist": {
        "description": "GIST global layout features 64d",
        "path": PROJECT_ROOT / "data" / "features" / "F5_gist" / "F5_gist_64.npz",
    },
    "F6_sift_bof": {
        "description": "SIFT Bag-of-Features histogram 128d",
        "path": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_sift_bof_128.npz",
    },
    "F6_bof": {
        "description": "SIFT + ORB Bag-of-Features histograms 256d",
        "path": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_bof_256.npz",
    },
}


def log(message: str) -> None:
    """Print a flush-safe progress line."""
    print(message, flush=True)


def module_version(module_name: str) -> str:
    """Return an importable module version string."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report unavailable versions clearly.
        return f"unavailable: {exc.__class__.__name__}: {exc}"
    return str(getattr(module, "__version__", "unknown"))


def write_dependency_versions() -> dict[str, str]:
    """Save dependency versions used by this reproduction script."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": module_version("numpy"),
        "scikit_learn": module_version("sklearn"),
        "xgboost": module_version("xgboost"),
        "lightgbm": module_version("lightgbm"),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DEPENDENCY_JSON.write_text(
        json.dumps(versions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return versions


def jsonable(value: Any) -> Any:
    """Convert numpy and estimator values to JSON-friendly values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return str(value)
        return value
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
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return value.__class__.__name__


def params_text(params: dict[str, Any]) -> str:
    """Compact parameter summary for CSV readability."""
    return json.dumps(jsonable(params), ensure_ascii=False, sort_keys=True)


def format_seconds(seconds: float | None) -> str:
    """Return a short human-readable duration."""
    if seconds is None or math.isnan(seconds):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    return f"{minutes / 60:.1f}h"


def load_feature_cache(feature_id: str) -> dict[str, Any]:
    """Load one feature cache."""
    config = FEATURE_CONFIGS[feature_id]
    cache = np.load(config["path"], allow_pickle=False)
    return {
        "X_train": cache["X_train"],
        "y_train": cache["y_train"],
        "labels_train": cache["labels_train"],
        "image_ids_train": cache["image_ids_train"],
        "paths_train": cache["paths_train"],
        "X_validation": cache["X_validation"],
        "y_validation": cache["y_validation"],
        "labels_validation": cache["labels_validation"],
        "image_ids_validation": cache["image_ids_validation"],
        "paths_validation": cache["paths_validation"],
        "X_test": cache["X_test"],
        "y_test": cache["y_test"],
        "labels_test": cache["labels_test"],
        "image_ids_test": cache["image_ids_test"],
        "paths_test": cache["paths_test"],
        "label_names": cache["label_names"].tolist(),
    }


def assert_same_split(reference: dict[str, Any], current: dict[str, Any], split: str, feature_id: str) -> None:
    """Validate that feature caches are aligned before concatenation."""
    for key in (f"y_{split}", f"labels_{split}", f"image_ids_{split}", f"paths_{split}"):
        if not np.array_equal(reference[key], current[key]):
            raise ValueError(f"Feature cache {feature_id} is not aligned for {key}.")
    if reference["label_names"] != current["label_names"]:
        raise ValueError(f"Feature cache {feature_id} has a different label mapping.")


def load_feature_bundle(feature_ids: list[str], include_test: bool) -> dict[str, Any]:
    """Concatenate one candidate's feature groups with split alignment checks."""
    caches = [load_feature_cache(feature_id) for feature_id in feature_ids]
    reference = caches[0]
    splits = ["train", "validation"]
    if include_test:
        splits.append("test")

    for feature_id, cache in zip(feature_ids[1:], caches[1:], strict=True):
        for split in splits:
            assert_same_split(reference, cache, split, feature_id)

    bundle: dict[str, Any] = {
        "feature_ids": feature_ids,
        "feature_descriptions": [FEATURE_CONFIGS[feature_id]["description"] for feature_id in feature_ids],
        "label_names": reference["label_names"],
    }
    for split in splits:
        bundle[f"X_{split}"] = np.concatenate(
            [cache[f"X_{split}"] for cache in caches],
            axis=1,
        ).astype(np.float32, copy=False)
        for key in ("y", "labels", "image_ids", "paths"):
            bundle[f"{key}_{split}"] = reference[f"{key}_{split}"]
    return bundle


def make_candidate_configs(num_classes: int) -> list[dict[str, Any]]:
    """Return selected literature-inspired candidate pipelines."""
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is unavailable but is required for the Nguyen LightGBM candidate.")
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is unavailable but is required for the Nguyen XGBoost candidate.")

    return [
        {
            "candidate_id": "yang_thung_2016_sift_bof_rbf_svm",
            "source_short": "Yang & Thung 2016",
            "source_title": "Classification of Trash for Recyclability Status",
            "source_url": "https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf",
            "source_model": "SIFT descriptors + Bag of Features + RBF-kernel SVM",
            "source_parameter_note": "Paper explicitly reports RBF kernel, C=1000, gamma=0.5.",
            "parameter_source": "paper_reported",
            "feature_ids": ["F6_sift_bof"],
            "feature_mapping_note": (
                "Uses the local SIFT BoF cache. The paper used k-means BoF with k equal to its "
                "training-example count; this project already has a train-only 128-word SIFT vocabulary."
            ),
            "model_label": "RBF_SVM",
            "scaling_family": "kernel",
            "use_sample_weight": False,
            "estimator": SVC(
                kernel="rbf",
                C=1000.0,
                gamma=0.5,
                decision_function_shape="ovr",
                cache_size=1000,
            ),
        },
        {
            "candidate_id": "nguyen_2026_handcrafted_lightgbm",
            "source_short": "Nguyen et al. 2026",
            "source_title": "Towards Accurate and Efficient Waste Image Classification",
            "source_url": "https://arxiv.org/abs/2510.21833",
            "source_model": "Handcrafted color/shape/texture/keypoint/GIST features + LightGBM",
            "source_parameter_note": (
                "Paper lists the model family but not a full LightGBM hyperparameter table; "
                "this run uses fixed compact LightGBM parameters recorded in applied_params."
            ),
            "parameter_source": "paper_model_family_with_explicit_local_defaults",
            "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F6_bof"],
            "feature_mapping_note": (
                "Matches the paper's handcrafted groups closely: color 1039d, shape 12d, "
                "texture 30d, SIFT/ORB BoF 256d, GIST 64d."
            ),
            "model_label": "LightGBM",
            "scaling_family": "boosting",
            "use_sample_weight": False,
            "estimator": LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
        },
        {
            "candidate_id": "nguyen_2026_handcrafted_xgboost",
            "source_short": "Nguyen et al. 2026",
            "source_title": "Towards Accurate and Efficient Waste Image Classification",
            "source_url": "https://arxiv.org/abs/2510.21833",
            "source_model": "Handcrafted color/shape/texture/keypoint/GIST features + XGBoost",
            "source_parameter_note": (
                "Paper lists the model family but not a full XGBoost hyperparameter table; "
                "this run uses fixed compact XGBoost parameters recorded in applied_params."
            ),
            "parameter_source": "paper_model_family_with_explicit_local_defaults",
            "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F6_bof"],
            "feature_mapping_note": (
                "Matches the paper's handcrafted groups closely; no deep features are used."
            ),
            "model_label": "XGBoost",
            "scaling_family": "boosting",
            "use_sample_weight": True,
            "estimator": XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=1.0,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            ),
        },
        {
            "candidate_id": "mboli_ogungbemi_2026_rgb_pca_random_forest",
            "source_short": "Mboli & Ogungbemi 2026",
            "source_title": "AI-Enabled Waste Classification as a Data-Driven Decision Support Tool",
            "source_url": "https://arxiv.org/abs/2601.22418",
            "source_model": "Image-array features + PCA + Random Forest",
            "source_parameter_note": (
                "Paper compares traditional models with and without PCA but does not publish "
                "a full RF parameter table; this run fixes PCA=100 and RF parameters locally."
            ),
            "parameter_source": "paper_model_family_with_explicit_local_defaults",
            "feature_ids": ["F0_pixels_rgb"],
            "feature_mapping_note": (
                "Uses the local 32x32 RGB pixel feature as a lightweight analogue of resized image arrays; "
                "class space is adapted from binary organic/recyclable to this dataset's 10 classes."
            ),
            "model_label": "PCA_RandomForest",
            "scaling_family": "pca_forest",
            "use_sample_weight": False,
            "estimator": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=100,
                            svd_solver="randomized",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=180,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        },
        {
            "candidate_id": "mboli_ogungbemi_2026_rgb_pca_rbf_svm",
            "source_short": "Mboli & Ogungbemi 2026",
            "source_title": "AI-Enabled Waste Classification as a Data-Driven Decision Support Tool",
            "source_url": "https://arxiv.org/abs/2601.22418",
            "source_model": "Image-array features + PCA + SVM",
            "source_parameter_note": (
                "Paper compares SVM with and without PCA but does not publish full SVM hyperparameters; "
                "this run uses sklearn's standard RBF SVC defaults after PCA."
            ),
            "parameter_source": "paper_model_family_with_explicit_local_defaults",
            "feature_ids": ["F0_pixels_rgb"],
            "feature_mapping_note": (
                "Uses the local 32x32 RGB pixel feature as a lightweight analogue of resized image arrays; "
                "class space is adapted from binary organic/recyclable to this dataset's 10 classes."
            ),
            "model_label": "PCA_RBF_SVM",
            "scaling_family": "kernel",
            "use_sample_weight": False,
            "estimator": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=100,
                            svd_solver="randomized",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            decision_function_shape="ovr",
                            cache_size=1000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        },
        {
            "candidate_id": "sen_2025_hog_lbp_linear_svm_reference",
            "source_short": "Sen et al. 2025",
            "source_title": "Feature Engineering is Not Dead",
            "source_url": "https://arxiv.org/abs/2507.13772",
            "source_model": "HOG + LBP-style texture fusion + SVM",
            "source_parameter_note": (
                "This is a traditional image-classification reference rather than a waste-only paper; "
                "the paper uses HOG/LBP with SVM grid search, while this run fixes a balanced linear SVM."
            ),
            "parameter_source": "traditional_reference_with_explicit_local_defaults",
            "feature_ids": ["F3_hog", "F2_texture"],
            "feature_mapping_note": (
                "Uses local HOG plus the local texture cache, whose LBP part is included alongside GLCM."
            ),
            "model_label": "HOG_Texture_LinearSVM_SGD",
            "scaling_family": "linear",
            "use_sample_weight": False,
            "estimator": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SGDClassifier(
                            loss="hinge",
                            alpha=0.0001,
                            class_weight="balanced",
                            max_iter=2000,
                            tol=1e-3,
                            n_jobs=-1,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        },
    ]


def estimator_params(estimator: Any) -> dict[str, Any]:
    """Return estimator parameters suitable for reporting."""
    params = estimator.get_params(deep=True)
    compact: dict[str, Any] = {}
    for key, value in params.items():
        if key in {"memory", "steps", "verbose"}:
            continue
        if "__" not in key and hasattr(value, "get_params"):
            compact[key] = value.__class__.__name__
        else:
            compact[key] = value
    return compact


def stratified_subset_indices(y: np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    """Select a stratified subset of row indices."""
    if len(y) <= max_samples:
        return np.arange(len(y))
    indices = np.arange(len(y))
    selected, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )
    return np.sort(selected)


def fit_with_optional_sample_weight(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_sample_weight: bool,
) -> None:
    """Fit an estimator, adding balanced sample weights only when requested."""
    if not use_sample_weight:
        estimator.fit(X_train, y_train)
        return
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    estimator.fit(X_train, y_train, sample_weight=sample_weight)


def estimate_runtime(
    candidate: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    pilot_samples: int,
    pilot_eval_samples: int,
    skip_pilot_estimate: bool,
) -> dict[str, Any]:
    """Estimate full runtime from a small stratified pilot run."""
    n_train = len(y_train)
    n_validation = len(y_validation)
    if skip_pilot_estimate:
        return {
            "pilot_status": "skipped",
            "pilot_train_samples": 0,
            "pilot_eval_samples": 0,
            "pilot_train_seconds": float("nan"),
            "pilot_predict_seconds": float("nan"),
            "estimated_train_seconds": float("nan"),
            "estimated_predict_seconds": float("nan"),
            "estimated_total_seconds": float("nan"),
            "estimation_note": "pilot timing skipped by CLI option",
        }

    train_indices = stratified_subset_indices(y_train, pilot_samples, RANDOM_SEED)
    eval_indices = stratified_subset_indices(y_validation, pilot_eval_samples, RANDOM_SEED + 1)
    X_pilot_train = X_train[train_indices]
    y_pilot_train = y_train[train_indices]
    X_pilot_eval = X_validation[eval_indices]

    estimator = clone(candidate["estimator"])
    train_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit_with_optional_sample_weight(
            estimator,
            X_pilot_train,
            y_pilot_train,
            bool(candidate["use_sample_weight"]),
        )
    pilot_train_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    estimator.predict(X_pilot_eval)
    pilot_predict_seconds = time.perf_counter() - predict_start

    train_ratio = n_train / len(y_pilot_train)
    eval_ratio = n_validation / len(eval_indices)
    scaling_family = candidate["scaling_family"]
    if scaling_family == "kernel":
        train_scale = train_ratio**2
        predict_scale = train_ratio * eval_ratio
    elif scaling_family in {"boosting", "pca_forest"}:
        train_scale = train_ratio * (
            math.log2(max(n_train, 3)) / math.log2(max(len(y_pilot_train), 3))
        )
        predict_scale = eval_ratio
    else:
        train_scale = train_ratio
        predict_scale = eval_ratio

    estimated_train_seconds = pilot_train_seconds * train_scale
    estimated_predict_seconds = pilot_predict_seconds * predict_scale
    return {
        "pilot_status": "ok",
        "pilot_train_samples": int(len(y_pilot_train)),
        "pilot_eval_samples": int(len(eval_indices)),
        "pilot_train_seconds": round(pilot_train_seconds, 4),
        "pilot_predict_seconds": round(pilot_predict_seconds, 4),
        "estimated_train_seconds": round(estimated_train_seconds, 4),
        "estimated_predict_seconds": round(estimated_predict_seconds, 4),
        "estimated_total_seconds": round(estimated_train_seconds + estimated_predict_seconds, 4),
        "estimation_note": f"scaling_family={scaling_family}",
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    prefix: str,
) -> dict[str, float]:
    """Compute aggregate metrics and selected minority-class recalls."""
    labels = list(range(len(label_names)))
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        f"{prefix}_macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        f"{prefix}_macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        f"{prefix}_weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }
    for label in MINORITY_LABELS:
        label_index = label_names.index(label)
        metrics[f"{prefix}_recall_{label}"] = float(per_class_recall[label_index])
    metrics[f"{prefix}_minority_recall_mean"] = float(
        np.mean([metrics[f"{prefix}_recall_{label}"] for label in MINORITY_LABELS])
    )
    return {key: float(value) for key, value in metrics.items()}


def per_class_rows(
    candidate: dict[str, Any],
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Build per-class precision/recall/F1 rows."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    for label in label_names:
        item = report[label]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_short": candidate["source_short"],
                "model_label": candidate["model_label"],
                "split": split,
                "label": label,
                "precision": float(item["precision"]),
                "recall": float(item["recall"]),
                "f1_score": float(item["f1-score"]),
                "support": int(item["support"]),
            }
        )
    return rows


def confusion_rows(
    candidate: dict[str, Any],
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Build long-form confusion matrix rows."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
    )
    rows = []
    for true_index, true_label in enumerate(label_names):
        for pred_index, pred_label in enumerate(label_names):
            count = int(matrix[true_index, pred_index])
            if count == 0:
                continue
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_short": candidate["source_short"],
                    "model_label": candidate["model_label"],
                    "split": split,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                }
            )
    return rows


def prediction_rows(
    candidate: dict[str, Any],
    split: str,
    image_ids: np.ndarray,
    paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Build per-image classification output rows."""
    rows = []
    for image_id, path, true_index, pred_index in zip(image_ids, paths, y_true, y_pred, strict=True):
        true_label = label_names[int(true_index)]
        pred_label = label_names[int(pred_index)]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_short": candidate["source_short"],
                "model_label": candidate["model_label"],
                "split": split,
                "image_id": str(image_id),
                "path": str(path),
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": true_label == pred_label,
            }
        )
    return rows


def warning_summary(caught: list[warnings.WarningMessage]) -> str:
    """Return de-duplicated warning text for reports."""
    messages: list[str] = []
    for item in caught:
        message = str(item.message).replace("\n", " ")
        if message not in messages:
            messages.append(message)
    return " | ".join(messages[:6])


def run_candidate(
    candidate: dict[str, Any],
    include_test: bool,
    pilot_samples: int,
    pilot_eval_samples: int,
    skip_pilot_estimate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Train and evaluate one candidate pipeline."""
    bundle = load_feature_bundle(candidate["feature_ids"], include_test=include_test)
    label_names = bundle["label_names"]
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_validation = bundle["X_validation"]
    y_validation = bundle["y_validation"]

    estimate = estimate_runtime(
        candidate,
        X_train,
        y_train,
        X_validation,
        y_validation,
        pilot_samples=pilot_samples,
        pilot_eval_samples=pilot_eval_samples,
        skip_pilot_estimate=skip_pilot_estimate,
    )
    log(
        "    estimate: train="
        f"{format_seconds(estimate['estimated_train_seconds'])}, "
        f"validation_predict={format_seconds(estimate['estimated_predict_seconds'])}, "
        f"pilot={estimate['pilot_train_samples']} train / {estimate['pilot_eval_samples']} eval"
    )

    estimator = clone(candidate["estimator"])
    train_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_with_optional_sample_weight(
            estimator,
            X_train,
            y_train,
            bool(candidate["use_sample_weight"]),
        )
    train_seconds = time.perf_counter() - train_start
    warnings_text = warning_summary(caught)
    log(f"    fitted in {format_seconds(train_seconds)}")

    validation_predict_start = time.perf_counter()
    y_validation_pred = estimator.predict(X_validation)
    validation_predict_seconds = time.perf_counter() - validation_predict_start
    log(f"    validation predicted in {format_seconds(validation_predict_seconds)}")

    result: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "source_short": candidate["source_short"],
        "source_title": candidate["source_title"],
        "source_url": candidate["source_url"],
        "source_model": candidate["source_model"],
        "source_parameter_note": candidate["source_parameter_note"],
        "parameter_source": candidate["parameter_source"],
        "model_label": candidate["model_label"],
        "feature_ids": ";".join(candidate["feature_ids"]),
        "feature_descriptions": " | ".join(bundle["feature_descriptions"]),
        "feature_mapping_note": candidate["feature_mapping_note"],
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_validation.shape[0]),
        "test_samples": int(bundle["X_test"].shape[0]) if include_test else 0,
        "applied_params": params_text(estimator_params(candidate["estimator"])),
        "warnings": warnings_text,
        "actual_train_seconds": round(train_seconds, 4),
        "validation_predict_seconds": round(validation_predict_seconds, 4),
        "status": "ok",
    }
    result.update(estimate)
    result.update(evaluate_predictions(y_validation, y_validation_pred, label_names, "validation"))

    per_class = per_class_rows(candidate, "validation", y_validation, y_validation_pred, label_names)
    confusion = confusion_rows(candidate, "validation", y_validation, y_validation_pred, label_names)
    predictions = prediction_rows(
        candidate,
        "validation",
        bundle["image_ids_validation"],
        bundle["paths_validation"],
        y_validation,
        y_validation_pred,
        label_names,
    )
    prediction_arrays = {
        candidate["candidate_id"]: y_validation_pred.astype(np.int64, copy=False),
    }

    if include_test:
        X_test = bundle["X_test"]
        y_test = bundle["y_test"]
        test_predict_start = time.perf_counter()
        y_test_pred = estimator.predict(X_test)
        test_predict_seconds = time.perf_counter() - test_predict_start
        result["test_predict_seconds"] = round(test_predict_seconds, 4)
        result.update(evaluate_predictions(y_test, y_test_pred, label_names, "test"))
        per_class.extend(per_class_rows(candidate, "test", y_test, y_test_pred, label_names))
        confusion.extend(confusion_rows(candidate, "test", y_test, y_test_pred, label_names))
        predictions.extend(
            prediction_rows(
                candidate,
                "test",
                bundle["image_ids_test"],
                bundle["paths_test"],
                y_test,
                y_test_pred,
                label_names,
            )
        )
        prediction_arrays[f"{candidate['candidate_id']}__test"] = y_test_pred.astype(np.int64, copy=False)

    prediction_arrays["y_validation"] = y_validation.astype(np.int64, copy=False)
    prediction_arrays["label_names"] = np.asarray(label_names)
    return result, per_class, predictions, confusion, prediction_arrays


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_checkpoint(
    results: list[dict[str, Any]],
    per_class_rows_all: list[dict[str, Any]],
    prediction_rows_all: list[dict[str, Any]],
    confusion_rows_all: list[dict[str, Any]],
    prediction_arrays: dict[str, np.ndarray],
    candidate_configs: list[dict[str, Any]],
    include_test: bool,
    dependency_versions: dict[str, str],
) -> None:
    """Write all current report artifacts after each candidate."""
    result_fields = [
        "candidate_id",
        "source_short",
        "source_title",
        "source_url",
        "source_model",
        "source_parameter_note",
        "parameter_source",
        "model_label",
        "feature_ids",
        "feature_descriptions",
        "feature_mapping_note",
        "feature_dim",
        "train_samples",
        "validation_samples",
        "test_samples",
        "pilot_status",
        "pilot_train_samples",
        "pilot_eval_samples",
        "pilot_train_seconds",
        "pilot_predict_seconds",
        "estimated_train_seconds",
        "estimated_predict_seconds",
        "estimated_total_seconds",
        "estimation_note",
        "actual_train_seconds",
        "validation_predict_seconds",
        "test_predict_seconds",
        "validation_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_weighted_f1",
        "validation_recall_trash",
        "validation_recall_battery",
        "validation_recall_biological",
        "validation_minority_recall_mean",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_f1",
        "test_recall_trash",
        "test_recall_battery",
        "test_recall_biological",
        "test_minority_recall_mean",
        "applied_params",
        "warnings",
        "status",
        "error",
    ]
    write_csv(RESULTS_CSV, results, result_fields)
    write_csv(
        PER_CLASS_CSV,
        per_class_rows_all,
        [
            "candidate_id",
            "source_short",
            "model_label",
            "split",
            "label",
            "precision",
            "recall",
            "f1_score",
            "support",
        ],
    )
    write_csv(
        PREDICTIONS_CSV,
        prediction_rows_all,
        [
            "candidate_id",
            "source_short",
            "model_label",
            "split",
            "image_id",
            "path",
            "true_label",
            "predicted_label",
            "correct",
        ],
    )
    write_csv(
        CONFUSION_CSV,
        confusion_rows_all,
        [
            "candidate_id",
            "source_short",
            "model_label",
            "split",
            "true_label",
            "predicted_label",
            "count",
        ],
    )
    if prediction_arrays:
        np.savez_compressed(PREDICTIONS_NPZ, **prediction_arrays)

    ok_results = [row for row in results if row.get("status") == "ok"]
    output = {
        "purpose": "Literature-inspired traditional ML reproductions; not a core-model comparison.",
        "test_usage": (
            "test split included only because --include-test was passed"
            if include_test
            else "test split is sealed; only validation predictions/metrics are reported"
        ),
        "random_seed": RANDOM_SEED,
        "outputs": {
            "results_csv": str(RESULTS_CSV.relative_to(PROJECT_ROOT)),
            "per_class_csv": str(PER_CLASS_CSV.relative_to(PROJECT_ROOT)),
            "predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
            "confusion_csv": str(CONFUSION_CSV.relative_to(PROJECT_ROOT)),
            "predictions_npz": str(PREDICTIONS_NPZ.relative_to(PROJECT_ROOT)),
            "dependency_versions_json": str(DEPENDENCY_JSON.relative_to(PROJECT_ROOT)),
        },
        "dependency_versions": dependency_versions,
        "candidate_configs": [
            {
                key: jsonable(value)
                for key, value in candidate.items()
                if key != "estimator"
            }
            | {"applied_params": estimator_params(candidate["estimator"])}
            for candidate in candidate_configs
        ],
        "summary": {
            "num_candidates_planned": len(candidate_configs),
            "num_finished": len(results),
            "num_ok": len(ok_results),
            "num_error": len(results) - len(ok_results),
            "finished_candidate_ids": [row["candidate_id"] for row in results],
        },
        "results": results,
    }
    SUMMARY_JSON.write_text(
        json.dumps(jsonable(output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run literature-inspired traditional ML candidates on validation split.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Run only matching candidate_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also evaluate on the sealed test split. Off by default.",
    )
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=1200,
        help="Stratified pilot train size for runtime estimation.",
    )
    parser.add_argument(
        "--pilot-eval-samples",
        type=int,
        default=360,
        help="Stratified pilot validation size for runtime estimation.",
    )
    parser.add_argument(
        "--skip-pilot-estimate",
        action="store_true",
        help="Skip pilot timing and print unknown estimates.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Load candidates and print timing estimates without full training.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected candidates and write reports."""
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dependency_versions = write_dependency_versions()

    label_probe = load_feature_cache("F6_sift_bof")
    label_names = label_probe["label_names"]
    candidate_configs = make_candidate_configs(num_classes=len(label_names))
    if args.candidate:
        selected = set(args.candidate)
        candidate_configs = [
            candidate for candidate in candidate_configs if candidate["candidate_id"] in selected
        ]
        missing = selected - {candidate["candidate_id"] for candidate in candidate_configs}
        if missing:
            raise ValueError(f"Unknown candidate_id(s): {sorted(missing)}")

    log("Literature-inspired traditional ML reproduction")
    log(f"  candidates: {len(candidate_configs)}")
    log(f"  labels: {', '.join(label_names)}")
    log(
        "  test split: "
        + ("included because --include-test was passed" if args.include_test else "sealed by default")
    )
    log("  estimating runtime before each full fit...")

    results: list[dict[str, Any]] = []
    per_class_rows_all: list[dict[str, Any]] = []
    prediction_rows_all: list[dict[str, Any]] = []
    confusion_rows_all: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}

    for index, candidate in enumerate(candidate_configs, start=1):
        log("")
        log(
            f"[{index}/{len(candidate_configs)}] {candidate['candidate_id']} "
            f"({candidate['source_short']}, {candidate['model_label']})"
        )
        log(f"    features: {', '.join(candidate['feature_ids'])}")

        if args.estimate_only:
            bundle = load_feature_bundle(candidate["feature_ids"], include_test=args.include_test)
            estimate = estimate_runtime(
                candidate,
                bundle["X_train"],
                bundle["y_train"],
                bundle["X_validation"],
                bundle["y_validation"],
                pilot_samples=args.pilot_samples,
                pilot_eval_samples=args.pilot_eval_samples,
                skip_pilot_estimate=args.skip_pilot_estimate,
            )
            result = {
                "candidate_id": candidate["candidate_id"],
                "source_short": candidate["source_short"],
                "source_title": candidate["source_title"],
                "source_url": candidate["source_url"],
                "source_model": candidate["source_model"],
                "source_parameter_note": candidate["source_parameter_note"],
                "parameter_source": candidate["parameter_source"],
                "model_label": candidate["model_label"],
                "feature_ids": ";".join(candidate["feature_ids"]),
                "feature_descriptions": " | ".join(bundle["feature_descriptions"]),
                "feature_mapping_note": candidate["feature_mapping_note"],
                "feature_dim": int(bundle["X_train"].shape[1]),
                "train_samples": int(bundle["X_train"].shape[0]),
                "validation_samples": int(bundle["X_validation"].shape[0]),
                "test_samples": int(bundle["X_test"].shape[0]) if args.include_test else 0,
                "applied_params": params_text(estimator_params(candidate["estimator"])),
                "status": "estimate_only",
            }
            result.update(estimate)
            results.append(result)
            log(
                "    estimate only: train="
                f"{format_seconds(estimate['estimated_train_seconds'])}, "
                f"validation_predict={format_seconds(estimate['estimated_predict_seconds'])}"
            )
            continue

        try:
            result, per_class, predictions, confusion, arrays = run_candidate(
                candidate,
                include_test=args.include_test,
                pilot_samples=args.pilot_samples,
                pilot_eval_samples=args.pilot_eval_samples,
                skip_pilot_estimate=args.skip_pilot_estimate,
            )
            results.append(result)
            per_class_rows_all.extend(per_class)
            prediction_rows_all.extend(predictions)
            confusion_rows_all.extend(confusion)
            prediction_arrays.update(arrays)
            log(
                "    validation: "
                f"accuracy={result['validation_accuracy']:.4f}, "
                f"macro_f1={result['validation_macro_f1']:.4f}, "
                f"minority_recall={result['validation_minority_recall_mean']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001 - keep a failed candidate row in the report.
            log(f"    ERROR: {exc!r}")
            results.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_short": candidate["source_short"],
                    "source_title": candidate["source_title"],
                    "source_url": candidate["source_url"],
                    "source_model": candidate["source_model"],
                    "source_parameter_note": candidate["source_parameter_note"],
                    "parameter_source": candidate["parameter_source"],
                    "model_label": candidate["model_label"],
                    "feature_ids": ";".join(candidate["feature_ids"]),
                    "feature_mapping_note": candidate["feature_mapping_note"],
                    "applied_params": params_text(estimator_params(candidate["estimator"])),
                    "status": "error",
                    "error": repr(exc),
                }
            )

        write_checkpoint(
            results,
            per_class_rows_all,
            prediction_rows_all,
            confusion_rows_all,
            prediction_arrays,
            candidate_configs,
            include_test=args.include_test,
            dependency_versions=dependency_versions,
        )
        log(f"    checkpoint written: {RESULTS_CSV.relative_to(PROJECT_ROOT)}")

    if args.estimate_only:
        write_checkpoint(
            results,
            per_class_rows_all,
            prediction_rows_all,
            confusion_rows_all,
            prediction_arrays,
            candidate_configs,
            include_test=args.include_test,
            dependency_versions=dependency_versions,
        )
        log("")
        log(f"Estimate report written: {RESULTS_CSV}")
        return

    ok_results = [row for row in results if row.get("status") == "ok"]
    log("")
    log("Finished literature-inspired traditional ML runs.")
    for row in ok_results:
        log(
            f"  {row['candidate_id']}: "
            f"validation_macro_f1={row['validation_macro_f1']:.4f}, "
            f"validation_accuracy={row['validation_accuracy']:.4f}, "
            f"train={format_seconds(row['actual_train_seconds'])}"
        )
    log(f"Results CSV: {RESULTS_CSV}")
    log(f"Per-image validation predictions: {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
