"""Stage 4: tune strong boosting models on selected fusion candidates.

Notebook block purpose:
    Stage 4 strong-model tuning. This block tunes LightGBM and XGBoost on a
    small shortlist selected from Stage 3. It uses only the train split for
    parameter search via an internal stratified holdout, then refits the best
    parameters on the full train split and evaluates once on validation. The
    test split is intentionally not used.

Tuning strategy:
    1. Load cached handcrafted feature groups and validate alignment.
    2. Build selected Stage 4 feature combinations.
    3. Run compact randomized searches with early stopping inside train only.
    4. Refit the best trial on full train and evaluate on validation.
    5. Save trial and final candidate checkpoints so long runs can resume.
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
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
SEARCH_CSV = REPORT_DIR / "stage4_boosting_tuning_search.csv"
FINAL_CSV = REPORT_DIR / "stage4_boosting_tuning_final.csv"
BEST_BY_MODEL_CSV = REPORT_DIR / "stage4_boosting_best_by_model.csv"
SHORTLIST_CSV = REPORT_DIR / "stage4_boosting_shortlist.csv"
SUMMARY_JSON = REPORT_DIR / "stage4_boosting_tuning_summary.json"
DEPENDENCY_JSON = REPORT_DIR / "stage4_dependency_versions.json"


FEATURE_CONFIGS = {
    "F1_color": PROJECT_ROOT / "data" / "features" / "F1_color" / "F1_color_1039.npz",
    "F2_texture": PROJECT_ROOT / "data" / "features" / "F2_texture" / "F2_texture_30.npz",
    "F3_hog": PROJECT_ROOT / "data" / "features" / "F3_hog" / "F3_hog_1764.npz",
    "F4_shape": PROJECT_ROOT / "data" / "features" / "F4_shape" / "F4_shape_12.npz",
    "F5_gist": PROJECT_ROOT / "data" / "features" / "F5_gist" / "F5_gist_64.npz",
    "F6_bof": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_bof_256.npz",
}


CANDIDATE_CONFIGS = [
    {
        "candidate_id": "LGBM_A_C11_full",
        "model": "LightGBM",
        "combo_label": "A_C11_full",
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog", "F6_bof"],
        "stage3_macro_f1": 0.8096497497,
        "stage3_minority_recall_mean": 0.8157689034,
        "reason": "Stage 3 best validation macro-F1.",
    },
    {
        "candidate_id": "LGBM_B_no_shape",
        "model": "LightGBM",
        "combo_label": "B_no_shape",
        "feature_ids": ["F3_hog", "F1_color", "F2_texture", "F5_gist", "F6_bof"],
        "stage3_macro_f1": 0.8071305522,
        "stage3_minority_recall_mean": 0.8108669426,
        "reason": "Nearly tied with C11 while removing weak F4_shape.",
    },
    {
        "candidate_id": "LGBM_C_hog_color_texture",
        "model": "LightGBM",
        "combo_label": "C_hog_color_texture",
        "feature_ids": ["F3_hog", "F1_color", "F2_texture"],
        "stage3_macro_f1": 0.8011968050,
        "stage3_minority_recall_mean": 0.8005701396,
        "reason": "Compact main HOG+Color+Texture candidate.",
    },
    {
        "candidate_id": "LGBM_D_lowdim_color_texture_gist",
        "model": "LightGBM",
        "combo_label": "D_lowdim_color_texture_gist",
        "feature_ids": ["F1_color", "F2_texture", "F5_gist"],
        "stage3_macro_f1": 0.7844679925,
        "stage3_minority_recall_mean": 0.8103740611,
        "reason": "Low-dimensional candidate with strong minority recall.",
    },
    {
        "candidate_id": "LGBM_E_no_bof",
        "model": "LightGBM",
        "combo_label": "E_no_bof",
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog"],
        "stage3_macro_f1": 0.8019770252,
        "stage3_minority_recall_mean": 0.7961176801,
        "reason": "C10-style candidate without BoF.",
    },
    {
        "candidate_id": "XGB_A_C11_full",
        "model": "XGBoost",
        "combo_label": "A_C11_full",
        "feature_ids": ["F1_color", "F2_texture", "F4_shape", "F5_gist", "F3_hog", "F6_bof"],
        "stage3_macro_f1": 0.7973916286,
        "stage3_minority_recall_mean": 0.8079170901,
        "reason": "Best XGBoost full-reference comparison.",
    },
    {
        "candidate_id": "XGB_B_no_shape",
        "model": "XGBoost",
        "combo_label": "B_no_shape",
        "feature_ids": ["F3_hog", "F1_color", "F2_texture", "F5_gist", "F6_bof"],
        "stage3_macro_f1": 0.7980302997,
        "stage3_minority_recall_mean": 0.8140415458,
        "reason": "Stage 3 best XGBoost and strong minority recall.",
    },
]


SEARCH_FIELDNAMES = [
    "trial_id",
    "candidate_id",
    "model",
    "combo_label",
    "feature_ids",
    "feature_dim",
    "trial_index",
    "inner_train_samples",
    "inner_validation_samples",
    "inner_macro_f1",
    "inner_macro_precision",
    "inner_macro_recall",
    "inner_weighted_f1",
    "inner_minority_recall_mean",
    "fit_seconds",
    "predict_seconds",
    "best_iteration",
    "sample_weight_mode",
    "params",
    "warnings",
]


FINAL_FIELDNAMES = [
    "candidate_id",
    "model",
    "combo_label",
    "feature_ids",
    "feature_dim",
    "stage3_macro_f1",
    "stage3_minority_recall_mean",
    "best_inner_macro_f1",
    "best_inner_minority_recall_mean",
    "validation_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "validation_weighted_f1",
    "minority_recall_mean",
    "macro_f1_delta_vs_stage3",
    "minority_recall_delta_vs_stage3",
    "fit_seconds",
    "predict_seconds",
    "total_seconds",
    "best_iteration",
    "sample_weight_mode",
    "best_params",
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
        "xgboost": module_version("xgboost"),
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


def parse_params(text: str) -> dict[str, Any]:
    """Read parameter JSON stored in a CSV field."""
    return json.loads(text) if text else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows if the file exists."""
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
            raise ValueError(f"Label names mismatch for {feature_id}")
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


def stratified_subsample_indices(
    y: np.ndarray,
    indices: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    """Take a reproducible stratified subsample from given indices."""
    if max_samples is None or len(indices) <= max_samples:
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=random_state,
        stratify=y[indices],
    )
    return np.asarray(selected, dtype=np.int64)


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


def lightgbm_baseline_params() -> dict[str, Any]:
    """Return the Stage 3 LightGBM parameter anchor."""
    return {
        "n_estimators": 460,
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "sample_weight_mode": None,
    }


def xgboost_baseline_params() -> dict[str, Any]:
    """Return the Stage 3 XGBoost parameter anchor."""
    return {
        "n_estimators": 360,
        "max_depth": 5,
        "learning_rate": 0.055,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "sample_weight_mode": None,
    }


def lightgbm_space() -> dict[str, list[Any]]:
    """Return compact LightGBM randomized-search space."""
    return {
        "n_estimators": [320, 460, 650, 800],
        "num_leaves": [15, 31, 63, 95],
        "max_depth": [-1, 6, 8, 10, 12],
        "learning_rate": [0.03, 0.05, 0.07],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.65, 0.75, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "sample_weight_mode": [None, "balanced"],
    }


def xgboost_space() -> dict[str, list[Any]]:
    """Return compact XGBoost randomized-search space."""
    return {
        "n_estimators": [260, 360, 500, 650],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.07],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.65, 0.75, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 2.0, 5.0],
        "sample_weight_mode": [None, "balanced"],
    }


def trial_params(model_name: str, n_iter_lgbm: int, n_iter_xgb: int) -> list[dict[str, Any]]:
    """Build deterministic baseline-plus-random trial parameters."""
    if model_name == "LightGBM":
        baseline = lightgbm_baseline_params()
        sampled = list(
            ParameterSampler(lightgbm_space(), n_iter=n_iter_lgbm, random_state=RANDOM_SEED)
        )
    elif model_name == "XGBoost":
        baseline = xgboost_baseline_params()
        sampled = list(
            ParameterSampler(xgboost_space(), n_iter=n_iter_xgb, random_state=RANDOM_SEED)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    params: list[dict[str, Any]] = [baseline]
    seen = {params_text(baseline)}
    for item in sampled:
        text = params_text(item)
        if text not in seen:
            params.append(dict(item))
            seen.add(text)
    return params


def split_model_params(params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Separate estimator parameters from sample-weight mode."""
    copied = dict(params)
    mode = copied.pop("sample_weight_mode", None)
    return copied, mode


def sample_weight_for_mode(y: np.ndarray, mode: str | None) -> np.ndarray | None:
    """Return sample weights for a class-imbalance mode."""
    if mode == "balanced":
        return compute_sample_weight(class_weight="balanced", y=y)
    if mode is None:
        return None
    raise ValueError(f"Unsupported sample_weight_mode: {mode}")


def make_model(
    model_name: str,
    params: dict[str, Any],
    num_classes: int,
    use_early_stopping: bool = True,
) -> Any:
    """Create a LightGBM or XGBoost classifier for one trial."""
    model_params, _ = split_model_params(params)
    if model_name == "LightGBM":
        return LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
            **model_params,
        )
    if model_name == "XGBoost":
        early_stopping_params = {"early_stopping_rounds": 30} if use_early_stopping else {}
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
            **early_stopping_params,
            **model_params,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def best_iteration(model: Any, model_name: str, fallback: int) -> int:
    """Return best iteration after early stopping, falling back to n_estimators."""
    if model_name == "LightGBM":
        value = getattr(model, "best_iteration_", None)
        return int(value) if value else int(fallback)
    if model_name == "XGBoost":
        value = getattr(model, "best_iteration", None)
        return int(value) + 1 if value is not None else int(fallback)
    return int(fallback)


def warning_summary(caught: list[warnings.WarningMessage]) -> str:
    """Return compact warning text."""
    messages: list[str] = []
    for item in caught:
        message = str(item.message).replace("\n", " ")
        if message not in messages:
            messages.append(message)
    return " | ".join(messages[:5])


def run_trial(
    candidate: dict[str, Any],
    params: dict[str, Any],
    trial_index: int,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    X_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Fit one internal-holdout trial and return its metrics."""
    model_name = str(candidate["model"])
    model_params, weight_mode = split_model_params(params)
    sample_weight = sample_weight_for_mode(y_inner_train, weight_mode)
    model = make_model(model_name, params, len(label_names), use_early_stopping=True)

    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_name == "LightGBM":
            model.fit(
                X_inner_train,
                y_inner_train,
                sample_weight=sample_weight,
                eval_set=[(X_inner_validation, y_inner_validation)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(30, verbose=False), log_evaluation(period=0)],
            )
        else:
            model.fit(
                X_inner_train,
                y_inner_train,
                sample_weight=sample_weight,
                eval_set=[(X_inner_validation, y_inner_validation)],
                verbose=False,
            )
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_inner_validation)
    predict_seconds = time.perf_counter() - predict_start

    metrics = evaluate_predictions(
        y_inner_validation,
        np.asarray(y_pred, dtype=np.int64),
        label_names,
        prefix="inner_",
    )
    fallback_n_estimators = int(model_params["n_estimators"])
    return {
        "trial_id": f"{candidate['candidate_id']}__trial_{trial_index:02d}",
        "candidate_id": candidate["candidate_id"],
        "model": model_name,
        "combo_label": candidate["combo_label"],
        "feature_ids": "+".join(candidate["feature_ids"]),
        "feature_dim": int(X_inner_train.shape[1]),
        "trial_index": trial_index,
        "inner_train_samples": int(len(y_inner_train)),
        "inner_validation_samples": int(len(y_inner_validation)),
        "inner_macro_f1": round(metrics["inner_macro_f1"], 10),
        "inner_macro_precision": round(metrics["inner_macro_precision"], 10),
        "inner_macro_recall": round(metrics["inner_macro_recall"], 10),
        "inner_weighted_f1": round(metrics["inner_weighted_f1"], 10),
        "inner_minority_recall_mean": round(metrics["inner_minority_recall_mean"], 10),
        "fit_seconds": round(float(fit_seconds), 4),
        "predict_seconds": round(float(predict_seconds), 4),
        "best_iteration": best_iteration(model, model_name, fallback_n_estimators),
        "sample_weight_mode": weight_mode or "",
        "params": params_text(params),
        "warnings": warning_summary(caught),
    }


def refit_and_validate(
    candidate: dict[str, Any],
    params: dict[str, Any],
    best_trial: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Refit best parameters on full train and evaluate once on validation."""
    model_name = str(candidate["model"])
    final_params = dict(params)
    model_params, weight_mode = split_model_params(final_params)
    best_iter = int(float(best_trial["best_iteration"]))
    model_params["n_estimators"] = max(1, best_iter)
    final_params = dict(model_params)
    final_params["sample_weight_mode"] = weight_mode
    sample_weight = sample_weight_for_mode(y_train, weight_mode)
    model = make_model(model_name, final_params, len(label_names), use_early_stopping=False)

    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = np.asarray(model.predict(X_validation), dtype=np.int64)
    predict_seconds = time.perf_counter() - predict_start
    metrics = evaluate_predictions(y_validation, y_pred, label_names, prefix="validation_")

    row = {
        "candidate_id": candidate["candidate_id"],
        "model": model_name,
        "combo_label": candidate["combo_label"],
        "feature_ids": "+".join(candidate["feature_ids"]),
        "feature_dim": int(X_train.shape[1]),
        "stage3_macro_f1": candidate["stage3_macro_f1"],
        "stage3_minority_recall_mean": candidate["stage3_minority_recall_mean"],
        "best_inner_macro_f1": best_trial["inner_macro_f1"],
        "best_inner_minority_recall_mean": best_trial["inner_minority_recall_mean"],
        "validation_accuracy": round(metrics["validation_accuracy"], 10),
        "validation_macro_precision": round(metrics["validation_macro_precision"], 10),
        "validation_macro_recall": round(metrics["validation_macro_recall"], 10),
        "validation_macro_f1": round(metrics["validation_macro_f1"], 10),
        "validation_weighted_f1": round(metrics["validation_weighted_f1"], 10),
        "minority_recall_mean": round(metrics["validation_minority_recall_mean"], 10),
        "macro_f1_delta_vs_stage3": round(
            metrics["validation_macro_f1"] - float(candidate["stage3_macro_f1"]),
            10,
        ),
        "minority_recall_delta_vs_stage3": round(
            metrics["validation_minority_recall_mean"]
            - float(candidate["stage3_minority_recall_mean"]),
            10,
        ),
        "fit_seconds": round(float(fit_seconds), 4),
        "predict_seconds": round(float(predict_seconds), 4),
        "total_seconds": round(float(fit_seconds + predict_seconds), 4),
        "best_iteration": model_params["n_estimators"],
        "sample_weight_mode": weight_mode or "",
        "best_params": params_text(final_params),
        "warnings": warning_summary(caught),
    }
    for label in MINORITY_LABELS:
        row[f"recall_{label}"] = round(metrics[f"validation_recall_{label}"], 10)
        row[f"f1_{label}"] = round(metrics[f"validation_f1_{label}"], 10)
    return row


def as_float(row: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    """Read a numeric field safely."""
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_final_rows(rows: list[dict[str, Any]]) -> None:
    """Write Stage 4 tuning summary files."""
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda row: as_float(row, "validation_macro_f1"), reverse=True)

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
    write_csv(BEST_BY_MODEL_CSV, best_by_model, FINAL_FIELDNAMES)

    best_macro = as_float(sorted_rows[0], "validation_macro_f1")
    shortlist: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        reasons: list[str] = []
        if rank <= 5:
            reasons.append("top5_stage4_macro_f1")
        if as_float(row, "validation_macro_f1") >= best_macro - 0.01:
            reasons.append("within_0.01_of_stage4_best")
        if as_float(row, "macro_f1_delta_vs_stage3") > 0:
            reasons.append("improves_stage3_candidate")
        if as_float(row, "minority_recall_delta_vs_stage3") > 0:
            reasons.append("improves_stage3_minority_recall")
        copied = dict(row)
        copied["rank"] = rank
        copied["shortlist_reasons"] = "; ".join(reasons)
        shortlist.append(copied)
    write_csv(SHORTLIST_CSV, shortlist, ["rank", "shortlist_reasons", *FINAL_FIELDNAMES])

    summary = {
        "num_completed_candidates": len(rows),
        "best_candidate": sorted_rows[0],
        "best_by_model": best_by_model,
        "outputs": {
            "search_csv": str(SEARCH_CSV.relative_to(PROJECT_ROOT)),
            "final_csv": str(FINAL_CSV.relative_to(PROJECT_ROOT)),
            "best_by_model_csv": str(BEST_BY_MODEL_CSV.relative_to(PROJECT_ROOT)),
            "shortlist_csv": str(SHORTLIST_CSV.relative_to(PROJECT_ROOT)),
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, ensure_ascii=False, indent=2)


def run_candidate(
    candidate: dict[str, Any],
    feature_store: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    existing_trials: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Run or resume tuning for one candidate and return final validation row."""
    feature_ids = list(candidate["feature_ids"])
    X_train_raw = build_combo_matrix(feature_store, feature_ids, "train")
    X_validation_raw = build_combo_matrix(feature_store, feature_ids, "validation")
    y_train = feature_store[feature_ids[0]]["y_train"]
    y_validation = feature_store[feature_ids[0]]["y_validation"]
    label_names = feature_store[feature_ids[0]]["label_names"]

    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train_raw)
    X_validation = selector.transform(X_validation_raw)

    inner_train_idx, inner_validation_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )
    inner_train_idx = stratified_subsample_indices(
        y_train,
        inner_train_idx,
        args.max_inner_train_samples,
        RANDOM_SEED,
    )
    inner_validation_idx = stratified_subsample_indices(
        y_train,
        inner_validation_idx,
        args.max_inner_validation_samples,
        RANDOM_SEED + 1,
    )

    X_inner_train = X_train[inner_train_idx]
    y_inner_train = y_train[inner_train_idx]
    X_inner_validation = X_train[inner_validation_idx]
    y_inner_validation = y_train[inner_validation_idx]

    params_list = trial_params(
        str(candidate["model"]),
        args.n_iter_lgbm,
        args.n_iter_xgb,
    )

    log(
        f"[candidate] {candidate['candidate_id']} | model={candidate['model']} | "
        f"dim={X_train.shape[1]} | trials={len(params_list)} | "
        f"inner_train={len(y_inner_train)} | inner_val={len(y_inner_validation)}"
    )
    trial_rows: list[dict[str, Any]] = []
    for trial_index, params in enumerate(params_list, start=1):
        trial_id = f"{candidate['candidate_id']}__trial_{trial_index:02d}"
        if args.resume and trial_id in existing_trials:
            row = dict(existing_trials[trial_id])
            trial_rows.append(row)
            log(f"[skip] {trial_id} already in search checkpoint.")
            continue

        log(f"[trial {trial_index}/{len(params_list)}] {trial_id} starting")
        trial_start = time.perf_counter()
        row = run_trial(
            candidate,
            params,
            trial_index,
            X_inner_train,
            y_inner_train,
            X_inner_validation,
            y_inner_validation,
            label_names,
        )
        append_csv_row(SEARCH_CSV, row, SEARCH_FIELDNAMES)
        existing_trials[trial_id] = {key: str(row.get(key, "")) for key in SEARCH_FIELDNAMES}
        trial_rows.append(row)
        log(
            f"[trial done] {trial_id} | inner_macro_f1={float(row['inner_macro_f1']):.4f} | "
            f"inner_minority={float(row['inner_minority_recall_mean']):.4f} | "
            f"time={time.perf_counter() - trial_start:.1f}s | "
            f"best_iter={row['best_iteration']}"
        )

    best_trial = max(
        trial_rows,
        key=lambda row: (
            as_float(row, "inner_macro_f1"),
            as_float(row, "inner_minority_recall_mean"),
        ),
    )
    best_params = parse_params(str(best_trial["params"]))

    log(
        f"[refit] {candidate['candidate_id']} best trial {best_trial['trial_id']} | "
        f"inner_macro_f1={float(best_trial['inner_macro_f1']):.4f} | "
        "refitting on full train"
    )
    final_row = refit_and_validate(
        candidate,
        best_params,
        best_trial,
        X_train,
        y_train,
        X_validation,
        y_validation,
        label_names,
    )
    append_csv_row(FINAL_CSV, final_row, FINAL_FIELDNAMES)
    log(
        f"[candidate done] {candidate['candidate_id']} | "
        f"validation_macro_f1={float(final_row['validation_macro_f1']):.4f} | "
        f"minority={float(final_row['minority_recall_mean']):.4f} | "
        f"delta={float(final_row['macro_f1_delta_vs_stage3']):+.4f}"
    )
    return final_row


def parse_args() -> argparse.Namespace:
    """Parse Stage 4 tuning CLI options."""
    parser = argparse.ArgumentParser(description="Tune Stage 4 boosting candidates.")
    parser.add_argument("--resume", action="store_true", help="Resume completed trials/candidates.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit candidates.")
    parser.add_argument(
        "--candidates",
        default="",
        help="Comma-separated candidate IDs to run. Empty means all candidates.",
    )
    parser.add_argument("--n-iter-lgbm", type=int, default=8, help="Random LightGBM trials.")
    parser.add_argument("--n-iter-xgb", type=int, default=6, help="Random XGBoost trials.")
    parser.add_argument("--max-inner-train-samples", type=int, default=5200)
    parser.add_argument("--max-inner-validation-samples", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    """Run Stage 4 boosting tuning."""
    args = parse_args()
    write_dependency_versions()
    feature_store = load_all_features()

    selected_ids = {item.strip() for item in args.candidates.split(",") if item.strip()}
    candidates = [
        candidate
        for candidate in CANDIDATE_CONFIGS
        if not selected_ids or candidate["candidate_id"] in selected_ids
    ]
    if selected_ids:
        found = {candidate["candidate_id"] for candidate in candidates}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Unknown candidate IDs: {missing}")
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    existing_final = {
        row["candidate_id"]: row for row in read_csv_rows(FINAL_CSV)
    } if args.resume else {}
    existing_trials = {
        row["trial_id"]: row for row in read_csv_rows(SEARCH_CSV)
    } if args.resume else {}

    pending = [
        candidate
        for candidate in candidates
        if not args.resume or candidate["candidate_id"] not in existing_final
    ]
    if not args.resume:
        write_csv(SEARCH_CSV, [], SEARCH_FIELDNAMES)
        write_csv(FINAL_CSV, [], FINAL_FIELDNAMES)

    total_trials = sum(
        len(trial_params(candidate["model"], args.n_iter_lgbm, args.n_iter_xgb))
        for candidate in pending
    )
    log(
        f"[plan] Stage 4 boosting tuning: {len(candidates)} selected candidates, "
        f"{len(pending)} pending candidates, {total_trials} pending search trials."
    )
    log(
        "[estimate] First-pass tuning is expected to take roughly 60-120 minutes. "
        "Every trial and final candidate is checkpointed; use --resume after interruption."
    )

    start = time.perf_counter()
    final_rows = list(read_csv_rows(FINAL_CSV)) if args.resume else []
    for index, candidate in enumerate(pending, start=1):
        log(f"[candidate {index}/{len(pending)}] starting {candidate['candidate_id']}")
        final_rows.append(run_candidate(candidate, feature_store, args, existing_trials))

    final_rows = read_csv_rows(FINAL_CSV)
    summarize_final_rows(final_rows)
    log(f"[complete] Stage 4 boosting tuning invocation finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
