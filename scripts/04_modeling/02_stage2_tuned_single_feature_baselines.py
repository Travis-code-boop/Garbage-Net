"""Stage 2 tuned screening for selected single-feature baseline models.

Notebook block purpose:
    This block refines Stage 2 without an exhaustive grid search. It first
    selects promising single-feature/model pairs from the fixed-parameter
    Stage 2 report, then uses small randomized searches inside the training
    split only. The best parameters are refit on the full training split and
    evaluated once on validation. The test split is intentionally not used.

Tuning strategy:
    1. Read fixed-parameter Stage 2 results.
    2. Keep only promising or minority-helpful feature/model pairs.
    3. Run StratifiedKFold CV on train with RandomizedSearchCV.
    4. Refit the best pipeline on full train.
    5. Evaluate macro-F1, minority recall, and time on validation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATH = PROJECT_ROOT / "vendor" / "python"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))

import numpy as np  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (  # noqa: E402
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
BASELINE_JSON_PATH = REPORT_DIR / "stage2_single_feature_baselines.json"

TUNED_CANDIDATE_SELECTION_CSV = REPORT_DIR / "stage2_tuned_candidate_selection.csv"
TUNED_RESULTS_CSV = REPORT_DIR / "stage2_tuned_single_feature_baselines.csv"
TUNED_RESULTS_JSON = REPORT_DIR / "stage2_tuned_single_feature_baselines.json"
TUNED_CV_DETAILS_CSV = REPORT_DIR / "stage2_tuned_cv_details.csv"
TUNED_BEST_BY_FEATURE_CSV = REPORT_DIR / "stage2_tuned_best_by_feature.csv"
TUNED_BEST_BY_MODEL_CSV = REPORT_DIR / "stage2_tuned_best_by_model.csv"
TUNED_COMPLEMENTARITY_CSV = REPORT_DIR / "stage2_tuned_feature_complementarity.csv"
TUNED_VALIDATION_PREDICTIONS_NPZ = REPORT_DIR / "stage2_tuned_validation_predictions.npz"


def load_feature_cache(path: Path) -> dict[str, Any]:
    """Load train and validation data from one feature cache."""
    cache = np.load(path, allow_pickle=False)
    return {
        "X_train": cache["X_train"],
        "y_train": cache["y_train"],
        "X_validation": cache["X_validation"],
        "y_validation": cache["y_validation"],
        "label_names": cache["label_names"].tolist(),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    """Compute validation metrics and selected minority-class recalls."""
    labels = list(range(len(label_names)))
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics = {
        "validation_accuracy": accuracy_score(y_true, y_pred),
        "validation_macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "validation_macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "validation_macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "validation_weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    for label in MINORITY_LABELS:
        label_index = label_names.index(label)
        metrics[f"recall_{label}"] = float(per_class_recall[label_index])

    metrics["minority_recall_mean"] = float(
        np.mean([metrics[f"recall_{label}"] for label in MINORITY_LABELS])
    )
    return {key: float(value) for key, value in metrics.items()}


def feature_config_map(baseline_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return feature configs with absolute cache paths."""
    configs = {}
    for config in baseline_report["feature_configs"]:
        copied = dict(config)
        copied["path"] = PROJECT_ROOT / config["path"]
        configs[config["feature_id"]] = copied
    return configs


def select_tuning_candidates(
    baseline_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select promising fixed-parameter pairs and record skipped reasons."""
    sorted_results = sorted(
        baseline_results,
        key=lambda row: row.get("validation_macro_f1", -1),
        reverse=True,
    )
    top8_keys = {(row["feature_id"], row["model"]) for row in sorted_results[:8]}
    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for row in sorted_results:
        key = (row["feature_id"], row["model"])
        macro_f1 = float(row.get("validation_macro_f1", 0.0))
        minority = float(row.get("minority_recall_mean", 0.0))
        train_seconds = float(row.get("train_seconds", 0.0))
        reasons: list[str] = []

        if key in top8_keys:
            reasons.append("top8_macro_f1")
        if macro_f1 >= 0.40:
            reasons.append("macro_f1_at_least_0.40")
        if minority >= 0.48 and macro_f1 >= 0.35:
            reasons.append("minority_recall_helpful")
        if row["feature_id"] == "F6_bof" and row["model"] == "KNN":
            reasons.append("bof_knn_histogram_candidate")
        if row["feature_id"] == "F2_texture" and row["model"] == "DecisionTree":
            reasons.append("low_dim_tree_candidate")

        excluded_reasons: list[str] = []
        if row["feature_id"] == "F4_shape":
            excluded_reasons.append("shape_feature_best_fixed_result_is_weak")
        if row["model"] == "LinearSVM_SGD" and train_seconds > 15 and key not in top8_keys:
            excluded_reasons.append("linear_svm_fixed_run_too_slow_for_gain")

        is_selected = bool(reasons) and not excluded_reasons
        audit_rows.append(
            {
                "combo_id": row["combo_id"],
                "feature_id": row["feature_id"],
                "model": row["model"],
                "fixed_validation_macro_f1": macro_f1,
                "fixed_minority_recall_mean": minority,
                "fixed_train_seconds": train_seconds,
                "selected_for_tuning": is_selected,
                "selection_reasons": "; ".join(reasons),
                "exclusion_reasons": "; ".join(excluded_reasons),
            }
        )
        if is_selected:
            selected.append(row)

    return selected, audit_rows


def make_pipeline(model_name: str) -> Pipeline | DecisionTreeClassifier:
    """Create an untuned estimator for one selected model family."""
    if model_name == "LogisticRegression":
        return Pipeline(
            steps=[
                ("prep", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if model_name == "KNN":
        return Pipeline(
            steps=[
                ("prep", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "LinearSVM_SGD":
        return Pipeline(
            steps=[
                ("prep", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="hinge",
                        alpha=0.0001,
                        max_iter=3000,
                        tol=1e-3,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=RANDOM_SEED)
    raise ValueError(f"Unsupported model: {model_name}")


def parameter_space(model_name: str, feature_id: str) -> tuple[dict[str, list[Any]], int]:
    """Return a compact randomized-search space and iteration budget."""
    standard = StandardScaler()
    robust = RobustScaler()
    minmax = MinMaxScaler()
    l2norm = Normalizer(norm="l2")

    if model_name == "LogisticRegression":
        prep_options: list[Any] = [standard, robust, minmax]
        if feature_id == "F6_bof":
            prep_options.append(l2norm)
        return (
            {
                "prep": prep_options,
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__class_weight": [None, "balanced"],
            },
            12,
        )

    if model_name == "KNN":
        prep_options = [standard, robust, minmax, l2norm]
        if feature_id in {"F2_texture", "F5_gist"}:
            k_values = [1, 3, 5, 7, 9, 11, 15, 21]
        else:
            k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
        return (
            {
                "prep": prep_options,
                "model__n_neighbors": k_values,
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan", "cosine"],
            },
            14,
        )

    if model_name == "LinearSVM_SGD":
        return (
            {
                "prep": [standard, robust],
                "model__loss": ["hinge"],
                "model__penalty": ["l2"],
                "model__alpha": [3e-5, 1e-4, 3e-4, 1e-3],
                "model__class_weight": [None, "balanced"],
            },
            6,
        )

    if model_name == "DecisionTree":
        return (
            {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [5, 8, 12, 16, 20, 30, None],
                "min_samples_split": [2, 5, 10, 20, 50],
                "min_samples_leaf": [1, 2, 5, 10, 20, 40],
                "max_features": [None, "sqrt", "log2", 0.5],
                "class_weight": [None, "balanced"],
                "ccp_alpha": [0.0, 0.0001, 0.001],
            },
            18,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def finite_space_size(param_space: dict[str, list[Any]]) -> int:
    """Count combinations when every parameter value is listed explicitly."""
    count = 1
    for values in param_space.values():
        count *= len(values)
    return count


def choose_search_strategy(model_name: str, feature_id: str, feature_dim: int) -> dict[str, Any]:
    """Choose CV or train-internal holdout based on expected search cost."""
    if model_name == "KNN" and feature_dim >= 1000:
        return {
            "search_strategy": "train_internal_holdout_subsample",
            "cv_folds": 1,
            "max_inner_train_samples": 2600,
            "max_inner_validation_samples": 1200,
            "reason": "high-dimensional KNN distance search is expensive",
        }
    if model_name == "KNN" and feature_id == "F6_bof":
        return {
            "search_strategy": "train_internal_holdout",
            "cv_folds": 1,
            "max_inner_train_samples": None,
            "max_inner_validation_samples": None,
            "reason": "BoF histogram KNN is retained mainly as a complementarity check",
        }
    return {
        "search_strategy": "stratified_3fold_cv",
        "cv_folds": 3,
        "max_inner_train_samples": None,
        "max_inner_validation_samples": None,
        "reason": "candidate is tractable enough for 3-fold CV",
    }


def jsonable(value: Any) -> Any:
    """Convert parameters and numpy values to JSON-friendly values."""
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


def unique_warning_texts(caught_warnings: list[warnings.WarningMessage]) -> str:
    """Return a short de-duplicated warning summary."""
    messages: list[str] = []
    for item in caught_warnings:
        message = str(item.message).replace("\n", " ")
        if message not in messages:
            messages.append(message)
    return " | ".join(messages[:5])


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Take a reproducible stratified subset, staying inside the train split."""
    if max_samples is None or len(y) <= max_samples:
        return X, y
    indices = np.arange(len(y))
    selected, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )
    return X[selected], y[selected]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_holdout_random_search(
    estimator: Any,
    param_space: dict[str, list[Any]],
    n_iter: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: dict[str, Any],
) -> tuple[dict[str, Any], float, float, list[dict[str, Any]], str, float]:
    """Run randomized search on an internal train-only holdout split."""
    train_indices, validation_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )
    X_inner_train = X_train[train_indices]
    y_inner_train = y_train[train_indices]
    X_inner_validation = X_train[validation_indices]
    y_inner_validation = y_train[validation_indices]

    X_inner_train, y_inner_train = stratified_subsample(
        X_inner_train,
        y_inner_train,
        strategy["max_inner_train_samples"],
        RANDOM_SEED,
    )
    X_inner_validation, y_inner_validation = stratified_subsample(
        X_inner_validation,
        y_inner_validation,
        strategy["max_inner_validation_samples"],
        RANDOM_SEED + 1,
    )

    sampled_params = list(
        ParameterSampler(
            param_space,
            n_iter=n_iter,
            random_state=RANDOM_SEED,
        )
    )
    warning_messages: list[str] = []
    search_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = float("-inf")
    search_start = time.perf_counter()

    for params in sampled_params:
        candidate_estimator = clone(estimator)
        candidate_estimator.set_params(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            candidate_estimator.fit(X_inner_train, y_inner_train)
            y_inner_pred = candidate_estimator.predict(X_inner_validation)
        score = float(f1_score(y_inner_validation, y_inner_pred, average="macro"))

        for item in caught:
            message = str(item.message).replace("\n", " ")
            if message not in warning_messages:
                warning_messages.append(message)

        search_rows.append(
            {
                "candidate_id": "",
                "feature_id": "",
                "model": "",
                "rank_within_candidate": "",
                "sklearn_rank": "",
                "mean_cv_macro_f1": score,
                "std_cv_macro_f1": "",
                "params": params_text(params),
                "search_strategy": strategy["search_strategy"],
                "inner_train_samples": int(len(y_inner_train)),
                "inner_validation_samples": int(len(y_inner_validation)),
            }
        )
        if score > best_score:
            best_score = score
            best_params = params

    search_seconds = time.perf_counter() - search_start
    if best_params is None:
        raise RuntimeError("Holdout random search did not evaluate any parameters.")

    ranked_rows = sorted(
        search_rows,
        key=lambda row: float(row["mean_cv_macro_f1"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_within_candidate"] = rank
        row["sklearn_rank"] = rank

    return (
        best_params,
        best_score,
        float("nan"),
        ranked_rows[:10],
        " | ".join(warning_messages[:5]),
        search_seconds,
    )


def run_tuned_candidate(
    candidate: dict[str, Any],
    config: dict[str, Any],
    cv: StratifiedKFold,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Tune one selected pair, refit on train, and evaluate on validation."""
    data = load_feature_cache(Path(config["path"]))
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_validation = data["X_validation"]
    y_validation = data["y_validation"]
    label_names = data["label_names"]

    model_name = candidate["model"]
    estimator = make_pipeline(model_name)
    param_space, requested_iter = parameter_space(model_name, candidate["feature_id"])
    n_iter = min(requested_iter, finite_space_size(param_space))
    strategy = choose_search_strategy(model_name, candidate["feature_id"], int(X_train.shape[1]))

    if strategy["search_strategy"] == "stratified_3fold_cv":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_SEED,
            refit=False,
            n_jobs=1,
            error_score=np.nan,
            return_train_score=False,
        )

        search_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", FitFailedWarning)
            search.fit(X_train, y_train)
        search_seconds = time.perf_counter() - search_start
        warning_summary = unique_warning_texts(caught)

        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
        best_index = int(search.best_index_)
        cv_std = float(search.cv_results_["std_test_score"][best_index])

        cv_rows: list[dict[str, Any]] = []
        means = search.cv_results_["mean_test_score"]
        stds = search.cv_results_["std_test_score"]
        ranks = search.cv_results_["rank_test_score"]
        params = search.cv_results_["params"]
        ordered_indices = sorted(
            range(len(means)),
            key=lambda idx: (
                int(ranks[idx]),
                -float(means[idx]) if not np.isnan(means[idx]) else float("-inf"),
            ),
        )
        for rank_position, idx in enumerate(ordered_indices[:10], start=1):
            cv_rows.append(
                {
                    "candidate_id": "",
                    "feature_id": "",
                    "model": "",
                    "rank_within_candidate": rank_position,
                    "sklearn_rank": int(ranks[idx]),
                    "mean_cv_macro_f1": float(means[idx]) if not np.isnan(means[idx]) else "",
                    "std_cv_macro_f1": float(stds[idx]) if not np.isnan(stds[idx]) else "",
                    "params": params_text(params[idx]),
                    "search_strategy": strategy["search_strategy"],
                    "inner_train_samples": "",
                    "inner_validation_samples": "",
                }
            )
    else:
        (
            best_params,
            best_cv_score,
            cv_std,
            cv_rows,
            warning_summary,
            search_seconds,
        ) = run_holdout_random_search(
            estimator,
            param_space,
            n_iter,
            X_train,
            y_train,
            strategy,
        )

    final_estimator = clone(estimator)
    final_estimator.set_params(**best_params)

    train_start = time.perf_counter()
    final_estimator.fit(X_train, y_train)
    train_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = final_estimator.predict(X_validation)
    predict_seconds = time.perf_counter() - predict_start

    metrics = evaluate_predictions(y_validation, y_pred, label_names)
    result: dict[str, Any] = {
        "candidate_id": f'{candidate["combo_id"]}_{candidate["model"]}',
        "combo_id": candidate["combo_id"],
        "feature_id": candidate["feature_id"],
        "description": candidate["description"],
        "model": model_name,
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_validation.shape[0]),
        "fixed_validation_macro_f1": float(candidate["validation_macro_f1"]),
        "fixed_minority_recall_mean": float(candidate["minority_recall_mean"]),
        "n_iter": n_iter,
        "cv_folds": strategy["cv_folds"],
        "search_strategy": strategy["search_strategy"],
        "search_strategy_reason": strategy["reason"],
        "cv_macro_f1_mean": best_cv_score,
        "cv_macro_f1_std": cv_std,
        "best_params": params_text(best_params),
        "search_seconds": round(search_seconds, 4),
        "final_train_seconds": round(train_seconds, 4),
        "predict_seconds": round(predict_seconds, 4),
        "warnings": warning_summary,
    }
    result.update(metrics)

    for row in cv_rows:
        row["candidate_id"] = result["candidate_id"]
        row["feature_id"] = candidate["feature_id"]
        row["model"] = model_name

    return result, cv_rows, y_pred


def best_rows_by_key(
    rows: list[dict[str, Any]],
    key_name: str,
    score_name: str = "validation_macro_f1",
) -> dict[str, dict[str, Any]]:
    """Select the best row for each key by validation macro-F1."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row[key_name]
        if key not in best or float(row[score_name]) > float(best[key][score_name]):
            best[key] = row
    return best


def compute_complementarity(
    tuned_rows: list[dict[str, Any]],
    validation_predictions: dict[str, np.ndarray],
    y_validation: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Compare tuned validation predictions pairwise for fusion planning."""
    minority_indices = {label_names.index(label) for label in MINORITY_LABELS}
    rows: list[dict[str, Any]] = []
    by_id = {row["candidate_id"]: row for row in tuned_rows}

    for left_id, right_id in combinations(validation_predictions.keys(), 2):
        left_pred = validation_predictions[left_id]
        right_pred = validation_predictions[right_id]
        left_correct = left_pred == y_validation
        right_correct = right_pred == y_validation
        left_wrong_right_correct = (~left_correct) & right_correct
        right_wrong_left_correct = (~right_correct) & left_correct
        minority_mask = np.array([label in minority_indices for label in y_validation])

        rows.append(
            {
                "left_candidate_id": left_id,
                "left_feature_id": by_id[left_id]["feature_id"],
                "left_model": by_id[left_id]["model"],
                "right_candidate_id": right_id,
                "right_feature_id": by_id[right_id]["feature_id"],
                "right_model": by_id[right_id]["model"],
                "prediction_disagreement_rate": float(np.mean(left_pred != right_pred)),
                "left_wrong_right_correct_count": int(np.sum(left_wrong_right_correct)),
                "right_wrong_left_correct_count": int(np.sum(right_wrong_left_correct)),
                "left_wrong_right_correct_minority_count": int(
                    np.sum(left_wrong_right_correct & minority_mask)
                ),
                "right_wrong_left_correct_minority_count": int(
                    np.sum(right_wrong_left_correct & minority_mask)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["left_wrong_right_correct_minority_count"]
            + row["right_wrong_left_correct_minority_count"],
            row["prediction_disagreement_rate"],
        ),
        reverse=True,
    )
    return rows


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_report = json.loads(BASELINE_JSON_PATH.read_text(encoding="utf-8"))
    configs = feature_config_map(baseline_report)

    selected_candidates, selection_audit = select_tuning_candidates(baseline_report["results"])
    write_csv(
        TUNED_CANDIDATE_SELECTION_CSV,
        selection_audit,
        [
            "combo_id",
            "feature_id",
            "model",
            "fixed_validation_macro_f1",
            "fixed_minority_recall_mean",
            "fixed_train_seconds",
            "selected_for_tuning",
            "selection_reasons",
            "exclusion_reasons",
        ],
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    tuned_rows: list[dict[str, Any]] = []
    cv_detail_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}
    y_validation_reference: np.ndarray | None = None
    label_names_reference: list[str] | None = None

    print(f"Selected candidates for tuning: {len(selected_candidates)}", flush=True)
    for index, candidate in enumerate(selected_candidates, start=1):
        feature_id = candidate["feature_id"]
        model_name = candidate["model"]
        config = configs[feature_id]
        quick_data = load_feature_cache(Path(config["path"]))
        quick_strategy = choose_search_strategy(
            model_name,
            feature_id,
            int(quick_data["X_train"].shape[1]),
        )
        print(
            f"[{index}/{len(selected_candidates)}] "
            f"{candidate['combo_id']} {feature_id} + {model_name} "
            f"({quick_strategy['search_strategy']})",
            flush=True,
        )
        result, cv_rows, y_pred = run_tuned_candidate(candidate, config, cv)
        tuned_rows.append(result)
        cv_detail_rows.extend(cv_rows)
        validation_predictions[result["candidate_id"]] = y_pred

        cache = load_feature_cache(Path(config["path"]))
        if y_validation_reference is None:
            y_validation_reference = cache["y_validation"]
            label_names_reference = cache["label_names"]

        print(
            "  fixed_macro_f1="
            f"{result['fixed_validation_macro_f1']:.4f}, "
            f"cv_macro_f1={result['cv_macro_f1_mean']:.4f}, "
            f"val_macro_f1={result['validation_macro_f1']:.4f}, "
            f"minority_mean={result['minority_recall_mean']:.4f}, "
            f"search_seconds={result['search_seconds']}, "
            f"final_train_seconds={result['final_train_seconds']}",
            flush=True,
        )

    tuned_rows.sort(key=lambda row: row["validation_macro_f1"], reverse=True)
    best_by_feature = best_rows_by_key(tuned_rows, "feature_id")
    best_by_model = best_rows_by_key(tuned_rows, "model")

    result_fields = [
        "candidate_id",
        "combo_id",
        "feature_id",
        "description",
        "model",
        "feature_dim",
        "train_samples",
        "validation_samples",
        "fixed_validation_macro_f1",
        "fixed_minority_recall_mean",
        "n_iter",
        "cv_folds",
        "search_strategy",
        "search_strategy_reason",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "validation_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_weighted_f1",
        "recall_trash",
        "recall_battery",
        "recall_biological",
        "minority_recall_mean",
        "search_seconds",
        "final_train_seconds",
        "predict_seconds",
        "best_params",
        "warnings",
    ]
    write_csv(TUNED_RESULTS_CSV, tuned_rows, result_fields)
    write_csv(
        TUNED_CV_DETAILS_CSV,
        cv_detail_rows,
        [
            "candidate_id",
            "feature_id",
            "model",
            "rank_within_candidate",
            "sklearn_rank",
            "mean_cv_macro_f1",
            "std_cv_macro_f1",
            "params",
            "search_strategy",
            "inner_train_samples",
            "inner_validation_samples",
        ],
    )

    write_csv(
        TUNED_BEST_BY_FEATURE_CSV,
        sorted(best_by_feature.values(), key=lambda row: row["validation_macro_f1"], reverse=True),
        result_fields,
    )
    write_csv(
        TUNED_BEST_BY_MODEL_CSV,
        sorted(best_by_model.values(), key=lambda row: row["validation_macro_f1"], reverse=True),
        result_fields,
    )

    complementarity_rows: list[dict[str, Any]] = []
    if y_validation_reference is not None and label_names_reference is not None:
        complementarity_rows = compute_complementarity(
            tuned_rows,
            validation_predictions,
            y_validation_reference,
            label_names_reference,
        )
        write_csv(
            TUNED_COMPLEMENTARITY_CSV,
            complementarity_rows,
            [
                "left_candidate_id",
                "left_feature_id",
                "left_model",
                "right_candidate_id",
                "right_feature_id",
                "right_model",
                "prediction_disagreement_rate",
                "left_wrong_right_correct_count",
                "right_wrong_left_correct_count",
                "left_wrong_right_correct_minority_count",
                "right_wrong_left_correct_minority_count",
            ],
        )
        np.savez_compressed(
            TUNED_VALIDATION_PREDICTIONS_NPZ,
            candidate_ids=np.array(list(validation_predictions.keys())),
            y_validation=y_validation_reference,
            label_names=np.array(label_names_reference),
            **validation_predictions,
        )

    feature_recommendations = sorted(
        best_by_feature.values(),
        key=lambda row: (
            row["validation_macro_f1"],
            row["minority_recall_mean"],
            -row["feature_dim"],
        ),
        reverse=True,
    )

    output = {
        "candidate_selection_csv": str(TUNED_CANDIDATE_SELECTION_CSV.relative_to(PROJECT_ROOT)),
        "results_csv": str(TUNED_RESULTS_CSV.relative_to(PROJECT_ROOT)),
        "cv_details_csv": str(TUNED_CV_DETAILS_CSV.relative_to(PROJECT_ROOT)),
        "best_by_feature_csv": str(TUNED_BEST_BY_FEATURE_CSV.relative_to(PROJECT_ROOT)),
        "best_by_model_csv": str(TUNED_BEST_BY_MODEL_CSV.relative_to(PROJECT_ROOT)),
        "complementarity_csv": str(TUNED_COMPLEMENTARITY_CSV.relative_to(PROJECT_ROOT)),
        "validation_predictions_npz": str(TUNED_VALIDATION_PREDICTIONS_NPZ.relative_to(PROJECT_ROOT)),
        "test_usage": "test split is not used in stage 2 tuning",
        "selection_rule": (
            "Tune top fixed macro-F1 pairs, pairs with fixed macro-F1 >= 0.40, "
            "minority-helpful pairs, plus selected BoF/texture candidates; "
            "exclude shape-only weak pairs and very slow weak linear-SVM pairs."
        ),
        "summary": {
            "num_fixed_pairs": len(baseline_report["results"]),
            "num_selected_for_tuning": len(selected_candidates),
            "num_tuned_ok": len(tuned_rows),
            "top_10_by_validation_macro_f1": tuned_rows[:10],
            "best_by_feature": best_by_feature,
            "best_by_model": best_by_model,
            "feature_stage3_recommendations": feature_recommendations,
            "top_10_complementarity_pairs": complementarity_rows[:10],
        },
        "results": tuned_rows,
        "cv_details_top10_each_candidate": cv_detail_rows,
        "candidate_selection": selection_audit,
    }
    TUNED_RESULTS_JSON.write_text(
        json.dumps(jsonable(output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTop tuned results by validation macro-F1:", flush=True)
    for rank, row in enumerate(tuned_rows[:10], start=1):
        print(
            f"{rank:02d}. {row['candidate_id']} "
            f"val_macro_f1={row['validation_macro_f1']:.4f} "
            f"minority={row['minority_recall_mean']:.4f} "
            f"cv={row['cv_macro_f1_mean']:.4f}",
            flush=True,
        )

    print("\nBest tuned result by feature:", flush=True)
    for row in feature_recommendations:
        print(
            f"{row['feature_id']}: {row['model']} "
            f"val_macro_f1={row['validation_macro_f1']:.4f} "
            f"minority={row['minority_recall_mean']:.4f}",
            flush=True,
        )

    print(f"\nWrote {TUNED_RESULTS_CSV}", flush=True)
    print(f"Wrote {TUNED_RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
