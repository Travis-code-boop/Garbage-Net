"""Stage 2: single feature groups with four baseline classifiers.

Notebook block purpose:
    Stage 2 single-feature screening. This code loads each planned single
    feature group C0-C6, trains four baseline classifiers on the train split,
    evaluates on the validation split, and writes a comparison report. The test
    split is intentionally not used here.

Baseline models:
    Logistic Regression
    KNN
    Linear SVM trained with SGD
    Decision Tree
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATH = PROJECT_ROOT / "vendor" / "python"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))

import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
REPORT_CSV_PATH = REPORT_DIR / "stage2_single_feature_baselines.csv"
REPORT_JSON_PATH = REPORT_DIR / "stage2_single_feature_baselines.json"

RANDOM_SEED = 42
MINORITY_LABELS = ("trash", "battery", "biological")

FEATURE_CONFIGS = [
    {
        "combo_id": "C0",
        "feature_id": "F0_pixels_gray",
        "description": "gray pixel baseline 1024d",
        "path": PROJECT_ROOT / "data" / "features" / "F0_pixels" / "F0_pixels_gray_1024.npz",
    },
    {
        "combo_id": "C1",
        "feature_id": "F1_color",
        "description": "color features 1039d",
        "path": PROJECT_ROOT / "data" / "features" / "F1_color" / "F1_color_1039.npz",
    },
    {
        "combo_id": "C2",
        "feature_id": "F2_texture",
        "description": "texture features 30d",
        "path": PROJECT_ROOT / "data" / "features" / "F2_texture" / "F2_texture_30.npz",
    },
    {
        "combo_id": "C3",
        "feature_id": "F3_hog",
        "description": "HOG features 1764d",
        "path": PROJECT_ROOT / "data" / "features" / "F3_hog" / "F3_hog_1764.npz",
    },
    {
        "combo_id": "C4",
        "feature_id": "F4_shape",
        "description": "shape features 12d",
        "path": PROJECT_ROOT / "data" / "features" / "F4_shape" / "F4_shape_12.npz",
    },
    {
        "combo_id": "C5",
        "feature_id": "F5_gist",
        "description": "GIST features 64d",
        "path": PROJECT_ROOT / "data" / "features" / "F5_gist" / "F5_gist_64.npz",
    },
    {
        "combo_id": "C6",
        "feature_id": "F6_bof",
        "description": "SIFT+ORB BoF features 256d",
        "path": PROJECT_ROOT / "data" / "features" / "F6_bof" / "F6_bof_256.npz",
    },
]


def make_models() -> dict[str, object]:
    """Create fresh baseline model instances."""
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
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
        ),
        "LinearSVM_SGD": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="hinge",
                        alpha=0.0001,
                        max_iter=1000,
                        tol=1e-3,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    }


def load_feature_cache(path: Path) -> dict[str, object]:
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


def model_params_summary(model_name: str) -> str:
    """Return compact parameter text for report readability."""
    if model_name == "LogisticRegression":
        return "StandardScaler + LogisticRegression(C=1, solver=lbfgs, max_iter=1000)"
    if model_name == "KNN":
        return "StandardScaler + KNN(k=5, weights=distance, euclidean)"
    if model_name == "LinearSVM_SGD":
        return "StandardScaler + SGDClassifier(loss=hinge, alpha=0.0001, max_iter=1000)"
    if model_name == "DecisionTree":
        return "DecisionTree(default, random_state=42)"
    return ""


def run_one_experiment(config: dict[str, object], model_name: str, model: object) -> dict[str, object]:
    """Train one model on one feature cache and evaluate on validation split."""
    data = load_feature_cache(Path(config["path"]))
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_validation = data["X_validation"]
    y_validation = data["y_validation"]
    label_names = data["label_names"]

    result: dict[str, object] = {
        "combo_id": config["combo_id"],
        "feature_id": config["feature_id"],
        "description": config["description"],
        "model": model_name,
        "model_params": model_params_summary(model_name),
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_validation.shape[0]),
        "status": "ok",
    }

    start_train = time.perf_counter()
    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
            warning_messages = [str(warning.message) for warning in caught_warnings]
        train_seconds = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        y_pred = model.predict(X_validation)
        predict_seconds = time.perf_counter() - start_predict

        result.update(evaluate_predictions(y_validation, y_pred, label_names))
        result["train_seconds"] = round(train_seconds, 4)
        result["predict_seconds"] = round(predict_seconds, 4)
        result["warnings"] = " | ".join(warning_messages)
    except Exception as exc:  # noqa: BLE001 - stage report should keep failed rows.
        result["status"] = "error"
        result["error"] = repr(exc)
        result["train_seconds"] = round(time.perf_counter() - start_train, 4)
        result["predict_seconds"] = ""
        result["warnings"] = " | ".join(warning_messages)

    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write stage 2 results to CSV."""
    fieldnames = [
        "combo_id",
        "feature_id",
        "description",
        "model",
        "model_params",
        "feature_dim",
        "train_samples",
        "validation_samples",
        "status",
        "validation_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_weighted_f1",
        "recall_trash",
        "recall_battery",
        "recall_biological",
        "minority_recall_mean",
        "train_seconds",
        "predict_seconds",
        "warnings",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build compact ranking summary."""
    ok_rows = [row for row in rows if row["status"] == "ok"]
    ranked = sorted(ok_rows, key=lambda row: row["validation_macro_f1"], reverse=True)

    best_by_feature = {}
    for row in ok_rows:
        feature_id = str(row["feature_id"])
        if (
            feature_id not in best_by_feature
            or row["validation_macro_f1"] > best_by_feature[feature_id]["validation_macro_f1"]
        ):
            best_by_feature[feature_id] = row

    best_by_model = {}
    for row in ok_rows:
        model_name = str(row["model"])
        if (
            model_name not in best_by_model
            or row["validation_macro_f1"] > best_by_model[model_name]["validation_macro_f1"]
        ):
            best_by_model[model_name] = row

    return {
        "num_experiments": len(rows),
        "num_ok": len(ok_rows),
        "num_error": len(rows) - len(ok_rows),
        "top_10_by_macro_f1": ranked[:10],
        "best_by_feature": best_by_feature,
        "best_by_model": best_by_model,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []

    for config in FEATURE_CONFIGS:
        print(f"loading feature {config['combo_id']} {config['feature_id']}", flush=True)
        for model_name, model in make_models().items():
            print(f"  training {model_name}", flush=True)
            result = run_one_experiment(config, model_name, model)
            all_results.append(result)
            if result["status"] == "ok":
                print(
                    "    macro_f1="
                    f"{result['validation_macro_f1']:.4f}, "
                    f"minority_recall={result['minority_recall_mean']:.4f}, "
                    f"train_s={result['train_seconds']}",
                    flush=True,
                )
            else:
                print(f"    ERROR: {result.get('error')}", flush=True)

    write_csv(REPORT_CSV_PATH, all_results)
    summary = build_summary(all_results)
    report = {
        "report_csv": str(REPORT_CSV_PATH.relative_to(PROJECT_ROOT)),
        "feature_configs": [
            {
                key: str(value.relative_to(PROJECT_ROOT)) if key == "path" else value
                for key, value in config.items()
            }
            for config in FEATURE_CONFIGS
        ],
        "models": list(make_models().keys()),
        "minority_labels": MINORITY_LABELS,
        "test_usage": "test split is not used in stage 2",
        "summary": summary,
        "results": all_results,
    }
    REPORT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
