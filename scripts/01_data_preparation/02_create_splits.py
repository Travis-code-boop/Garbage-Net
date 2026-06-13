"""Create fixed stratified train/validation/test splits.

Notebook block purpose:
    Fixed data split. This code reads the image metadata produced in the
    previous step, performs a deterministic per-class stratified split, and
    saves the split assignment. All later feature extraction, feature
    selection, model screening, tuning, and final evaluation must reuse this
    split so experiments remain comparable and the test set stays sealed.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "metadata"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

METADATA_PATH = METADATA_DIR / "metadata.csv"
METADATA_WITH_SPLIT_PATH = METADATA_DIR / "metadata_with_split.csv"
SPLIT_INDICES_PATH = SPLIT_DIR / "split_indices.json"
SPLIT_SUMMARY_PATH = SPLIT_DIR / "split_summary.csv"
SPLIT_SUMMARY_JSON_PATH = SPLIT_DIR / "split_summary.json"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def read_metadata(path: Path) -> list[dict[str, str]]:
    """Read metadata rows and keep only readable images."""
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row["is_readable"] == "True"]


def split_class_rows(rows: list[dict[str, str]], rng: random.Random) -> dict[str, list[str]]:
    """Split one class into train/validation/test using deterministic shuffling."""
    shuffled = rows[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_val = round(total * VAL_RATIO)
    n_test = round(total * TEST_RATIO)
    n_train = total - n_val - n_test

    train_rows = shuffled[:n_train]
    val_rows = shuffled[n_train : n_train + n_val]
    test_rows = shuffled[n_train + n_val :]

    return {
        "train": [row["image_id"] for row in train_rows],
        "validation": [row["image_id"] for row in val_rows],
        "test": [row["image_id"] for row in test_rows],
    }


def build_splits(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Build full dataset split indices with stratification by label."""
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["label"]].append(row)

    rng = random.Random(RANDOM_SEED)
    split_indices = {"train": [], "validation": [], "test": []}

    for label in sorted(rows_by_label):
        class_split = split_class_rows(rows_by_label[label], rng)
        for split_name in split_indices:
            split_indices[split_name].extend(class_split[split_name])

    for split_name in split_indices:
        split_indices[split_name].sort()

    return split_indices


def attach_split(rows: list[dict[str, str]], split_indices: dict[str, list[str]]) -> list[dict[str, str]]:
    """Add the split column to each metadata row."""
    split_by_id = {}
    for split_name, image_ids in split_indices.items():
        for image_id in image_ids:
            split_by_id[image_id] = split_name

    rows_with_split = []
    for row in rows:
        updated = dict(row)
        updated["split"] = split_by_id[updated["image_id"]]
        rows_with_split.append(updated)
    return rows_with_split


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable column order."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows_with_split: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize split counts by split and label."""
    labels = sorted({row["label"] for row in rows_with_split})
    splits = ["train", "validation", "test"]

    counts = Counter((row["split"], row["label"]) for row in rows_with_split)
    split_totals = Counter(row["split"] for row in rows_with_split)
    label_totals = Counter(row["label"] for row in rows_with_split)

    summary_rows: list[dict[str, object]] = []
    for label in labels:
        label_total = label_totals[label]
        row = {
            "label": label,
            "total": label_total,
        }
        for split_name in splits:
            count = counts[(split_name, label)]
            row[f"{split_name}_count"] = count
            row[f"{split_name}_ratio"] = round(count / label_total, 4)
        summary_rows.append(row)

    summary_json = {
        "random_seed": RANDOM_SEED,
        "target_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "split_totals": dict(split_totals),
        "label_totals": dict(sorted(label_totals.items())),
        "per_label": summary_rows,
    }
    return summary_rows, summary_json


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(METADATA_PATH)
    split_indices = build_splits(rows)
    rows_with_split = attach_split(rows, split_indices)
    summary_rows, summary_json = summarize(rows_with_split)

    metadata_columns = list(rows_with_split[0].keys())
    write_csv(METADATA_WITH_SPLIT_PATH, rows_with_split, metadata_columns)

    SPLIT_INDICES_PATH.write_text(
        json.dumps(split_indices, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_columns = [
        "label",
        "total",
        "train_count",
        "train_ratio",
        "validation_count",
        "validation_ratio",
        "test_count",
        "test_ratio",
    ]
    write_csv(SPLIT_SUMMARY_PATH, summary_rows, summary_columns)
    SPLIT_SUMMARY_JSON_PATH.write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
