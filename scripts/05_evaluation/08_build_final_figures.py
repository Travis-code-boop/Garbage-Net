"""Build final evaluation figures without external plotting dependencies.

Notebook block purpose:
    Create report-ready SVG figures from the fixed final test evaluation
    outputs: model comparison, final stacking confusion matrix, and per-class
    F1/recall bars. The script intentionally avoids matplotlib so it can run
    with the current project dependencies.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports"
FIGURE_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
FINAL_MODEL_RESULTS_CSV = REPORT_DIR / "final_model_results.csv"
FINAL_BASE_RESULTS_CSV = REPORT_DIR / "final_base_results.csv"
FINAL_CONFUSION_CSV = REPORT_DIR / "final_confusion_matrices.csv"
FINAL_PER_CLASS_CSV = REPORT_DIR / "final_per_class_metrics.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short_model_name(model_id: str) -> str:
    """Return a compact display name."""
    names = {
        "final_stack_small_plus_lgbm_anchor": "Stacking final",
        "final_stack_small_only": "Stacking small",
        "final_avg_small_plus_lgbm_anchor_proba": "Avg + anchor",
        "final_avg_small_proba": "Avg small",
        "lgbm_anchor_c11_sqrt": "LGBM C11",
        "lgbm_small_hct_sqrt": "LGBM HCT",
        "rf_lowdim_balanced": "RF lowdim",
        "lr_c11_balanced": "LR C11",
        "sgd_svm_hct_balanced": "Linear SVM",
        "knn_compact_gist_texture_shape": "KNN compact",
    }
    return names.get(model_id, model_id)


def text(x: float, y: float, value: str, size: int = 12, fill: str = "#111827", anchor: str = "start") -> str:
    """Return one SVG text element."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none") -> str:
    """Return one SVG rect element."""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" />'
    )


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    """Write an SVG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def blue_scale(value: int, max_value: int) -> str:
    """Return a blue heatmap color."""
    if max_value <= 0:
        return "#EFF6FF"
    ratio = value / max_value
    light = int(245 - ratio * 145)
    mid = int(250 - ratio * 110)
    dark = int(255 - ratio * 35)
    return f"#{light:02x}{mid:02x}{dark:02x}"


def final_label_order() -> list[str]:
    """Return class order from final per-class metrics."""
    return [
        row["class_name"]
        for row in read_rows(FINAL_PER_CLASS_CSV)
        if row["model_id"] == "final_stack_small_plus_lgbm_anchor" and row["split"] == "test"
    ]


def save_model_comparison() -> None:
    """Save a macro-F1 comparison bar chart."""
    rows = read_rows(FINAL_MODEL_RESULTS_CSV) + read_rows(FINAL_BASE_RESULTS_CSV)
    rows = sorted(rows, key=lambda row: float(row["test_macro_f1"]), reverse=True)
    width, height = 980, 590
    left, top, chart_w = 230, 78, 610
    row_h = 46
    body = [
        text(width / 2, 34, "Final Test Model Comparison", size=20, anchor="middle"),
        rect(left, top - 28, chart_w, 1, "#D1D5DB"),
        text(left + chart_w + 26, top - 33, "Score", size=11, fill="#374151"),
    ]
    for tick in (0.5, 0.6, 0.7, 0.8):
        x = left + (tick - 0.48) / (0.86 - 0.48) * chart_w
        body.append(rect(x, top - 23, 1, len(rows) * row_h + 24, "#E5E7EB"))
        body.append(text(x, top - 31, f"{tick:.1f}", size=10, fill="#6B7280", anchor="middle"))
    body.append(rect(left + 440, height - 48, 16, 10, "#3B82F6"))
    body.append(text(left + 462, height - 39, "Test macro-F1", size=12, fill="#374151"))
    body.append(rect(left + 560, height - 48, 16, 10, "#F97316"))
    body.append(text(left + 582, height - 39, "Minority recall mean", size=12, fill="#374151"))

    for index, row in enumerate(rows):
        y = top + index * row_h
        label = short_model_name(row["model_id"])
        macro_f1 = float(row["test_macro_f1"])
        minority = float(row["test_minority_recall_mean"])
        macro_w = (macro_f1 - 0.48) / (0.86 - 0.48) * chart_w
        minority_w = (minority - 0.48) / (0.86 - 0.48) * chart_w
        body.append(text(28, y + 16, label, size=12, fill="#111827"))
        body.append(rect(left, y, max(0, macro_w), 14, "#3B82F6"))
        body.append(rect(left, y + 18, max(0, minority_w), 14, "#F97316"))
        body.append(text(left + max(0, macro_w) + 7, y + 11, f"{macro_f1:.3f}", size=11))
        body.append(text(left + max(0, minority_w) + 7, y + 29, f"{minority:.3f}", size=11))
    save_svg(FIGURE_DIR / "final_model_comparison.svg", width, height, body)


def save_confusion_matrix() -> None:
    """Save final stacking confusion matrix heatmap."""
    labels = final_label_order()
    rows = [
        row
        for row in read_rows(FINAL_CONFUSION_CSV)
        if row["model_id"] == "final_stack_small_plus_lgbm_anchor" and row["split"] == "test"
    ]
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        matrix[label_to_index[row["true_label"]]][label_to_index[row["predicted_label"]]] = int(row["count"])
    max_value = max(max(row) for row in matrix)

    cell = 54
    left, top = 150, 98
    width, height = 780, 760
    body = [
        text(width / 2, 34, "Final Stacking Test Confusion Matrix", size=20, anchor="middle"),
        text(left + cell * len(labels) / 2, height - 34, "Predicted label", size=14, anchor="middle"),
        text(23, top + cell * len(labels) / 2, "True label", size=14, anchor="middle"),
    ]
    for index, label in enumerate(labels):
        body.append(text(left - 8, top + index * cell + 32, label, size=11, anchor="end"))
        body.append(text(left + index * cell + 27, top - 10, label, size=11, anchor="middle"))
    for i, row_values in enumerate(matrix):
        for j, value in enumerate(row_values):
            x = left + j * cell
            y = top + i * cell
            fill = blue_scale(value, max_value)
            body.append(rect(x, y, cell, cell, fill, "#FFFFFF"))
            if value:
                body.append(
                    text(
                        x + cell / 2,
                        y + cell / 2 + 4,
                        str(value),
                        size=10,
                        fill="white" if value > max_value * 0.55 else "#111827",
                        anchor="middle",
                    )
                )
    save_svg(FIGURE_DIR / "final_stack_confusion_matrix.svg", width, height, body)


def save_per_class_f1() -> None:
    """Save final stacking per-class F1 and recall bars."""
    rows = [
        row
        for row in read_rows(FINAL_PER_CLASS_CSV)
        if row["model_id"] == "final_stack_small_plus_lgbm_anchor" and row["split"] == "test"
    ]
    width, height = 930, 500
    left, top, chart_h = 72, 68, 330
    group_w = 78
    body = [
        text(width / 2, 34, "Final Stacking Per-Class Test Performance", size=20, anchor="middle"),
        rect(left, top + chart_h, len(rows) * group_w + 18, 1, "#9CA3AF"),
        rect(width - 218, 48, 15, 10, "#10B981"),
        text(width - 196, 57, "F1", size=12, fill="#374151"),
        rect(width - 158, 48, 15, 10, "#6366F1"),
        text(width - 136, 57, "Recall", size=12, fill="#374151"),
    ]
    for tick in (0.6, 0.7, 0.8, 0.9):
        y = top + chart_h - (tick - 0.52) / (0.98 - 0.52) * chart_h
        body.append(rect(left, y, len(rows) * group_w + 18, 1, "#E5E7EB"))
        body.append(text(left - 10, y + 4, f"{tick:.1f}", size=10, fill="#6B7280", anchor="end"))
    for index, row in enumerate(rows):
        x = left + index * group_w + 12
        f1 = float(row["f1"])
        recall = float(row["recall"])
        f1_h = (f1 - 0.52) / (0.98 - 0.52) * chart_h
        recall_h = (recall - 0.52) / (0.98 - 0.52) * chart_h
        body.append(rect(x, top + chart_h - f1_h, 22, f1_h, "#10B981"))
        body.append(rect(x + 26, top + chart_h - recall_h, 22, recall_h, "#6366F1"))
        body.append(text(x + 24, top + chart_h + 22, row["class_name"], size=10, anchor="middle"))
        body.append(text(x + 11, top + chart_h - f1_h - 5, f"{f1:.2f}", size=9, anchor="middle"))
    save_svg(FIGURE_DIR / "final_stack_per_class_performance.svg", width, height, body)


def main() -> None:
    """Build all final figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_model_comparison()
    save_confusion_matrix()
    save_per_class_f1()
    print(f"Saved final SVG figures to {FIGURE_DIR}", flush=True)


if __name__ == "__main__":
    main()
