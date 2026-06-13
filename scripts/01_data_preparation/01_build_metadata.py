"""Build image metadata for the garbage classification dataset.

Notebook block purpose:
    Data loading and descriptive statistics. This code scans the selected
    standardized image directory, checks whether each image can be read,
    records image size/mode/path/label, and saves metadata tables for later
    splitting, feature extraction, and reporting.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "garbage-classification-v2" / "standardized_256"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "metadata"

METADATA_PATH = OUTPUT_DIR / "metadata.csv"
CLASS_DISTRIBUTION_PATH = OUTPUT_DIR / "class_distribution.csv"
QUALITY_SUMMARY_PATH = OUTPUT_DIR / "data_quality_summary.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_channel_count(mode: str) -> int | str:
    """Return a simple channel count from the PIL image mode."""
    mapping = {
        "1": 1,
        "L": 1,
        "P": 1,
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
    }
    return mapping.get(mode, "unknown")


def scan_images() -> list[dict[str, object]]:
    """Scan the selected data directory and collect one metadata row per image."""
    rows: list[dict[str, object]] = []
    classes = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())

    image_index = 0
    for class_dir in classes:
        label = class_dir.name
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            image_index += 1
            row: dict[str, object] = {
                "image_id": f"img_{image_index:06d}",
                "label": label,
                "path": str(image_path.relative_to(PROJECT_ROOT)),
                "file_name": image_path.name,
                "extension": image_path.suffix.lower(),
                "file_size_bytes": image_path.stat().st_size,
                "is_readable": False,
                "width": "",
                "height": "",
                "mode": "",
                "channels": "",
                "format": "",
                "error": "",
            }

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    mode = image.mode
                    row.update(
                        {
                            "is_readable": True,
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "channels": image_channel_count(mode),
                            "format": image.format or "",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - metadata audit should record all failures.
                row["error"] = repr(exc)

            rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable column order."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = scan_images()

    metadata_columns = [
        "image_id",
        "label",
        "path",
        "file_name",
        "extension",
        "file_size_bytes",
        "is_readable",
        "width",
        "height",
        "mode",
        "channels",
        "format",
        "error",
    ]
    write_csv(METADATA_PATH, rows, metadata_columns)

    readable_rows = [row for row in rows if row["is_readable"]]
    class_counts = Counter(str(row["label"]) for row in readable_rows)
    class_distribution_rows = [
        {"label": label, "count": class_counts[label]} for label in sorted(class_counts)
    ]
    write_csv(CLASS_DISTRIBUTION_PATH, class_distribution_rows, ["label", "count"])

    size_counts = Counter(f"{row['width']}x{row['height']}" for row in readable_rows)
    mode_counts = Counter(str(row["mode"]) for row in readable_rows)
    extension_counts = Counter(str(row["extension"]) for row in rows)

    summary = {
        "data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
        "total_files": len(rows),
        "readable_files": len(readable_rows),
        "unreadable_files": len(rows) - len(readable_rows),
        "num_classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "size_counts": dict(sorted(size_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
    }
    QUALITY_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
