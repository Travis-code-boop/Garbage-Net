"""Check standard image preprocessing on one train sample per class.

Notebook block purpose:
    Preprocessing sanity check. This code selects one training image from each
    class, creates the standard image versions used by later feature extraction,
    saves a compact CSV summary, and writes a visual preview grid including
    RGB, HSV-H, HSV-S, HSV-V, grayscale, and binary mask views.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from image_preprocessing import (
    PROJECT_ROOT,
    array_to_uint8_image,
    mask_to_preview,
    prepare_image_versions,
    read_metadata_rows,
    select_one_train_sample_per_class,
    summarize_versions,
)


METADATA_WITH_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_with_split.csv"
PREVIEW_DIR = PROJECT_ROOT / "data" / "processed" / "previews"
PREPROCESSING_CHECK_PATH = PREVIEW_DIR / "preprocessing_check.csv"
PREVIEW_GRID_PATH = PREVIEW_DIR / "preprocessing_preview_grid.jpg"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable column order."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def labeled_tile(image: Image.Image, label: str, size: tuple[int, int] = (128, 152)) -> Image.Image:
    """Create a small labeled preview tile."""
    image = image.resize((128, 128), Image.Resampling.BILINEAR).convert("RGB")
    tile = Image.new("RGB", size, "white")
    tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 132), label[:22], fill=(0, 0, 0))
    return tile


def make_preview_grid(sample_rows: list[dict[str, str]]) -> Image.Image:
    """Create a preview grid with RGB, HSV channels, grayscale, and mask columns."""
    columns = ["rgb", "hsv_h", "hsv_s", "hsv_v", "gray", "mask"]
    tile_width, tile_height = 128, 152
    grid = Image.new(
        "RGB",
        (tile_width * len(columns), tile_height * len(sample_rows)),
        "white",
    )

    for row_index, row in enumerate(sample_rows):
        versions = prepare_image_versions(row["path"])
        previews = [
            labeled_tile(array_to_uint8_image(versions.rgb), f"{row['label']} rgb"),
            labeled_tile(array_to_uint8_image(versions.hsv[:, :, 0]), "hsv_h"),
            labeled_tile(array_to_uint8_image(versions.hsv[:, :, 1]), "hsv_s"),
            labeled_tile(array_to_uint8_image(versions.hsv[:, :, 2]), "hsv_v"),
            labeled_tile(array_to_uint8_image(versions.gray), "gray"),
            labeled_tile(mask_to_preview(versions.binary_mask), "mask"),
        ]
        for column_index, preview in enumerate(previews):
            grid.paste(preview, (column_index * tile_width, row_index * tile_height))

    return grid


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_metadata_rows(METADATA_WITH_SPLIT_PATH)
    sample_rows = select_one_train_sample_per_class(rows)

    summary_rows = []
    for row in sample_rows:
        versions = prepare_image_versions(row["path"])
        summary_rows.append(summarize_versions(row, versions))

    summary_columns = [
        "image_id",
        "label",
        "path",
        "rgb_shape",
        "hsv_shape",
        "gray_shape",
        "gray_128_shape",
        "mask_shape",
        "mask_foreground_ratio",
        "rgb_dtype",
        "gray_dtype",
    ]
    write_csv(PREPROCESSING_CHECK_PATH, summary_rows, summary_columns)

    preview_grid = make_preview_grid(sample_rows)
    preview_grid.save(PREVIEW_GRID_PATH, quality=95)

    for row in summary_rows:
        print(row)
    print(f"saved_csv={PREPROCESSING_CHECK_PATH.relative_to(PROJECT_ROOT)}")
    print(f"saved_preview={PREVIEW_GRID_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
