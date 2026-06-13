"""Extract F4 shape features with the planned 12-dimensional design.

Notebook block purpose:
    F4 shape feature extraction. For each image, this code uses the binary
    foreground mask as a fast object-body approximation, extracts five geometric
    contour features, computes seven Hu moment features, and saves the resulting
    split matrices for later single-feature model screening.

Feature dimensions:
    Geometry: area_ratio, perimeter_norm, aspect_ratio, extent, solidity = 5
    Hu moments: signed-log Hu moment representation = 7
    Total: 12
"""

from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from feature_io import (
    FEATURE_DIR,
    PROJECT_ROOT,
    label_mapping,
    read_metadata_with_split,
    rows_by_split,
    save_feature_cache,
    summarize_feature_cache,
)
from image_preprocessing import prepare_image_versions


OUTPUT_DIR = FEATURE_DIR / "F4_shape"
FEATURE_CACHE_PATH = OUTPUT_DIR / "F4_shape_12.npz"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "F4_shape_12_summary.json"

MIN_COMPONENT_PIXELS = 10
HU_EPS = 1e-30


def largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the largest 8-connected component from a binary mask."""
    foreground = mask.astype(bool)
    visited = np.zeros(foreground.shape, dtype=bool)
    best_coords: list[tuple[int, int]] = []
    height, width = foreground.shape
    ys, xs = np.nonzero(foreground)

    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        current_coords: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True

        while queue:
            y, x = queue.popleft()
            current_coords.append((y, x))

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if len(current_coords) > len(best_coords):
            best_coords = current_coords

    component = np.zeros(foreground.shape, dtype=np.uint8)
    if best_coords:
        coord_array = np.asarray(best_coords, dtype=np.int32)
        component[coord_array[:, 0], coord_array[:, 1]] = 1

    return component, len(best_coords)


def boundary_mask(component: np.ndarray) -> np.ndarray:
    """Find boundary pixels using 4-neighborhood erosion logic."""
    padded = np.pad(component.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    interior = center & up & down & left & right
    return (center & ~interior).astype(np.uint8)


def perimeter_4_connected(component: np.ndarray) -> float:
    """Compute pixel-edge perimeter from a binary component."""
    padded = np.pad(component.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = 0
    perimeter += np.sum(center & ~padded[:-2, 1:-1])
    perimeter += np.sum(center & ~padded[2:, 1:-1])
    perimeter += np.sum(center & ~padded[1:-1, :-2])
    perimeter += np.sum(center & ~padded[1:-1, 2:])
    return float(perimeter)


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Compute a 2D monotonic-chain convex hull from x/y points."""
    if len(points) <= 1:
        return [(float(points[0, 0]), float(points[0, 1]))] if len(points) == 1 else []

    unique_points = sorted({(float(x), float(y)) for x, y in points})

    def cross(
        origin: tuple[float, float],
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Compute polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    x = array[:, 0]
    y = array[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def central_moment(component: np.ndarray, x_center: float, y_center: float, p: int, q: int) -> float:
    """Compute a central binary moment mu_pq."""
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return 0.0
    return float(np.sum(((xs - x_center) ** p) * ((ys - y_center) ** q)))


def hu_moments_signed_log(component: np.ndarray) -> np.ndarray:
    """Compute seven Hu moments and return signed-log transformed values."""
    ys, xs = np.nonzero(component)
    if len(xs) < MIN_COMPONENT_PIXELS:
        return np.zeros(7, dtype=np.float32)

    m00 = float(len(xs))
    x_center = float(xs.mean())
    y_center = float(ys.mean())

    eta: dict[tuple[int, int], float] = {}
    for p, q in ((2, 0), (0, 2), (1, 1), (3, 0), (1, 2), (2, 1), (0, 3)):
        mu = central_moment(component, x_center, y_center, p, q)
        eta[(p, q)] = mu / (m00 ** (1.0 + (p + q) / 2.0))

    n20 = eta[(2, 0)]
    n02 = eta[(0, 2)]
    n11 = eta[(1, 1)]
    n30 = eta[(3, 0)]
    n12 = eta[(1, 2)]
    n21 = eta[(2, 1)]
    n03 = eta[(0, 3)]

    hu = np.asarray(
        [
            n20 + n02,
            (n20 - n02) ** 2 + 4.0 * n11**2,
            (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2,
            (n30 + n12) ** 2 + (n21 + n03) ** 2,
            (n30 - 3.0 * n12)
            * (n30 + n12)
            * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2)
            + (3.0 * n21 - n03)
            * (n21 + n03)
            * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
            (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4.0 * n11 * (n30 + n12) * (n21 + n03),
            (3.0 * n21 - n03)
            * (n30 + n12)
            * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2)
            - (n30 - 3.0 * n12)
            * (n21 + n03)
            * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        ],
        dtype=np.float64,
    )

    return (-np.sign(hu) * np.log10(np.abs(hu) + HU_EPS)).astype(np.float32)


def geometry_features(component: np.ndarray) -> np.ndarray:
    """Extract five normalized geometric shape features."""
    height, width = component.shape
    ys, xs = np.nonzero(component)
    area = float(len(xs))
    if area < MIN_COMPONENT_PIXELS:
        return np.zeros(5, dtype=np.float32)

    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    bbox_width = float(x_max - x_min + 1)
    bbox_height = float(y_max - y_min + 1)
    bbox_area = bbox_width * bbox_height

    area_ratio = area / float(height * width)
    perimeter_norm = perimeter_4_connected(component) / float(2 * (height + width))
    aspect_ratio = bbox_width / max(bbox_height, 1.0)
    extent = area / max(bbox_area, 1.0)

    boundary = boundary_mask(component)
    boundary_y, boundary_x = np.nonzero(boundary)
    if len(boundary_x) >= 3:
        hull_points = convex_hull(np.column_stack([boundary_x, boundary_y]))
        hull_area = polygon_area(hull_points)
    else:
        hull_area = 0.0
    hull_area = max(hull_area, area)
    solidity = area / hull_area

    return np.asarray(
        [area_ratio, perimeter_norm, aspect_ratio, extent, solidity],
        dtype=np.float32,
    )


def extract_shape_features_for_image(relative_path: str) -> tuple[np.ndarray, bool]:
    """Extract the planned 12-dimensional shape vector for one image."""
    versions = prepare_image_versions(relative_path)
    component = versions.binary_mask.astype(np.uint8)
    if int(component.sum()) < MIN_COMPONENT_PIXELS:
        return np.zeros(12, dtype=np.float32), False

    features = np.concatenate([geometry_features(component), hu_moments_signed_log(component)])
    return features.astype(np.float32), True


def build_feature_names() -> tuple[list[str], list[str]]:
    """Build stable feature names and group labels for shape features."""
    feature_names = [
        "shape_area_ratio",
        "shape_perimeter_norm",
        "shape_aspect_ratio",
        "shape_extent",
        "shape_solidity",
    ] + [f"shape_hu_signed_log_{index:02d}" for index in range(1, 8)]
    feature_groups = ["shape_geometry"] * 5 + ["shape_hu"] * 7
    return feature_names, feature_groups


def extract_split_features(split_rows: list[dict[str, str]]) -> tuple[np.ndarray, int]:
    """Extract shape features for all rows in one split."""
    matrix = np.empty((len(split_rows), 12), dtype=np.float32)
    failures = 0
    for index, row in enumerate(split_rows):
        matrix[index], ok = extract_shape_features_for_image(row["path"])
        if not ok:
            failures += 1
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(split_rows)} images for split")
    return matrix, failures


def main() -> None:
    start_time = time.perf_counter()
    rows = read_metadata_with_split()
    split_rows = rows_by_split(rows)
    mapping = label_mapping(rows)
    feature_names, feature_groups = build_feature_names()

    split_features = {}
    split_failures = {}
    for split_name, current_rows in split_rows.items():
        print(f"extracting {split_name}: {len(current_rows)} images")
        split_features[split_name], split_failures[split_name] = extract_split_features(current_rows)

    params = {
        "feature_id": "F4_shape",
        "feature_dim": 12,
        "foreground_mask": "Otsu threshold with light/dark foreground heuristic",
        "component": "full binary foreground mask used as fast body approximation",
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "geometry_features": [
            "area_ratio",
            "perimeter_norm",
            "aspect_ratio",
            "extent",
            "solidity",
        ],
        "perimeter": "4-connected pixel-edge perimeter normalized by 2*(height+width)",
        "solidity": "area / max(convex_hull_area_of_boundary_points, area)",
        "hu_moments": "seven binary Hu moments transformed as -sign(hu)*log10(abs(hu)+eps)",
        "hu_eps": HU_EPS,
        "source_metadata": str(
            (PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_with_split.csv").relative_to(
                PROJECT_ROOT
            )
        ),
    }

    save_feature_cache(
        FEATURE_CACHE_PATH,
        split_rows,
        split_features,
        feature_names,
        feature_groups,
        params,
        mapping,
    )

    summary = summarize_feature_cache(split_features, feature_names, feature_groups)
    summary["output_path"] = str(FEATURE_CACHE_PATH.relative_to(PROJECT_ROOT))
    summary["elapsed_seconds"] = round(time.perf_counter() - start_time, 3)
    summary["component_failures"] = split_failures
    summary["params"] = params

    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
