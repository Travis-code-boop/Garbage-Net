"""Extract F6 ORB Bag-of-Features vectors.

Notebook block purpose:
    F6 local keypoint feature extraction, ORB part. This code uses OpenCV ORB
    to extract local binary descriptors, learns a visual vocabulary from train
    images only, and converts each image into a normalized visual-word
    histogram. The output can be concatenated with SIFT-BoF to form F6_bof_256.

Feature dimensions:
    ORB vocabulary size: 128 visual words
    ORB-BoF vector: 128-dimensional L1-normalized histogram
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATH = PROJECT_ROOT / "vendor" / "python"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from feature_io import (  # noqa: E402
    FEATURE_DIR,
    label_mapping,
    read_metadata_with_split,
    rows_by_split,
    save_feature_cache,
    summarize_feature_cache,
)
from image_preprocessing import prepare_image_versions  # noqa: E402


OUTPUT_DIR = FEATURE_DIR / "F6_bof"
VOCAB_PATH = OUTPUT_DIR / "vocab_orb_k128.npy"
FEATURE_CACHE_PATH = OUTPUT_DIR / "F6_orb_bof_128.npz"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "F6_orb_bof_128_summary.json"

RANDOM_SEED = 42
ORB_VOCAB_SIZE = 128
ORB_NFEATURES = 500
ORB_SCALE_FACTOR = 1.2
ORB_NLEVELS = 8
MAX_DESCRIPTORS_PER_IMAGE_FOR_VOCAB = 100
MAX_TOTAL_DESCRIPTORS_FOR_VOCAB = 100_000
KMEANS_ATTEMPTS = 3
KMEANS_MAX_ITER = 100
KMEANS_EPS = 1e-4


def create_orb() -> cv2.ORB:
    """Create the OpenCV ORB extractor."""
    return cv2.ORB_create(
        nfeatures=ORB_NFEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_NLEVELS,
    )


def extract_orb_descriptors(relative_path: str, orb: cv2.ORB) -> np.ndarray | None:
    """Extract ORB descriptors from one image; return None if no keypoints exist."""
    versions = prepare_image_versions(relative_path)
    _keypoints, descriptors = orb.detectAndCompute(versions.gray, None)
    if descriptors is None or len(descriptors) == 0:
        return None
    return descriptors.astype(np.float32, copy=False)


def sample_descriptors(
    descriptors: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample up to max_count descriptors from one image."""
    if len(descriptors) <= max_count:
        return descriptors
    indices = rng.choice(len(descriptors), size=max_count, replace=False)
    return descriptors[indices]


def collect_train_descriptors(
    train_rows: list[dict[str, str]],
    orb: cv2.ORB,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Collect sampled ORB descriptors from train images only for vocabulary fitting."""
    descriptor_chunks = []
    total_descriptors_before_sampling = 0
    no_descriptor_images = 0

    for index, row in enumerate(train_rows):
        descriptors = extract_orb_descriptors(row["path"], orb)
        if descriptors is None:
            no_descriptor_images += 1
        else:
            total_descriptors_before_sampling += len(descriptors)
            descriptor_chunks.append(
                sample_descriptors(descriptors, MAX_DESCRIPTORS_PER_IMAGE_FOR_VOCAB, rng)
            )

        if (index + 1) % 1000 == 0:
            print(f"collected ORB descriptors from {index + 1}/{len(train_rows)} train images")

    if not descriptor_chunks:
        raise RuntimeError("No ORB descriptors found in the training set.")

    descriptors_for_vocab = np.vstack(descriptor_chunks).astype(np.float32, copy=False)
    if len(descriptors_for_vocab) > MAX_TOTAL_DESCRIPTORS_FOR_VOCAB:
        indices = rng.choice(
            len(descriptors_for_vocab),
            size=MAX_TOTAL_DESCRIPTORS_FOR_VOCAB,
            replace=False,
        )
        descriptors_for_vocab = descriptors_for_vocab[indices]

    stats = {
        "train_images": len(train_rows),
        "train_images_without_orb_descriptors": no_descriptor_images,
        "train_descriptors_before_sampling": int(total_descriptors_before_sampling),
        "train_descriptors_for_vocab": int(len(descriptors_for_vocab)),
    }
    return descriptors_for_vocab, stats


def fit_visual_vocabulary(descriptors: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit k-means visual vocabulary with OpenCV kmeans."""
    if len(descriptors) < ORB_VOCAB_SIZE:
        raise RuntimeError(f"Need at least {ORB_VOCAB_SIZE} descriptors, got {len(descriptors)}.")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITER,
        KMEANS_EPS,
    )
    compactness, _labels, centers = cv2.kmeans(
        descriptors.astype(np.float32, copy=False),
        ORB_VOCAB_SIZE,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32), float(compactness)


def descriptors_to_histogram(descriptors: np.ndarray | None, vocabulary: np.ndarray) -> np.ndarray:
    """Convert ORB descriptors into an L1-normalized visual-word histogram."""
    histogram = np.zeros(ORB_VOCAB_SIZE, dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return histogram

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(descriptors.astype(np.float32, copy=False), vocabulary)
    word_indices = [match.trainIdx for match in matches]
    histogram += np.bincount(word_indices, minlength=ORB_VOCAB_SIZE).astype(np.float32)
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram


def extract_split_features(
    split_rows: list[dict[str, str]],
    orb: cv2.ORB,
    vocabulary: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract ORB-BoF histograms for all rows in one split."""
    matrix = np.empty((len(split_rows), ORB_VOCAB_SIZE), dtype=np.float32)
    no_descriptor_images = 0
    total_descriptors = 0

    for index, row in enumerate(split_rows):
        descriptors = extract_orb_descriptors(row["path"], orb)
        if descriptors is None:
            no_descriptor_images += 1
        else:
            total_descriptors += len(descriptors)
        matrix[index] = descriptors_to_histogram(descriptors, vocabulary)

        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(split_rows)} images for split")

    stats = {
        "images": len(split_rows),
        "images_without_orb_descriptors": no_descriptor_images,
        "descriptors_total": int(total_descriptors),
    }
    return matrix, stats


def build_feature_names() -> tuple[list[str], list[str]]:
    """Build stable ORB-BoF feature names and groups."""
    feature_names = [f"bof_orb_word_{index:03d}" for index in range(ORB_VOCAB_SIZE)]
    feature_groups = ["bof_orb"] * ORB_VOCAB_SIZE
    return feature_names, feature_groups


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_metadata_with_split()
    split_rows = rows_by_split(rows)
    mapping = label_mapping(rows)
    feature_names, feature_groups = build_feature_names()
    rng = np.random.default_rng(RANDOM_SEED)
    orb = create_orb()

    print("collecting train descriptors for ORB vocabulary")
    descriptors_for_vocab, vocab_collection_stats = collect_train_descriptors(
        split_rows["train"],
        orb,
        rng,
    )

    print(f"fitting ORB visual vocabulary: {len(descriptors_for_vocab)} descriptors")
    vocabulary, compactness = fit_visual_vocabulary(descriptors_for_vocab)
    np.save(VOCAB_PATH, vocabulary)

    split_features = {}
    split_descriptor_stats = {}
    for split_name, current_rows in split_rows.items():
        print(f"extracting {split_name}: {len(current_rows)} images")
        split_features[split_name], split_descriptor_stats[split_name] = extract_split_features(
            current_rows,
            orb,
            vocabulary,
        )

    params = {
        "feature_id": "F6_orb_bof",
        "feature_dim": ORB_VOCAB_SIZE,
        "opencv_version": cv2.__version__,
        "orb_nfeatures": ORB_NFEATURES,
        "orb_scale_factor": ORB_SCALE_FACTOR,
        "orb_nlevels": ORB_NLEVELS,
        "descriptor_dim": 32,
        "descriptor_note": "ORB binary descriptors converted to float32 for k-means",
        "vocabulary_size": ORB_VOCAB_SIZE,
        "vocabulary_fit_split": "train only",
        "max_descriptors_per_image_for_vocab": MAX_DESCRIPTORS_PER_IMAGE_FOR_VOCAB,
        "max_total_descriptors_for_vocab": MAX_TOTAL_DESCRIPTORS_FOR_VOCAB,
        "kmeans": {
            "implementation": "cv2.kmeans",
            "attempts": KMEANS_ATTEMPTS,
            "max_iter": KMEANS_MAX_ITER,
            "eps": KMEANS_EPS,
            "init": "KMEANS_PP_CENTERS",
            "compactness": compactness,
        },
        "histogram_normalization": "L1",
        "empty_descriptor_policy": "all-zero histogram",
        "vocabulary_path": str(VOCAB_PATH.relative_to(PROJECT_ROOT)),
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
    summary["vocab_collection_stats"] = vocab_collection_stats
    summary["split_descriptor_stats"] = split_descriptor_stats
    summary["params"] = params

    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
