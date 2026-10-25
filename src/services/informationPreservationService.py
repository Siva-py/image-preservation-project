"""
Information Preservation Service
---------------------------------
Selects a fixed-budget subset of images using a joint feature-based heuristic:
1. Feature-space coverage
2. Facility-location-style representativeness
3. Feature-mean preservation

The service returns selected paths and an optional JSON report. When output_dir
is supplied, readable images are copied into selected_images/ and removed_images/
for visual inspection. Source files are never modified, moved, or deleted.

Dependencies:
    pip install torch torchvision pillow numpy

The first run may download pretrained ResNet-18 weights.

Scalability notes:
- DataLoader workers parallelize CPU image decoding and preprocessing.
- Selection uses a bounded set of reference images for approximate
  representativeness and batched greedy rounds, avoiding the full N x N matrix
  and reducing per-image score recomputation.
- Selection is still greedy and can be compute-intensive for large budgets.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def _validate_weights(
    coverage_weight: float,
    representativeness_weight: float,
    mean_preservation_weight: float,
) -> tuple[float, float, float]:
    weights = np.asarray(
        [coverage_weight, representativeness_weight, mean_preservation_weight],
        dtype=np.float64,
    )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Objective weights must be non-negative and not all zero.")
    weights /= weights.sum()
    return tuple(float(w) for w in weights)


class _ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int = 224):
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, str(path), ""
        except Exception as exc:
            # Match the configured image size so a bad file cannot break collation.
            return torch.zeros(3, self.image_size, self.image_size), str(path), str(exc)


def extract_features(
    image_paths: Sequence[Path],
    batch_size: int = 32,
    image_size: int = 224,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> tuple[list[Path], np.ndarray, list[dict[str, str]]]:
    """Extract L2-normalized ResNet-18 embeddings with parallel data loading."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if image_size < 1:
        raise ValueError("image_size must be >= 1.")

    if num_workers is None:
        # Conservative default; users can override after benchmarking.
        num_workers = min(4, os.cpu_count() or 1)
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(backbone.children())[:-1])
    model.eval().to(resolved_device)

    dataset = _ImageDataset(image_paths, image_size=image_size)
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": resolved_device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(**loader_options)

    valid_paths: list[Path] = []
    embedding_chunks: list[np.ndarray] = []
    errors: list[dict[str, str]] = []

    with torch.inference_mode():
        for images, path_strings, error_strings in loader:
            keep_indices = []
            for i, error in enumerate(error_strings):
                if error:
                    errors.append({"path": path_strings[i], "error": error})
                else:
                    keep_indices.append(i)

            if not keep_indices:
                continue

            batch = images[keep_indices].to(
                resolved_device, non_blocking=(resolved_device.type == "cuda")
            )
            vectors = model(batch).flatten(1)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            embedding_chunks.append(vectors.cpu().numpy().astype(np.float32))
            valid_paths.extend(Path(path_strings[i]) for i in keep_indices)

    if not embedding_chunks:
        return [], np.empty((0, 512), dtype=np.float32), errors
    return valid_paths, np.concatenate(embedding_chunks, axis=0), errors


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _choose_reference_indices(n: int, max_references: int, seed: int) -> np.ndarray:
    """Choose deterministic reference images for scalable approximate facility gain."""
    if n <= max_references:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_references, replace=False))


def select_images_jointly(
    embeddings: np.ndarray,
    number_to_select: int,
    coverage_weight: float = 1.0,
    representativeness_weight: float = 1.0,
    mean_preservation_weight: float = 1.0,
    random_seed: int = 42,
    max_references: int = 512,
    selection_batch_size: int = 32,
) -> list[int]:
    """
    Approximate joint selection with batched greedy rounds.

    Coverage tracks each candidate's similarity to the selected subset.
    Representativeness is approximated over at most max_references images.
    Mean preservation steers the selected feature sum toward k * dataset mean.

    In each round, a batch of high-scoring candidates is accepted together.
    This is faster than recalculating scores after every single selection,
    but is an approximation because candidates within the same batch do not
    receive individual score updates. Benchmark quality against baselines.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    n, _ = x.shape
    if n == 0 or number_to_select <= 0:
        return []
    k = min(int(number_to_select), n)
    if k == n:
        return list(range(n))
    if max_references < 1:
        raise ValueError("max_references must be >= 1.")
    if selection_batch_size < 1:
        raise ValueError("selection_batch_size must be >= 1.")

    w_cov, w_rep, w_mean = _validate_weights(
        coverage_weight, representativeness_weight, mean_preservation_weight
    )

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)

    ref_indices = _choose_reference_indices(n, max_references, random_seed)
    refs = x[ref_indices]
    # R x N similarity matrix, where R is bounded (default 512).
    ref_similarity = np.clip(refs @ x.T, -1.0, 1.0)
    ref_best_similarity = np.full(len(ref_indices), -1.0, dtype=np.float32)

    dataset_mean = x.mean(axis=0, dtype=np.float64)
    mean_norm = np.linalg.norm(dataset_mean)
    if mean_norm > 1e-12:
        first = int(np.argmax(x @ (dataset_mean / mean_norm).astype(np.float32)))
    else:
        first = int(np.random.default_rng(random_seed).integers(n))

    selected = [first]
    chosen = np.zeros(n, dtype=bool)
    chosen[first] = True
    nearest_similarity = np.clip(x @ x[first], -1.0, 1.0)
    selected_feature_sum = x[first].astype(np.float64)
    target_sum = dataset_mean * k

    while len(selected) < k:
        coverage_gain = 1.0 - nearest_similarity
        facility_gain = np.maximum(
            0.0, ref_similarity - ref_best_similarity[:, None]
        ).sum(axis=0, dtype=np.float64)

        residual = target_sum - selected_feature_sum
        residual_norm = np.linalg.norm(residual)
        if residual_norm > 1e-12:
            mean_gain = x.astype(np.float64) @ (residual / residual_norm)
        else:
            mean_gain = np.zeros(n, dtype=np.float64)

        score = (
            w_cov * _minmax_normalize(coverage_gain)
            + w_rep * _minmax_normalize(facility_gain)
            + w_mean * _minmax_normalize(mean_gain)
        )
        score[chosen] = -np.inf

        take = min(selection_batch_size, k - len(selected))
        # Top-k candidates for this round. Scores are refreshed after the batch.
        candidates = np.argpartition(score, -take)[-take:]
        candidates = candidates[np.argsort(score[candidates])[::-1]]
        candidates = [int(i) for i in candidates if not chosen[i]]
        if not candidates:
            break

        selected.extend(candidates)
        chosen[candidates] = True
        selected_feature_sum += x[candidates].sum(axis=0, dtype=np.float64)

        # Batch matrix products update state more efficiently than one candidate
        # at a time. Similarities are clipped to handle floating-point noise.
        selected_vectors = x[candidates]
        batch_similarity = np.clip(x @ selected_vectors.T, -1.0, 1.0)
        nearest_similarity = np.maximum(
            nearest_similarity, batch_similarity.max(axis=1)
        )
        ref_batch_similarity = np.clip(refs @ selected_vectors.T, -1.0, 1.0)
        ref_best_similarity = np.maximum(
            ref_best_similarity, ref_batch_similarity.max(axis=1)
        )

    return selected[:k]



def _copy_images_to_output(
    selected_paths: Sequence[str],
    all_readable_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Copy selected and non-selected readable images into separate folders."""
    selected_dir = output_dir / "selected_images"
    removed_dir = output_dir / "removed_images"
    selected_dir.mkdir(parents=True, exist_ok=True)
    removed_dir.mkdir(parents=True, exist_ok=True)

    selected_set = {str(Path(path).resolve()) for path in selected_paths}
    source_paths = [Path(path).resolve() for path in all_readable_paths]
    if source_paths:
        try:
            common_root = Path(os.path.commonpath([str(p.parent) for p in source_paths]))
        except ValueError:
            common_root = source_paths[0].parent
    else:
        common_root = Path.cwd()

    copied_selected = 0
    copied_removed = 0
    copy_errors: list[dict[str, str]] = []

    def copy_one(source_path: Path, target_root: Path) -> bool:
        try:
            relative = source_path.relative_to(common_root)
        except ValueError:
            relative = Path(source_path.name)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            stem, suffix = target.stem, target.suffix
            counter = 1
            while True:
                candidate = target.with_name(f"{stem}_{counter}{suffix}")
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
        try:
            shutil.copy2(source_path, target)
            return True
        except Exception as exc:
            copy_errors.append({"path": str(source_path), "error": str(exc)})
            return False

    for source_path in source_paths:
        if str(source_path) in selected_set:
            copied_selected += int(copy_one(source_path, selected_dir))
        else:
            copied_removed += int(copy_one(source_path, removed_dir))

    return {
        "selected_output_dir": str(selected_dir.resolve()),
        "removed_output_dir": str(removed_dir.resolve()),
        "selected_files_copied": copied_selected,
        "removed_files_copied": copied_removed,
        "copy_error_count": len(copy_errors),
        "copy_errors": copy_errors,
    }


def run_service(
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    retain_ratio: float = 0.80,
    coverage_weight: float = 1.0,
    representativeness_weight: float = 1.0,
    mean_preservation_weight: float = 1.0,
    batch_size: int = 32,
    image_size: int = 224,
    device: Optional[str] = None,
    random_seed: int = 42,
    num_workers: Optional[int] = None,
    max_references: int = 512,
    selection_batch_size: int = 32,
    copy_images: bool = True,
) -> dict[str, Any]:
    """Run the service and return selected paths plus an optional JSON report."""
    if not 0 < retain_ratio <= 1:
        raise ValueError("retain_ratio must be in the interval (0, 1].")
    _validate_weights(
        coverage_weight, representativeness_weight, mean_preservation_weight
    )

    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source}")

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None else None
    )

    def is_inside(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    image_paths = sorted(
        p for p in source.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and (destination is None or not is_inside(p, destination))
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in: {source}")

    started = time.perf_counter()
    feature_started = time.perf_counter()
    valid_paths, embeddings, errors = extract_features(
        image_paths=image_paths,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
        num_workers=num_workers,
    )
    feature_seconds = time.perf_counter() - feature_started
    if len(valid_paths) == 0:
        raise ValueError("No readable images found; inspect decoding errors.")

    number_to_select = min(
        len(valid_paths), max(1, math.ceil(len(valid_paths) * retain_ratio))
    )
    selection_started = time.perf_counter()
    selected_indices = select_images_jointly(
        embeddings=embeddings,
        number_to_select=number_to_select,
        coverage_weight=coverage_weight,
        representativeness_weight=representativeness_weight,
        mean_preservation_weight=mean_preservation_weight,
        random_seed=random_seed,
        max_references=max_references,
        selection_batch_size=selection_batch_size,
    )
    selection_seconds = time.perf_counter() - selection_started
    selected_paths = [str(valid_paths[i]) for i in selected_indices]
    total_seconds = time.perf_counter() - started

    effective_workers = min(4, os.cpu_count() or 1) if num_workers is None else num_workers
    report: dict[str, Any] = {
        "input_dir": str(source),
        "method": "joint_greedy_coverage_approx_facility_mean",
        "total_images_found": len(image_paths),
        "readable_images": len(valid_paths),
        "unreadable_image_count": len(errors),
        "unreadable_images": errors,
        "selected_count": len(selected_paths),
        "retain_ratio_requested": retain_ratio,
        "actual_retain_ratio_of_readable": len(selected_paths) / len(valid_paths),
        "reduction_ratio_of_readable": 1.0 - len(selected_paths) / len(valid_paths),
        "objective_weights": {
            "coverage": coverage_weight,
            "representativeness": representativeness_weight,
            "mean_preservation": mean_preservation_weight,
        },
        "feature_extractor": "ImageNet-pretrained ResNet-18",
        "batch_size": batch_size,
        "num_workers": effective_workers,
        "max_references_for_approximate_representativeness": max_references,
        "selection_batch_size": selection_batch_size,
        "feature_extraction_seconds": round(feature_seconds, 3),
        "selection_seconds": round(selection_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "selected_images": selected_paths,
        "note": (
            "Images are selected by path; source files are not modified. "
            "Representativeness uses a sampled reference set when image count "
            "exceeds max_references."
        ),
    }

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        if copy_images:
            report["copied_images"] = _copy_images_to_output(
                selected_paths=selected_paths,
                all_readable_paths=valid_paths,
                output_dir=destination,
            )

        report_path = destination / "information_preservation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report["report_path"] = str(report_path)

    return report


if __name__ == "__main__":
    # Edit these paths for your machine.
    INPUT_DIR = r"C:\DM_Drive\Projects\Dissertation1 - Image Preservation\image-preservation-project\data\InputForInfo"
    OUTPUT_DIR = r"C:\DM_Drive\Projects\Dissertation1 - Image Preservation\image-preservation-project\data\outputForInfo"

    result = run_service(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        retain_ratio=0.50,
        batch_size=32,
        num_workers=5,       # CPU workers for image loading/preprocessing
        max_references=512,  # bounded reference set for scalable selection
        selection_batch_size=32, # batched selection trades some greediness for speed
        copy_images=True,         # copy selected and removed images into output folders
    )
    print(f"Found: {result['total_images_found']}")
    print(f"Readable: {result['readable_images']}")
    print(f"Selected: {result['selected_count']}")
    print(f"Feature extraction: {result['feature_extraction_seconds']} seconds")
    print(f"Selection: {result['selection_seconds']} seconds")
    print(f"Total: {result['total_seconds']} seconds")
    print(f"Report: {result.get('report_path', 'not saved')}")
    if "copied_images" in result:
        print(f"Selected folder: {result['copied_images']['selected_output_dir']}")
        print(f"Removed folder: {result['copied_images']['removed_output_dir']}")
        print(f"Copied selected: {result['copied_images']['selected_files_copied']}")
        print(f"Copied removed: {result['copied_images']['removed_files_copied']}")
