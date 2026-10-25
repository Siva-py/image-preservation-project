
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
from PIL import Image, ImageOps

from src.services.inputService import InputService


@dataclass(frozen=True)
class ImageQualityFilteringConfig:
    output_dir: Path
    blur_threshold: float = 100.0
    phash_hash_size: int = 16
    phash_threshold: int = 24
    hash_chunk_size: int = 1024 * 1024

    def validate(self) -> None:
        if self.blur_threshold < 0:
            raise ValueError("blur_threshold cannot be negative.")
        if self.phash_hash_size < 1:
            raise ValueError("phash_hash_size must be at least 1.")
        if self.phash_threshold < 0:
            raise ValueError("phash_threshold cannot be negative.")
        if self.hash_chunk_size < 1:
            raise ValueError("hash_chunk_size must be positive.")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


def perceptual_hash(path: Path, hash_size: int) -> str:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return str(imagehash.phash(image, hash_size=hash_size))


def laplacian_variance(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def relative_path(path: Path, source_root: Path | None) -> str:
    if source_root is not None:
        try:
            return str(path.resolve().relative_to(source_root.resolve()))
        except ValueError:
            pass

    return path.name


def quality_rank(record: dict[str, Any]) -> tuple:
    """
    Higher tuple values mean higher selection priority.

    Priority:
    1. Total pixel count
    2. Image sharpness
    3. File size
    4. Stable path tie-breaker
    """
    width = int(record.get("width") or 0)
    height = int(record.get("height") or 0)
    sharpness = float(record.get("blur_score") or 0)
    file_size = int(record.get("size_bytes") or 0)
    path = str(record.get("original_relative_path") or "").lower()

    return (
        width * height,
        sharpness,
        file_size,
        path,
    )


class ImageQualityFilteringService:
    """Filter blurry images and retain the best representative of duplicates."""

    def __init__(self, config: ImageQualityFilteringConfig):
        config.validate()
        self.config = config

    def analyze(self, source: str | Path) -> dict[str, Any]:
        started = time.perf_counter()

        source = Path(source).expanduser().resolve()
        output_dir = Path(self.config.output_dir).expanduser().resolve()

        method_dir = output_dir / "methods" / "image_quality_filtering"
        curated_dir = method_dir / "curated_dataset"

        method_dir.mkdir(parents=True, exist_ok=True)

        source_root = source if source.is_dir() else None

        records: list[dict[str, Any]] = []
        skipped: list[str] = []

        with InputService() as input_service:
            result = input_service.process(source)

            skipped = [str(path) for path in result.skipped_files]

            for path in result.image_paths:
                record = {
                    "original_path": str(path),
                    "original_relative_path": relative_path(
                        path, source_root
                    ),
                    "filename": path.name,
                    "size_bytes": None,
                    "width": None,
                    "height": None,
                    "format": None,
                    "modified_time": None,
                    "sha256": None,
                    "phash": None,
                    "blur_score": None,
                    "is_potentially_blurry": None,
                    "status": "pending",
                    "removal_reason": None,
                    "duplicate_group_id": None,
                    "retained_representative": None,
                    "curated_path": None,
                    "error": None,
                }

                try:
                    stat = path.stat()

                    record["size_bytes"] = stat.st_size
                    record["modified_time"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S",
                        time.localtime(stat.st_mtime),
                    )

                    with Image.open(path) as image:
                        oriented = ImageOps.exif_transpose(image)
                        record["width"], record["height"] = oriented.size
                        record["format"] = image.format

                    record["sha256"] = sha256_file(
                        path, self.config.hash_chunk_size
                    )

                    record["phash"] = perceptual_hash(
                        path, self.config.phash_hash_size
                    )

                    record["blur_score"] = laplacian_variance(path)

                    record["is_potentially_blurry"] = (
                        record["blur_score"] < self.config.blur_threshold
                    )

                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["status"] = "removed"
                    record["removal_reason"] = "processing_error"

                records.append(record)

        usable = [r for r in records if r["error"] is None]

        blurry = [
            r for r in usable
            if r["is_potentially_blurry"]
        ]

        sharp = [
            r for r in usable
            if not r["is_potentially_blurry"]
        ]

        # Detect exact and perceptual duplicates among non-blurry images.
        parent = list(range(len(sharp)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]

            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)

            if root_a != root_b:
                parent[root_b] = root_a

        hashes = [
            (record, imagehash.hex_to_hash(record["phash"]))
            for record in sharp
        ]

        pairs: list[dict[str, Any]] = []

        for i, (record_a, hash_a) in enumerate(hashes):
            for j in range(i + 1, len(hashes)):
                record_b, hash_b = hashes[j]

                same_bytes = (
                    record_a["sha256"] == record_b["sha256"]
                )

                distance = int(hash_a - hash_b)

                if same_bytes or distance <= self.config.phash_threshold:
                    union(i, j)

                    pairs.append({
                        "file_a": record_a["original_path"],
                        "file_b": record_b["original_path"],
                        "sha256_equal": same_bytes,
                        "phash_distance": distance,
                        "phash_threshold": self.config.phash_threshold,
                        "match_type": (
                            "exact_and_perceptual"
                            if same_bytes
                            else "perceptual"
                        ),
                    })

        components: dict[int, list[dict[str, Any]]] = {}

        for index, record in enumerate(sharp):
            components.setdefault(find(index), []).append(record)

        duplicate_groups: list[dict[str, Any]] = []
        grouped_paths: set[str] = set()

        group_number = 0

        for members in components.values():
            if len(members) < 2:
                continue

            group_number += 1
            group_id = f"DUP-{group_number:04d}"

            # Always select the highest-ranked image in the group.
            winner = max(members, key=quality_rank)

            paths = [member["original_path"] for member in members]
            grouped_paths.update(paths)

            duplicate_groups.append({
                "duplicate_group_id": group_id,
                "files": paths,
                "retained_representative": winner["original_path"],
                "selection_rule": (
                    "highest pixel count, then highest Laplacian "
                    "variance, then largest file size, then stable path"
                ),
            })

            for member in members:
                member["duplicate_group_id"] = group_id

                if member is winner:
                    member["status"] = "retained"
                    member["retained_representative"] = None
                else:
                    member["status"] = "removed"
                    member["removal_reason"] = "duplicate_lower_quality"
                    member["retained_representative"] = (
                        winner["original_path"]
                    )

        # Keep sharp images that are not duplicates.
        for record in sharp:
            if record["original_path"] not in grouped_paths:
                record["status"] = "retained"

        # Blurry images are always removed.
        for record in blurry:
            record["status"] = "removed"
            record["removal_reason"] = "blurry"

        # Link blurry images to a similar retained sharp image, if found.
        for record in blurry:
            blurry_hash = imagehash.hex_to_hash(record["phash"])
            candidates = []

            for other in sharp:
                if other["status"] != "retained":
                    continue

                other_hash = imagehash.hex_to_hash(other["phash"])
                distance = int(blurry_hash - other_hash)

                if distance <= self.config.phash_threshold:
                    candidates.append((
                        distance,
                        quality_rank(other),
                        other["original_path"],
                    ))

            if candidates:
                # Closest perceptual match first. If tied, prefer quality.
                candidates.sort(key=lambda item: (
                    item[0],
                    tuple(-x if isinstance(x, (int, float)) else x
                          for x in item[1][:3]),
                    item[2].lower(),
                ))

                record["retained_representative"] = candidates[0][2]

        retained = [
            record for record in records
            if record["status"] == "retained"
        ]

        removed = [
            record for record in records
            if record["status"] == "removed"
        ]

        # Rebuild only the generated output folder.
        # The original input dataset is never deleted or modified.
        if curated_dir.exists():
            shutil.rmtree(curated_dir)

        curated_dir.mkdir(parents=True, exist_ok=True)

        for record in retained:
            source_file = Path(record["original_path"])
            relative = Path(record["original_relative_path"])

            if relative.is_absolute() or ".." in relative.parts:
                relative = Path(source_file.name)

            target = curated_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                suffix = hashlib.sha256(
                    str(source_file).encode()
                ).hexdigest()[:8]

                target = target.with_name(
                    f"{target.stem}_{suffix}{target.suffix}"
                )

            shutil.copy2(source_file, target)
            record["curated_path"] = str(target)

        elapsed = time.perf_counter() - started

        report = {
            "service": "image-quality-filtering",
            "methods": {
                "blur_detection": "Laplacian variance",
                "exact_duplicates": "SHA-256 equality",
                "near_duplicates": "pHash Hamming distance",
                "quality_ranking": [
                    "pixel_count",
                    "laplacian_variance",
                    "file_size",
                    "stable_path",
                ],
            },
            "settings": {
                "blur_threshold": self.config.blur_threshold,
                "phash_hash_size": self.config.phash_hash_size,
                "phash_threshold": self.config.phash_threshold,
                "output_dir": str(output_dir),
            },
            "summary": {
                "images_discovered": len(records),
                "images_analyzed": len(usable),
                "images_with_errors": sum(
                    bool(record["error"]) for record in records
                ),
                "skipped_by_input_service": len(skipped),
                "images_flagged_blurry": len(blurry),
                "duplicate_groups": len(duplicate_groups),
                "duplicate_candidate_pairs": len(pairs),
                "images_retained": len(retained),
                "images_removed": len(removed),
                "elapsed_seconds": round(elapsed, 3),
            },
            "duplicate_groups": duplicate_groups,
            "images": records,
            "removed_images": removed,
            "skipped_files": skipped,
        }

        report_path = method_dir / "image_quality_filtering_report.json"
        removed_csv = method_dir / "removed_images.csv"
        pairs_csv = method_dir / "duplicate_pairs.csv"

        report["report_paths"] = {
            "json": str(report_path),
            "removed_csv": str(removed_csv),
            "duplicate_pairs_csv": str(pairs_csv),
            "curated_dataset": str(curated_dir),
        }

        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        removal_fields = [
            "original_path",
            "original_relative_path",
            "filename",
            "sha256",
            "phash",
            "width",
            "height",
            "format",
            "size_bytes",
            "modified_time",
            "blur_score",
            "is_potentially_blurry",
            "removal_reason",
            "duplicate_group_id",
            "retained_representative",
            "error",
        ]

        with removed_csv.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=removal_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(removed)

        pair_fields = [
            "file_a",
            "file_b",
            "sha256_equal",
            "phash_distance",
            "phash_threshold",
            "match_type",
        ]

        with pairs_csv.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=pair_fields)
            writer.writeheader()
            writer.writerows(pairs)

        return report


def main() -> None:
    source_text = input(
        "Image file, directory, ZIP, or TAR path: "
    ).strip().strip('"')

    if not source_text:
        print("No input path supplied.")
        return

    output_text = input(
        "Output directory [output_dir]: "
    ).strip().strip('"')

    try:
        service = ImageQualityFilteringService(
            ImageQualityFilteringConfig(
                output_dir=Path(output_text or "output_dir")
            )
        )

        report = service.analyze(Path(source_text))

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return

    print("\nIMAGE QUALITY FILTERING SUMMARY")
    print("=" * 45)

    for key, value in report["summary"].items():
        print(f"{key}: {value}")

    for key, value in report["report_paths"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
