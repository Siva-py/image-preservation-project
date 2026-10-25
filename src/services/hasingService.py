from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps

try:
    from src.services.inputService import InputService
except ImportError:
    from src.services.inputService import InputService


@dataclass(frozen=True)
class HashingConfig:
    """Settings for exact and perceptual image hashing."""

    output_dir: Path
    phash_hash_size: int = 16
    phash_threshold: int = 24
    hash_chunk_size: int = 1024 * 1024

    def validate(self) -> None:
        if self.phash_hash_size < 1:
            raise ValueError("phash_hash_size must be at least 1.")
        if self.phash_threshold < 0:
            raise ValueError("phash_threshold cannot be negative.")
        if self.hash_chunk_size < 1:
            raise ValueError("hash_chunk_size must be positive.")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path, hash_size: int) -> str:
    """Return a pHash hex string after applying EXIF orientation."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return str(imagehash.phash(image, hash_size=hash_size))


class HashingService:
    """Detect exact duplicates and candidate visual near-duplicates.

    This service never deletes, moves, compresses, or modifies input images.
    """

    def __init__(self, config: HashingConfig):
        config.validate()
        self.config = config

    def analyze(self, source: str | Path) -> dict[str, Any]:
        started = time.perf_counter()
        output_dir = Path(self.config.output_dir)
        report_dir = output_dir / "methods" / "hashing"
        report_dir.mkdir(parents=True, exist_ok=True)

        # InputService validates images and supports directories and archives.
        with InputService() as input_service:
            input_result = input_service.process(source)
            image_paths = input_result.image_paths
            skipped_files = input_result.skipped_files

            records: list[dict[str, Any]] = []
            for path in image_paths:
                record: dict[str, Any] = {
                    "file": str(path),
                    "file_name": path.name,
                    "size_bytes": None,
                    "sha256": None,
                    "phash": None,
                    "error": None,
                }
                try:
                    record["size_bytes"] = path.stat().st_size
                    record["sha256"] = sha256_file(
                        path, chunk_size=self.config.hash_chunk_size
                    )
                    record["phash"] = perceptual_hash(
                        path, hash_size=self.config.phash_hash_size
                    )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)

            usable = [r for r in records if r["error"] is None]

            # Exact duplicates are files with matching SHA-256 digests.
            exact_groups: dict[str, list[str]] = {}
            for record in usable:
                exact_groups.setdefault(record["sha256"], []).append(record["file"])

            exact_duplicate_groups = [
                {"sha256": digest, "files": files}
                for digest, files in exact_groups.items()
                if len(files) > 1
            ]

            # Compare every unique pair using pHash Hamming distance (O(n^2)).
            pair_records: list[dict[str, Any]] = []
            near_duplicate_pairs: list[dict[str, Any]] = []
            for i, first in enumerate(usable):
                hash_a = imagehash.hex_to_hash(first["phash"])
                for second in usable[i + 1:]:
                    hash_b = imagehash.hex_to_hash(second["phash"])
                    distance = int(hash_a - hash_b)
                    pair = {
                        "file_a": first["file"],
                        "file_b": second["file"],
                        "sha256_equal": first["sha256"] == second["sha256"],
                        "phash_distance": distance,
                        "phash_threshold": self.config.phash_threshold,
                        "is_near_duplicate_candidate": (
                            distance <= self.config.phash_threshold
                        ),
                    }
                    pair_records.append(pair)
                    if pair["is_near_duplicate_candidate"]:
                        near_duplicate_pairs.append(pair)

        elapsed = time.perf_counter() - started
        report: dict[str, Any] = {
            "service": "image-hashing-service",
            "methods": {
                "exact_duplicate": "SHA-256 file digest equality",
                "near_duplicate": "Perceptual hash (pHash) Hamming distance",
            },
            "settings": {
                "output_dir": str(self.config.output_dir),
                "phash_hash_size": self.config.phash_hash_size,
                "phash_threshold": self.config.phash_threshold,
                "hash_chunk_size": self.config.hash_chunk_size,
            },
            "summary": {
                "images_discovered": len(records),
                "images_hashed_successfully": len(usable),
                "images_with_errors": len(records) - len(usable),
                "skipped_by_input_service": len(skipped_files),
                "exact_duplicate_groups": len(exact_duplicate_groups),
                "exact_duplicate_files_in_groups": sum(
                    len(group["files"]) for group in exact_duplicate_groups
                ),
                "image_pairs_compared": len(pair_records),
                "near_duplicate_candidate_pairs": len(near_duplicate_pairs),
                "elapsed_seconds": round(elapsed, 3),
            },
            "exact_duplicate_groups": exact_duplicate_groups,
            "near_duplicate_pairs": near_duplicate_pairs,
            "images": records,
            "skipped_files": skipped_files,
        }

        json_path = report_dir / "hashing_report.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        csv_path = report_dir / "hashing_pairs.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            fieldnames = [
                "file_a", "file_b", "sha256_equal", "phash_distance",
                "phash_threshold", "is_near_duplicate_candidate",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_records)

        report["report_paths"] = {
            "json": str(json_path),
            "pairs_csv": str(csv_path),
        }
        return report


def main() -> None:
    """Command-line entry point for testing the hashing service."""
    source_text = input("Image file, directory, ZIP, or TAR path: ").strip().strip('"')
    if not source_text:
        print("No input path supplied.")
        return

    output_text = input("Output directory [output_dir]: ").strip().strip('"')
    output_dir = Path(output_text or "output_dir")

    try:
        service = HashingService(HashingConfig(output_dir=output_dir))
        report = service.analyze(Path(source_text))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return

    summary = report["summary"]
    print("\nIMAGE HASHING SUMMARY")
    print("=" * 40)
    print(f"Images discovered:       {summary['images_discovered']}")
    print(f"Images hashed:           {summary['images_hashed_successfully']}")
    print(f"Image processing errors: {summary['images_with_errors']}")
    print(f"Exact duplicate groups:  {summary['exact_duplicate_groups']}")
    print(f"Pairs compared:          {summary['image_pairs_compared']}")
    print(f"Near-duplicate pairs:    {summary['near_duplicate_candidate_pairs']}")
    print(f"Elapsed time:            {summary['elapsed_seconds']} seconds")
    print(f"JSON report:             {report['report_paths']['json']}")
    print(f"Pairs CSV:               {report['report_paths']['pairs_csv']}")


if __name__ == "__main__":
    main()
