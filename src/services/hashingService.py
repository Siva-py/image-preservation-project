from __future__ import annotations
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps

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
    """Calculate a SHA-256 digest without loading the entire file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


def perceptual_hash(path: Path, hash_size: int) -> str:
    """Calculate a perceptual hash after applying EXIF orientation."""

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return str(imagehash.phash(image, hash_size=hash_size))


class HashingService:
    """Detect exact duplicates and near-duplicate image candidates."""

    def __init__(self, config: HashingConfig):
        config.validate()
        self.config = config

    def analyze(self, source: str | Path) -> dict[str, Any]:
        started = time.perf_counter()

        output_dir = Path(self.config.output_dir)
        report_dir = output_dir / "methods" / "hashing"
        report_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        skipped_files: list[Any] = []

        with InputService() as input_service:
            input_result = input_service.process(source)

            image_paths = input_result.image_paths
            skipped_files = input_result.skipped_files

            # Calculate hashes for every discovered image.
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
                        path,
                        chunk_size=self.config.hash_chunk_size,
                    )

                    record["phash"] = perceptual_hash(
                        path,
                        hash_size=self.config.phash_hash_size,
                    )

                except Exception as exc:
                    record["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                records.append(record)

            usable = [
                record
                for record in records
                if record["error"] is None
            ]

            # --------------------------------------------------
            # 1. Detect exact duplicates using SHA-256.
            # --------------------------------------------------

            exact_groups: dict[str, list[str]] = {}

            for record in usable:
                exact_groups.setdefault(
                    record["sha256"], []
                ).append(record["file"])

            exact_duplicate_groups = [
                {
                    "sha256": digest,
                    "files": files,
                }
                for digest, files in exact_groups.items()
                if len(files) > 1
            ]

            # --------------------------------------------------
            # 2. Compare perceptual hashes.
            #
            # Only near-duplicate candidates are stored.
            # All unique pairs are still compared.
            # --------------------------------------------------

            near_duplicate_pairs: list[dict[str, Any]] = []

            # Convert hashes once instead of repeatedly.
            hashed_records = [
                (
                    record,
                    imagehash.hex_to_hash(record["phash"]),
                )
                for record in usable
            ]

            total_pairs = (
                len(hashed_records) * (len(hashed_records) - 1) // 2
            )

            for i, (first, hash_a) in enumerate(hashed_records):
                for second, hash_b in hashed_records[i + 1:]:
                    distance = int(hash_a - hash_b)

                    # Ignore pairs outside the configured threshold.
                    if distance > self.config.phash_threshold:
                        continue

                    near_duplicate_pairs.append({
                        "file_a": first["file"],
                        "file_b": second["file"],
                        "sha256_equal": (
                            first["sha256"] == second["sha256"]
                        ),
                        "phash_distance": distance,
                        "phash_threshold": self.config.phash_threshold,
                        "is_near_duplicate_candidate": True,
                    })

        elapsed = time.perf_counter() - started

        # ------------------------------------------------------
        # 3. Build the JSON report.
        # ------------------------------------------------------

        report: dict[str, Any] = {
            "service": "image-hashing-service",

            "methods": {
                "exact_duplicate": "SHA-256 file digest equality",
                "near_duplicate": (
                    "Perceptual hash (pHash) Hamming distance"
                ),
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
                    len(group["files"])
                    for group in exact_duplicate_groups
                ),
                "image_pairs_compared": total_pairs,
                "near_duplicate_candidate_pairs": len(
                    near_duplicate_pairs
                ),
                "elapsed_seconds": round(elapsed, 3),
            },

            "exact_duplicate_groups": exact_duplicate_groups,
            "near_duplicate_pairs": near_duplicate_pairs,
            "images": records,
            "skipped_files": skipped_files,
        }

        # ------------------------------------------------------
        # 4. Save JSON report.
        # ------------------------------------------------------

        json_path = report_dir / "hashing_report.json"

        json_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        # ------------------------------------------------------
        # 5. Save ONLY near-duplicate candidates to CSV.
        # ------------------------------------------------------

        csv_path = report_dir / "hashing_pairs.csv"

        fieldnames = [
            "file_a",
            "file_b",
            "sha256_equal",
            "phash_distance",
            "phash_threshold",
            "is_near_duplicate_candidate",
        ]

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(near_duplicate_pairs)

        report["report_paths"] = {
            "json": str(json_path),
            "pairs_csv": str(csv_path),
        }

        # Include report paths in the JSON as well.
        json_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        return report


def main() -> None:
    """Command-line entry point."""

    source_text = input(
        "Image file, directory, ZIP, or TAR path: "
    ).strip().strip('"')

    if not source_text:
        print("No input path supplied.")
        return

    output_text = input(
        "Output directory [output_dir]: "
    ).strip().strip('"')

    output_dir = Path(output_text or "output_dir")

    try:
        service = HashingService(
            HashingConfig(output_dir=output_dir)
        )

        report = service.analyze(Path(source_text))

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return

    summary = report["summary"]

    print("\nIMAGE HASHING SUMMARY")
    print("=" * 45)

    print(
        f"Images discovered:       "
        f"{summary['images_discovered']}"
    )

    print(
        f"Images hashed:           "
        f"{summary['images_hashed_successfully']}"
    )

    print(
        f"Image processing errors: "
        f"{summary['images_with_errors']}"
    )

    print(
        f"Exact duplicate groups:  "
        f"{summary['exact_duplicate_groups']}"
    )

    print(
        f"Total pairs compared:    "
        f"{summary['image_pairs_compared']}"
    )

    print(
        f"Near-duplicate pairs:    "
        f"{summary['near_duplicate_candidate_pairs']}"
    )

    print(
        f"Elapsed time:            "
        f"{summary['elapsed_seconds']} seconds"
    )

    print(
        f"JSON report:             "
        f"{report['report_paths']['json']}"
    )

    print(
        f"Pairs CSV:               "
        f"{report['report_paths']['pairs_csv']}"
    )


if __name__ == "__main__":
    main()
