
from __future__ import annotations

import csv
import json
import shutil
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

try:
    from src.services.inputService import InputService
except ImportError:
    from inputService import InputService


@dataclass(frozen=True)
class AIDetectionConfig:
    output_dir: Path
    ai_threshold: float = 0.60
    real_threshold: float = 0.40
    batch_size: int = 4
    model_id: str = "abacusai/Smaug-AI-Image-Detector"

    def validate(self) -> None:
        if not 0 <= self.real_threshold < self.ai_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= real_threshold < ai_threshold <= 1."
            )

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")


class AIDetectionService:
    """Detect likely AI-generated images using Smaug.

    Uses batched inference to improve throughput and the available
    GPU when supported by the installed model and PyTorch.

    Original input images are never modified or deleted.
    """

    def __init__(self, config: AIDetectionConfig):
        config.validate()
        self.config = config
        self._detector = None

        self.report_dir = (
            Path(config.output_dir)
            / "methods"
            / "ai_detection"
        )
        self.ai_output_dir = self.report_dir / "ai_generated"

    def _load_model(self):
        """Load the model once and reuse it for every batch."""
        if self._detector is not None:
            return self._detector

        print("Loading Smaug AI Image Detector...")

        model_dir = snapshot_download(
            repo_id=self.config.model_id
        )

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from smaug_image_detect import SmaugImageDetect

        # Smaug automatically selects an available accelerator.
        self._detector = SmaugImageDetect.from_pretrained(
            model_dir
        )

        print("Model loaded.")
        return self._detector

    @staticmethod
    def _to_probability(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()

        score = float(value)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Invalid AI probability: {score}"
            )

        return score

    def _classify(self, score: float) -> str:
        if score >= self.config.ai_threshold:
            return "likely_ai_generated"

        if score <= self.config.real_threshold:
            return "likely_real"

        return "uncertain"

    def _predict_batch(
        self, paths: list[Path]
    ) -> list[float]:
        """Run one batch through the shared detector."""
        detector = self._load_model()

        scores = detector(
            [str(path) for path in paths],
            batch_size=self.config.batch_size,
        )

        if len(scores) != len(paths):
            raise ValueError(
                "Model returned a different number of scores "
                "than input images."
            )

        return [
            self._to_probability(score)
            for score in scores
        ]

    def _copy_ai_image(
        self,
        source: Path,
        index: int,
    ) -> str:
        """Copy a flagged image without changing the original."""
        self.ai_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Prefix avoids overwriting images with identical filenames.
        destination = self.ai_output_dir / (
            f"{index:06d}_{source.name}"
        )

        shutil.copy2(source, destination)

        return str(destination)

    def analyze(self, source: str | Path) -> dict[str, Any]:
        started = time.perf_counter()

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        records: list[dict[str, Any]] = []
        skipped_files: list[str] = []
        processed = 0

        # Keep this context open until all images have been scored
        # and flagged images copied. Archive files are temporary.
        with InputService() as input_service:
            input_result = input_service.process(source)
            image_paths = input_result.image_paths
            skipped_files = input_result.skipped_files

            total = len(image_paths)

            for start in range(
                0, total, self.config.batch_size
            ):
                batch = image_paths[
                    start:start + self.config.batch_size
                ]

                batch_started = time.perf_counter()

                try:
                    scores = self._predict_batch(batch)

                    batch_results = list(zip(batch, scores))

                except Exception as batch_error:
                    # A failed batch should not automatically make
                    # every image in that batch fail.
                    print(
                        f"Batch failed ({type(batch_error).__name__}); "
                        "retrying images individually."
                    )

                    batch_results = []

                    for path in batch:
                        try:
                            scores = self._predict_batch([path])
                            batch_results.append(
                                (path, scores[0])
                            )
                        except Exception as exc:
                            batch_results.append((path, exc))

                for path, result in batch_results:
                    processed += 1

                    record: dict[str, Any] = {
                        "file": str(path),
                        "file_name": path.name,
                        "ai_probability": None,
                        "classification": "error",
                        "ai_copy": None,
                        "error": None,
                    }

                    if isinstance(result, Exception):
                        record["error"] = (
                            f"{type(result).__name__}: {result}"
                        )
                    else:
                        score = result
                        classification = self._classify(score)

                        record["ai_probability"] = round(
                            score, 6
                        )
                        record["classification"] = classification

                        if classification == "likely_ai_generated":
                            try:
                                record["ai_copy"] = (
                                    self._copy_ai_image(
                                        path,
                                        processed,
                                    )
                                )
                            except Exception as exc:
                                record["error"] = (
                                    "AI image detected, but copying "
                                    f"failed: {type(exc).__name__}: {exc}"
                                )

                    records.append(record)

                elapsed_batch = (
                    time.perf_counter() - batch_started
                )

                print(
                    f"Processed {processed}/{total} images "
                    f"({elapsed_batch:.2f}s for this batch)"
                )

        successful = [
            record for record in records
            if record["ai_probability"] is not None
        ]

        ai_records = [
            record for record in successful
            if record["classification"] == "likely_ai_generated"
        ]

        report = {
            "service": "ai-image-detection-service",
            "model": self.config.model_id,
            "settings": {
                "ai_threshold": self.config.ai_threshold,
                "real_threshold": self.config.real_threshold,
                "batch_size": self.config.batch_size,
                "inference_mode": "batched",
            },
            "summary": {
                "images_discovered": len(records),
                "images_scored": len(successful),
                "images_with_errors": (
                    len(records) - len(successful)
                ),
                "likely_ai_generated": len(ai_records),
                "ai_images_copied": sum(
                    record["ai_copy"] is not None
                    for record in ai_records
                ),
                "ai_image_copy_errors": sum(
                    record["ai_copy"] is None
                    and record["error"] is not None
                    for record in ai_records
                ),
                "likely_real": sum(
                    record["classification"] == "likely_real"
                    for record in successful
                ),
                "uncertain": sum(
                    record["classification"] == "uncertain"
                    for record in successful
                ),
                "elapsed_seconds": round(
                    time.perf_counter() - started, 3
                ),
            },
            "output_directories": {
                "ai_generated": str(self.ai_output_dir),
            },
            "images": records,
            "skipped_files": skipped_files,
        }

        json_path = (
            self.report_dir / "ai_detection_report.json"
        )
        json_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        csv_path = (
            self.report_dir / "ai_detection_report.csv"
        )

        fieldnames = [
            "file",
            "file_name",
            "ai_probability",
            "classification",
            "ai_copy",
            "error",
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
            writer.writerows(records)

        report["report_paths"] = {
            "json": str(json_path),
            "csv": str(csv_path),
        }

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
        config = AIDetectionConfig(
            output_dir=Path(output_text or "output_dir"),
            batch_size=4,
        )

        service = AIDetectionService(config)
        report = service.analyze(source_text)

    except Exception as exc:
        print(f"Detection failed: {type(exc).__name__}: {exc}")
        return

    print("\nAI IMAGE DETECTION SUMMARY")
    print("=" * 40)

    for key, value in report["summary"].items():
        print(f"{key}: {value}")

    print("\nAI-generated image folder:")
    print(report["output_directories"]["ai_generated"])

    print("\nReports:")
    print(report["report_paths"]["json"])
    print(report["report_paths"]["csv"])


if __name__ == "__main__":
    main()
