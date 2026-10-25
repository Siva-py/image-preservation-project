from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class AIDetectionConfig:
    """Configuration for the optional final AI-image detection stage."""

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
    """Detect likely AI-generated images from an already-selected image list.

    This service is designed to run AFTER information preservation.
    Original/selected source images are never modified or deleted.

    Likely AI-generated images are copied to ``ai_generated/`` and excluded
    from ``final_images/``. Likely-real, uncertain, and inference-error images
    are retained in ``final_images/`` by default.
    """

    def __init__(self, config: AIDetectionConfig):
        config.validate()
        self.config = config
        self._detector = None

        self.report_dir = (
            Path(config.output_dir) / "methods" / "ai_detection"
        )
        self.ai_output_dir = self.report_dir / "ai_generated"
        self.final_output_dir = self.report_dir / "final_images"

    def _prepare_output_dirs(self) -> None:
        """Clear only this service's dedicated result folders for a clean run."""
        self.report_dir.mkdir(parents=True, exist_ok=True)

        for directory in (self.ai_output_dir, self.final_output_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load the detector once and reuse it for all batches."""
        if self._detector is not None:
            return self._detector

        print("Loading Smaug AI Image Detector...")
        model_dir = snapshot_download(repo_id=self.config.model_id)

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from smaug_image_detect import SmaugImageDetect

        # The model library chooses its execution device internally.
        self._detector = SmaugImageDetect.from_pretrained(model_dir)
        print("Model loaded.")
        return self._detector

    @staticmethod
    def _to_probability(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()

        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid AI probability: {score}")
        return score

    def _classify(self, score: float) -> str:
        if score >= self.config.ai_threshold:
            return "likely_ai_generated"
        if score <= self.config.real_threshold:
            return "likely_real"
        return "uncertain"

    def _predict_batch(self, paths: list[Path]) -> list[float]:
        detector = self._load_model()
        scores = detector(
            [str(path) for path in paths],
            batch_size=self.config.batch_size,
        )

        # Some libraries may return a single scalar for a one-image batch.
        if len(paths) == 1 and not isinstance(scores, (list, tuple)):
            scores = [scores]

        if len(scores) != len(paths):
            raise ValueError(
                "Model returned a different number of scores "
                "than input images."
            )

        return [self._to_probability(score) for score in scores]

    @staticmethod
    def _copy_image(
        source: Path,
        destination_dir: Path,
        index: int,
    ) -> str:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{index:06d}_{source.name}"
        shutil.copy2(source, destination)
        return str(destination)

    def _retain_image(
        self,
        record: dict[str, Any],
        source: Path,
        index: int,
    ) -> None:
        """Retain a selected image; record copy failures without hiding them."""
        try:
            record["final_image"] = self._copy_image(
                source, self.final_output_dir, index
            )
        except Exception as exc:
            message = (
                f"Final image copy failed: {type(exc).__name__}: {exc}"
            )
            record["error"] = (
                f'{record["error"]}; {message}'
                if record.get("error")
                else message
            )

    def analyze(self, image_paths: Sequence[str | Path]) -> dict[str, Any]:
        """Analyze selected image paths and return report plus retained paths.

        Pass the selected image paths returned by information preservation.
        This method does not scan the original dataset or invoke InputService.
        """
        started = time.perf_counter()
        paths = [Path(path) for path in image_paths]

        invalid_paths = [str(path) for path in paths if not path.is_file()]
        if invalid_paths:
            raise FileNotFoundError(
                "These selected image paths do not exist or are not files: "
                + ", ".join(invalid_paths[:10])
            )

        self._prepare_output_dirs()
        records: list[dict[str, Any]] = []
        processed = 0
        total = len(paths)

        for start in range(0, total, self.config.batch_size):
            batch = paths[start : start + self.config.batch_size]
            batch_started = time.perf_counter()

            try:
                scores = self._predict_batch(batch)
                batch_results: list[tuple[Path, float | Exception]] = list(
                    zip(batch, scores)
                )
            except Exception as batch_error:
                print(
                    f"Batch failed ({type(batch_error).__name__}); "
                    "retrying images individually."
                )
                batch_results = []
                for path in batch:
                    try:
                        individual_scores = self._predict_batch([path])
                        batch_results.append((path, individual_scores[0]))
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
                    "final_image": None,
                    "error": None,
                }

                if isinstance(result, Exception):
                    record["error"] = f"{type(result).__name__}: {result}"
                    # Conservative fallback: don't drop an image because
                    # the detector failed to score it.
                    self._retain_image(record, path, processed)
                else:
                    score = result
                    classification = self._classify(score)
                    record["ai_probability"] = round(score, 6)
                    record["classification"] = classification

                    if classification == "likely_ai_generated":
                        try:
                            record["ai_copy"] = self._copy_image(
                                path, self.ai_output_dir, processed
                            )
                        except Exception as exc:
                            record["error"] = (
                                "AI image detected, but copying to the "
                                f"AI results folder failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            # If separation fails, retain it rather than
                            # silently losing a selected image.
                            self._retain_image(record, path, processed)
                    else:
                        self._retain_image(record, path, processed)

                records.append(record)

            elapsed_batch = time.perf_counter() - batch_started
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

        json_path = self.report_dir / "ai_detection_report.json"
        csv_path = self.report_dir / "ai_detection_report.csv"

        report: dict[str, Any] = {
            "service": "ai-image-detection-service",
            "model": self.config.model_id,
            "settings": {
                "ai_threshold": self.config.ai_threshold,
                "real_threshold": self.config.real_threshold,
                "batch_size": self.config.batch_size,
                "inference_mode": "batched_with_individual_fallback",
            },
            "summary": {
                "images_received": total,
                "images_scored": len(successful),
                "images_with_inference_errors": total - len(successful),
                "likely_ai_generated": len(ai_records),
                "ai_images_copied": sum(
                    record["ai_copy"] is not None for record in ai_records
                ),
                "ai_image_copy_errors": sum(
                    record["classification"] == "likely_ai_generated"
                    and record["ai_copy"] is None
                    for record in records
                ),
                "likely_real": sum(
                    record["classification"] == "likely_real"
                    for record in successful
                ),
                "uncertain": sum(
                    record["classification"] == "uncertain"
                    for record in successful
                ),
                "final_images_retained": sum(
                    record["final_image"] is not None for record in records
                ),
                "final_image_copy_errors": sum(
                    record["final_image"] is None for record in records
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            "output_directories": {
                "ai_generated": str(self.ai_output_dir),
                "final_images": str(self.final_output_dir),
            },
            "report_paths": {
                "json": str(json_path),
                "csv": str(csv_path),
            },
            "images": records,
        }

        json_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        fieldnames = [
            "file",
            "file_name",
            "ai_probability",
            "classification",
            "ai_copy",
            "final_image",
            "error",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print("\nAI IMAGE DETECTION COMPLETE")
        print("=" * 40)
        for key, value in report["summary"].items():
            print(f"{key}: {value}")
        print(f"JSON report: {json_path}")
        print(f"CSV report:  {csv_path}")

        return report


def main() -> None:
    """Standalone test entry point; the FastAPI pipeline should call analyze()."""
    try:
        from src.services.inputService import InputService
    except ImportError:
        from inputService import InputService

    source_text = input(
        "Image file, directory, ZIP, or TAR path: "
    ).strip().strip('"')
    if not source_text:
        print("No input path supplied.")
        return

    output_text = input("Output directory [output_dir]: ").strip().strip('"')

    try:
        config = AIDetectionConfig(
            output_dir=Path(output_text or "output_dir"),
            batch_size=4,
        )
        service = AIDetectionService(config)

        # Keep InputService's context open while temporary/archive paths
        # are being used. This wrapper is for standalone testing only.
        with InputService() as input_service:
            input_result = input_service.process(source_text)
            report = service.analyze(input_result.image_paths)

    except Exception as exc:
        print(f"Detection failed: {type(exc).__name__}: {exc}")
        return

    print("\nAI-generated image folder:")
    print(report["output_directories"]["ai_generated"])
    print("\nFinal retained images folder:")
    print(report["output_directories"]["final_images"])


if __name__ == "__main__":
    main()
