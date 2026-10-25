"""Run the complete image-preservation pipeline.

Expected project layout:
    data/input/                         Original images
    data/output/compression/            Compression artifacts and selected images
    data/output/quality_filtering/       Quality-filtered curated dataset
    data/output/information_preservation/ Information-preservation outputs
    data/output/ai_detection/            Optional final AI-detection outputs

Run from the project root:
    python main.py
    python main.py --lossless
    python main.py --input path/to/images --output path/to/output
    python main.py --skip-compression
    python main.py --detect-ai
    python main.py --clean-output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from src.services.compressionService import CompressionConfig, CompressionService
from src.services.qualityFilteringService import (
    ImageQualityFilteringConfig,
    ImageQualityFilteringService,
)
from src.services.informationPreservationService import (
    run_service as run_information_preservation,
)
from src.services.aiDetectionService import (
    AIDetectionConfig,
    AIDetectionService,
)


def parse_args() -> argparse.Namespace:
    # main.py lives in src/ and is run from the repository root with python -m src.main.
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the image-preservation pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "input",
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "output",
        help="Root directory for generated service outputs.",
    )
    parser.add_argument(
        "--max-size-kb",
        type=float,
        default=500.0,
        help="Target maximum compressed image size in KB (default: 500).",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Use lossless compression instead of the default lossy mode.",
    )
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Skip compression and pass original images to quality filtering.",
    )
    parser.add_argument(
        "--detect-ai",
        action="store_true",
        help="Run optional AI-image detection after information preservation.",
    )
    parser.add_argument(
        "--ai-threshold",
        type=float,
        default=0.60,
        help="Probability at or above which an image is flagged as likely AI-generated.",
    )
    parser.add_argument(
        "--real-threshold",
        type=float,
        default=0.40,
        help="Probability at or below which an image is classified as likely real.",
    )
    parser.add_argument(
        "--ai-batch-size",
        type=int,
        default=4,
        help="Batch size for AI-image detection.",
    )
    parser.add_argument(
        "--retain-ratio",
        type=float,
        default=0.80,
        help="Fraction of readable quality-filtered images to retain (default: 0.80).",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the configured output directory before starting.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Compression worker count (defaults to the service setting).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Information-preservation image-loader workers (defaults to service setting).",
    )
    return parser.parse_args()


def write_pipeline_report(output_root: Path, results: dict[str, Any]) -> Path:
    report_path = output_root / "pipeline_report.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def contains_files(directory: Path) -> bool:
    return directory.is_dir() and any(
        path.is_file() for path in directory.rglob("*")
    )


def prepare_final_dataset(
    final_dataset_dir: Path,
    source_images: list[Path],
) -> list[Path]:
    """Copy the current run's final images into one stable output directory.

    Only files directly inside this dedicated final_dataset directory are
    cleared. Source images and other pipeline outputs are left untouched.
    """
    if final_dataset_dir.exists():
        shutil.rmtree(final_dataset_dir)
    final_dataset_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: list[Path] = []
    seen_names: set[str] = set()

    for source in source_images:
        source = Path(source)
        if not source.is_file():
            continue

        # Avoid overwriting if two source folders contain the same filename.
        target_name = source.name
        if target_name.lower() in seen_names:
            stem, suffix = source.stem, source.suffix
            counter = 2
            while f"{stem}_{counter}{suffix}".lower() in seen_names:
                counter += 1
            target_name = f"{stem}_{counter}{suffix}"

        target = final_dataset_dir / target_name
        shutil.copy2(source, target)
        seen_names.add(target_name.lower())
        copied_paths.append(target)

    return copied_paths


def resolve_selected_images(
    information_report: dict[str, Any],
    information_dir: Path,
) -> list[Path]:
    """Resolve selected images from the information-preservation output.

    The service's ``copied_images`` field has varied by implementation, so
    support common report shapes and fall back to the output directory.
    """
    candidates: list[Path] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            candidate = Path(value)
            if candidate.is_file():
                candidates.append(candidate)
        elif isinstance(value, Path):
            if value.is_file():
                candidates.append(value)
        elif isinstance(value, dict):
            # Prefer explicit selected-image/path fields, but recursively
            # inspect values so nested service report formats are supported.
            for key, nested in value.items():
                if key.lower() in {
                    "selected_images", "selected_paths", "image_paths",
                    "paths", "files", "selected_files",
                }:
                    collect(nested)
                elif isinstance(nested, (dict, list, tuple)):
                    collect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collect(nested)

    collect(information_report.get("copied_images", {}))
    collect(information_report.get("selected_images", []))
    collect(information_report.get("selected_paths", []))

    # Keep only unique paths in original order.
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    if unique:
        return unique

    # Common layout: selected images are copied beneath the information
    # preservation output directory. Avoid using report files as images.
    excluded_names = {
        "pipeline_report.json",
        "information_preservation_report.json",
        "report.json",
    }
    fallback = [
        path
        for path in information_dir.rglob("*")
        if path.is_file()
        and path.name.lower() not in excluded_names
        and path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    ]
    return sorted(fallback)


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        print(
            f"ERROR: Input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 2
    if args.max_size_kb <= 0:
        print("ERROR: --max-size-kb must be positive.", file=sys.stderr)
        return 2
    if not 0 < args.retain_ratio <= 1:
        print("ERROR: --retain-ratio must be in (0, 1].", file=sys.stderr)
        return 2
    if args.ai_batch_size < 1:
        print("ERROR: --ai-batch-size must be at least 1.", file=sys.stderr)
        return 2
    if not 0 <= args.real_threshold < args.ai_threshold <= 1:
        print(
            "ERROR: thresholds must satisfy "
            "0 <= --real-threshold < --ai-threshold <= 1.",
            file=sys.stderr,
        )
        return 2
    if args.max_workers is not None and args.max_workers < 1:
        print("ERROR: --max-workers must be at least 1.", file=sys.stderr)
        return 2
    if args.num_workers is not None and args.num_workers < 0:
        print("ERROR: --num-workers cannot be negative.", file=sys.stderr)
        return 2

    # Output must not live inside input, and input must not live inside output.
    # The latter avoids accidentally processing generated outputs on reruns.
    try:
        output_root.relative_to(input_dir)
        print(
            "ERROR: Output directory cannot be inside the input directory. "
            "Use separate data/input and data/output folders.",
            file=sys.stderr,
        )
        return 2
    except ValueError:
        pass
    try:
        input_dir.relative_to(output_root)
        print(
            "ERROR: Input directory cannot be inside the output directory.",
            file=sys.stderr,
        )
        return 2
    except ValueError:
        pass

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    compression_dir = output_root / "compression"
    quality_dir = output_root / "quality_filtering"
    information_dir = output_root / "information_preservation"
    ai_detection_dir = output_root / "ai_detection"
    final_dataset_dir = output_root / "final_dataset"
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "compression_enabled": not args.skip_compression,
        "compression_mode": (
            "skipped"
            if args.skip_compression
            else ("lossless" if args.lossless else "lossy")
        ),
        "ai_detection_enabled": bool(args.detect_ai),
        "retain_ratio": args.retain_ratio,
    }

    try:
        print("Processing started...")
        print("\n" + "=" * 60)

        # STAGE 1: Compression (optional)
        if args.skip_compression:
            print("[STAGE 1/4] IMAGE COMPRESSION — SKIPPED")
            compression_input = input_dir
            results["compression"] = {
                "status": "skipped",
                "reason": "disabled_by_user",
                "selected_dir": str(input_dir),
            }
        else:
            print("[STAGE 1/4] IMAGE COMPRESSION")
            print("=" * 60)
            print("Status: RUNNING")

            compression_config_kwargs: dict[str, Any] = {
                "input_dir": input_dir,
                "output_dir": compression_dir,
                "max_size_kb": args.max_size_kb,
                "lossy_compression": not args.lossless,
            }
            if args.max_workers is not None:
                compression_config_kwargs["max_workers"] = args.max_workers

            compression_service = CompressionService(
                CompressionConfig(**compression_config_kwargs)
            )
            compression_report = compression_service.compress_batch()

            results["compression"] = {
                "status": "completed",
                "report_path": compression_report.get("report_path"),
                "summary": compression_report.get("summary", {}),
                "selected_dir": str(compression_dir / "selected"),
            }

            compression_summary = compression_report.get("summary", {})
            print("\n" + "-" * 60)
            print("IMAGE COMPRESSION COMPLETED SUCCESSFULLY")
            print("-" * 60)
            print(
                f"Images processed: "
                f"{compression_summary.get('images_completed', 0)}"
            )
            print(
                f"Images with errors: "
                f"{compression_summary.get('images_with_errors', 0)}"
            )
            print(
                "Size reduction: "
                f"{compression_summary.get('overall_size_reduction_percent', 0):.2f}%"
            )
            print(f"Report: {compression_report.get('report_path')}")
            print("Status: COMPLETED ✓")

            compression_input = compression_dir / "selected"
            if not contains_files(compression_input):
                raise RuntimeError(
                    "Compression did not produce any selected images; "
                    "quality filtering cannot continue."
                )

        # STAGE 2: Quality filtering
        print("\n" + "=" * 60)
        print("[STAGE 2/4] QUALITY FILTERING")
        print("=" * 60)
        print("Status: RUNNING")

        quality_service = ImageQualityFilteringService(
            ImageQualityFilteringConfig(output_dir=quality_dir)
        )
        quality_report = quality_service.analyze(compression_input)
        quality_paths = quality_report.get("report_paths", {})
        curated_dir_value = quality_paths.get("curated_dataset")
        if not curated_dir_value:
            raise RuntimeError(
                "Quality filtering report did not provide a curated_dataset path."
            )

        curated_dir = Path(curated_dir_value)
        if not contains_files(curated_dir):
            raise RuntimeError(
                "Quality filtering produced no curated images; "
                "information preservation cannot continue."
            )

        results["quality_filtering"] = {
            "status": "completed",
            "report_paths": quality_paths,
            "summary": quality_report.get("summary", {}),
        }
        print("Quality filtering completed.")
        print(f"Curated dataset: {curated_dir}")

        # STAGE 3: Information preservation
        print("\n" + "=" * 60)
        print("[STAGE 3/4] INFORMATION PRESERVATION")
        print("=" * 60)
        print("Status: RUNNING")

        information_kwargs: dict[str, Any] = {
            "input_dir": curated_dir,
            "output_dir": information_dir,
            "retain_ratio": args.retain_ratio,
            "copy_images": True,
        }
        if args.num_workers is not None:
            information_kwargs["num_workers"] = args.num_workers

        information_report = run_information_preservation(**information_kwargs)
        selected_image_paths = resolve_selected_images(
            information_report,
            information_dir,
        )
        if not selected_image_paths:
            raise RuntimeError(
                "Information preservation did not expose any selected image "
                "paths, and no selected images were found in its output directory."
            )

        results["information_preservation"] = {
            "status": "completed",
            "report_path": information_report.get("report_path"),
            "selected_count": information_report.get("selected_count"),
            "readable_images": information_report.get("readable_images"),
            "copied_images": information_report.get("copied_images", {}),
            "resolved_selected_count": len(selected_image_paths),
            "selected_images": [str(path) for path in selected_image_paths],
        }
        print("Information preservation completed.")
        print(f"Selected images: {len(selected_image_paths)}")

        # STAGE 4: AI detection (optional, runs only after selection)
        print("\n" + "=" * 60)
        print("[STAGE 4/4] AI IMAGE DETECTION")
        print("=" * 60)

        if args.detect_ai:
            print("Status: RUNNING")
            ai_service = AIDetectionService(
                AIDetectionConfig(
                    output_dir=output_root,
                    ai_threshold=args.ai_threshold,
                    real_threshold=args.real_threshold,
                    batch_size=args.ai_batch_size,
                )
            )
            ai_report = ai_service.analyze(selected_image_paths)

            ai_retained_paths = [
                Path(record["final_image"])
                for record in ai_report.get("images", [])
                if record.get("final_image")
            ]
            final_image_paths = prepare_final_dataset(
                final_dataset_dir,
                ai_retained_paths,
            )

            results["ai_detection"] = {
                "status": "completed",
                "summary": ai_report.get("summary", {}),
                "report_paths": ai_report.get("report_paths", {}),
                "output_directories": ai_report.get("output_directories", {}),
            }
            results["final_dataset"] = {
                "source": "ai_detection",
                "directory": str(final_dataset_dir),
                "image_count": len(final_image_paths),
                "image_paths": [str(path) for path in final_image_paths],
            }
            print("AI detection completed.")
            print(f"Final dataset directory: {final_dataset_dir}")
            print(f"Final dataset images: {len(final_image_paths)}")
        else:
            print("Status: SKIPPED (disabled by user)")
            results["ai_detection"] = {
                "status": "skipped",
                "reason": "disabled_by_user",
            }
            final_image_paths = prepare_final_dataset(
                final_dataset_dir,
                selected_image_paths,
            )
            results["final_dataset"] = {
                "source": "information_preservation",
                "directory": str(final_dataset_dir),
                "image_count": len(final_image_paths),
                "image_paths": [str(path) for path in final_image_paths],
            }
            print(f"Final dataset directory: {final_dataset_dir}")
            print(f"Final dataset images: {len(final_image_paths)}")

        report_path = write_pipeline_report(output_root, results)
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Compression output: {compression_dir if not args.skip_compression else 'skipped'}")
        print(f"Quality output: {quality_dir}")
        print(f"Information-preservation output: {information_dir}")
        print(f"AI detection: {'enabled' if args.detect_ai else 'skipped'}")
        print(f"Final dataset images: {results['final_dataset']['image_count']}")
        print(f"Pipeline report: {report_path}")
        return 0

    except Exception as exc:
        results["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            report_path = write_pipeline_report(output_root, results)
            print(
                f"\nPipeline failed. Partial report: {report_path}",
                file=sys.stderr,
            )
        except Exception:
            pass
        print(
            f"\nPipeline failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
