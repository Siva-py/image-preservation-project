"""Run the complete image-preservation pipeline.

Expected project layout:
    data/input/                         Original images
    data/output/compression/            Compression artifacts and selected images
    data/output/quality_filtering/       Quality-filtered curated dataset
    data/output/information_preservation/ Information-preservation outputs

Run from the project root:
    python main.py
    python main.py --lossless
    python main.py --input path/to/images --output path/to/output
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
from src.services.informationPreservationService import run_service as run_information_preservation


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run the image-preservation pipeline.")
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


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if args.max_size_kb <= 0:
        print("ERROR: --max-size-kb must be positive.", file=sys.stderr)
        return 2
    if not 0 < args.retain_ratio <= 1:
        print("ERROR: --retain-ratio must be in (0, 1].", file=sys.stderr)
        return 2
    if args.max_workers is not None and args.max_workers < 1:
        print("ERROR: --max-workers must be at least 1.", file=sys.stderr)
        return 2
    if args.num_workers is not None and args.num_workers < 0:
        print("ERROR: --num-workers cannot be negative.", file=sys.stderr)
        return 2

    # Prevent the output from being placed inside the source tree. Services also
    # exclude their own output subtree, but separate roots are safer by design.
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

    if args.clean_output and output_root.exists():
        # Only remove the explicitly configured output root when the user opts in.
        shutil.rmtree(output_root)

    compression_dir = output_root / "compression"
    quality_dir = output_root / "quality_filtering"
    information_dir = output_root / "information_preservation"
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "compression_mode": "lossless" if args.lossless else "lossy",
    }

    try:
        print("Processing started....")
        print("\n" + "=" * 60)
        print("[STAGE 1/3] IMAGE COMPRESSION")
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

        # Record compression results
        results["compression"] = {
            "report_path": compression_report.get("report_path"),
            "summary": compression_report.get("summary", {}),
            "selected_dir": str(compression_dir / "selected"),
        }

        # Display completion status
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
            f"Size reduction: "
            f"{compression_summary.get('overall_size_reduction_percent', 0):.2f}%"
        )
        print(f"Report: {compression_report.get('report_path')}")
        print("Status: COMPLETED ✓")
        print("-" * 60 + "\n")

        compressed_input = compression_dir / "selected"
        if not compressed_input.is_dir() or not any(
            p.is_file() for p in compressed_input.rglob("*")
        ):
            raise RuntimeError(
                "Compression did not produce any selected images; "
                "quality filtering cannot continue."
            )

        print("\n[STAGE 2/3] QUALITY FILTERING")
        print("=" * 60)
        print("Status: RUNNING")

# Your existing quality filtering service call goes hereprint("\n[2/3] Quality filtering")
        quality_service = ImageQualityFilteringService(
            ImageQualityFilteringConfig(output_dir=quality_dir)
        )
        quality_report = quality_service.analyze(compressed_input)
        quality_paths = quality_report.get("report_paths", {})
        curated_dir_value = quality_paths.get("curated_dataset")
        if not curated_dir_value:
            raise RuntimeError(
                "Quality filtering report did not provide a curated_dataset path."
            )
        curated_dir = Path(curated_dir_value)
        if not curated_dir.is_dir() or not any(
            p.is_file() for p in curated_dir.rglob("*")
        ):
            raise RuntimeError(
                "Quality filtering produced no curated images; "
                "information preservation cannot continue."
            )
        results["quality_filtering"] = {
            "report_paths": quality_paths,
            "summary": quality_report.get("summary", {}),
        }

        print("\n[STAGE 3/3] Information preservation")
        information_kwargs: dict[str, Any] = {
            "input_dir": curated_dir,
            "output_dir": information_dir,
            "retain_ratio": args.retain_ratio,
            "copy_images": True,
        }
        if args.num_workers is not None:
            information_kwargs["num_workers"] = args.num_workers
        information_report = run_information_preservation(**information_kwargs)
        results["information_preservation"] = {
            "report_path": information_report.get("report_path"),
            "selected_count": information_report.get("selected_count"),
            "readable_images": information_report.get("readable_images"),
            "copied_images": information_report.get("copied_images", {}),
        }

        report_path = write_pipeline_report(output_root, results)
        print("\nPipeline completed successfully.")
        print(f"Compression output: {compression_dir}")
        print(f"Quality output: {quality_dir}")
        print(f"Information-preservation output: {information_dir}")
        print(f"Pipeline report: {report_path}")
        return 0

    except Exception as exc:
        results["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            report_path = write_pipeline_report(output_root, results)
            print(f"\nPipeline failed. Partial report: {report_path}", file=sys.stderr)
        except Exception:
            pass
        print(f"\nPipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
