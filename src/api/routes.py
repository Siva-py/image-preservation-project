from __future__ import annotations
import os
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from fastapi.responses import FileResponse
from zipfile import ZIP_DEFLATED, ZipFile

from src.api.schemas import PipelineConfig

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def build_pipeline_command(
    input_dir: Path,
    config: PipelineConfig,
) -> list[str]:
    """Build a CLI command for the existing pipeline without modifying it."""

    output_dir = Path(config.output_dir).expanduser()

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir = output_dir.resolve()
    input_dir = input_dir.resolve()

    # Match the safety checks in src/main.py.
    try:
        output_dir.relative_to(input_dir)
        raise HTTPException(
            status_code=400,
            detail="Output directory cannot be inside the input directory.",
        )
    except ValueError:
        pass

    try:
        input_dir.relative_to(output_dir)
        raise HTTPException(
            status_code=400,
            detail="Input directory cannot be inside the output directory.",
        )
    except ValueError:
        pass

    command = [
        sys.executable,
        "-m",
        "src.main",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--max-size-kb",
        str(config.max_size_kb),
        "--retain-ratio",
        str(config.retention_rate),
        "--ai-threshold",
        str(config.ai_threshold),
        "--real-threshold",
        str(config.real_threshold),
        "--ai-batch-size",
        str(config.ai_batch_size),
    ]

    if config.skip_compression:
        command.append("--skip-compression")

    if config.lossless:
        command.append("--lossless")

    if config.detect_ai:
        command.append("--detect-ai")

    if config.max_workers is not None:
        command.extend(["--max-workers", str(config.max_workers)])

    if config.num_workers is not None:
        command.extend(["--num-workers", str(config.num_workers)])

    return command



async def execute_pipeline(input_dir: Path, config: PipelineConfig) -> dict:
    command = build_pipeline_command(input_dir, config)

    # Run the existing pipeline in a worker thread.
    # This avoids Windows asyncio subprocess compatibility issues.
    process = await asyncio.to_thread(
        subprocess.run,
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )


    output_text = process.stdout or ""
    error_text = process.stderr or ""

    output_dir = Path(config.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    report_path = output_dir / "pipeline_report.json"
    dashboard_report_path = output_dir / "dashboard_report.json"

    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "The image-processing pipeline failed.",
                "return_code": process.returncode,
                "stdout_tail": output_text[-4000:],
                "stderr_tail": error_text[-4000:],
                "report_path": str(report_path),
            },
        )

    report_data = None

    if report_path.is_file():
        import json

        try:
            report_data = json.loads(
                report_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            report_data = None

    dashboard_data = None
    if dashboard_report_path.is_file():
        try:
            dashboard_data = json.loads(
                dashboard_report_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            dashboard_data = None

    return {
        "status": "completed",
        "message": "Image-processing pipeline completed successfully.",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir),
        "report_path": (
            str(report_path) if report_path.is_file() else None
        ),
        "dashboard_report_path": (
            str(dashboard_report_path)
            if dashboard_report_path.is_file() else None
        ),
        "dashboard": dashboard_data,
        "pipeline_report": report_data,
        "stdout": output_text[-4000:],
    }



def get_output_directory(configured_output_dir: str = "data/output") -> Path:
    output_dir = Path(configured_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir.resolve()


def list_dataset_images(kind: str) -> list[str]:
    output_dir = get_output_directory()
    if kind == "original":
        image_dir = output_dir / "original_images"
    elif kind == "final":
        image_dir = output_dir / "final_dataset"
    else:
        raise HTTPException(status_code=404, detail="Unknown dataset image category.")

    if not image_dir.is_dir():
        return []

    return sorted(
        path.name for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


@router.get("/dataset-images/{kind}")
async def get_dataset_image_list(kind: str):
    """List original input images or final selected images from the latest upload run."""
    if kind not in {"original", "final"}:
        raise HTTPException(status_code=404, detail="Use 'original' or 'final'.")
    return list_dataset_images(kind)


@router.get("/dataset-images/{kind}/{filename}")
async def get_dataset_image(kind: str, filename: str):
    """Serve a single image from the original or final dataset gallery."""
    if kind not in {"original", "final"}:
        raise HTTPException(status_code=404, detail="Use 'original' or 'final'.")

    safe_filename = Path(filename).name
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    directory_name = "original_images" if kind == "original" else "final_dataset"
    image_path = (get_output_directory() / directory_name / safe_filename).resolve()
    allowed_root = (get_output_directory() / directory_name).resolve()

    try:
        image_path.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image path.")

    if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(image_path)


@router.get("/final-dataset/download")
async def download_final_dataset():
    """Create and return a ZIP containing the latest final selected dataset."""
    output_dir = get_output_directory()
    final_dir = output_dir / "final_dataset"

    if not final_dir.is_dir():
        raise HTTPException(status_code=404, detail="The final dataset is not available yet.")

    image_paths = [
        path for path in final_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if not image_paths:
        raise HTTPException(status_code=404, detail="The final dataset contains no images.")

    zip_path = output_dir / "final_dataset.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for image_path in image_paths:
            archive.write(image_path, arcname=image_path.name)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="final_dataset.zip",
    )


@router.post("/process-folder")
async def process_folder(folder_path: str, config: PipelineConfig):
    """Run the existing pipeline on an existing local image folder."""

    input_dir = Path(folder_path).expanduser()

    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    input_dir = input_dir.resolve()

    if not input_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail="The supplied folder path does not exist or is not a directory.",
        )

    if not any(
        path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        for path in input_dir.rglob("*")
    ):
        raise HTTPException(
            status_code=400,
            detail="The folder contains no supported image files.",
        )

    return await execute_pipeline(input_dir, config)


@router.post("/upload-images")
async def upload_images(
    files: Annotated[list[UploadFile], File(...)],
    skip_compression: Annotated[bool, Form()] = False,
    detect_ai: Annotated[bool, Form()] = False,
    lossless: Annotated[bool, Form()] = False,
    max_size_kb: Annotated[float, Form(gt=0)] = 500.0,
    retention_rate: Annotated[float, Form(gt=0.0, le=1.0)] = 0.80,
    ai_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.60,
    real_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.40,
    ai_batch_size: Annotated[int, Form(ge=1)] = 4,
    max_workers: Annotated[int | None, Form(ge=1)] = None,
    num_workers: Annotated[int | None, Form(ge=0)] = None,
    output_dir: Annotated[str, Form()] = "data/output",
):
    """Save multiple uploaded images temporarily and run the pipeline."""

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one image.",
        )

    if not 0 <= real_threshold < ai_threshold <= 1:
        raise HTTPException(
            status_code=422,
            detail="Thresholds must satisfy 0 <= real_threshold < ai_threshold <= 1.",
        )

    with tempfile.TemporaryDirectory(prefix="image_preservation_api_") as temp:
        input_dir = Path(temp) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        try:
            for index, upload in enumerate(files, start=1):
                filename = Path(upload.filename or "").name

                if (
                    not filename
                    or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported image filename: {upload.filename}",
                    )

                destination = input_dir / f"{index:06d}_{filename}"

                with destination.open("wb") as output:
                    shutil.copyfileobj(upload.file, output)

                saved_count += 1

            config = PipelineConfig(
                skip_compression=skip_compression,
                detect_ai=detect_ai,
                lossless=lossless,
                max_size_kb=max_size_kb,
                retention_rate=retention_rate,
                ai_threshold=ai_threshold,
                real_threshold=real_threshold,
                ai_batch_size=ai_batch_size,
                max_workers=max_workers,
                num_workers=num_workers,
                output_dir=output_dir,
            )

            result = await execute_pipeline(input_dir, config)

            # Keep a copy of this run's original uploads for the Results gallery.
            # This happens after the pipeline because src/main.py may clean output_dir.
            output_root = get_output_directory(output_dir)
            original_gallery_dir = output_root / "original_images"
            if original_gallery_dir.exists():
                shutil.rmtree(original_gallery_dir)
            original_gallery_dir.mkdir(parents=True, exist_ok=True)

            for image_path in sorted(input_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in ALLOWED_EXTENSIONS:
                    shutil.copy2(image_path, original_gallery_dir / image_path.name)

            result["input_count"] = saved_count
            result["original_images_url"] = "/api/dataset-images/original"
            result["final_images_url"] = "/api/dataset-images/final"
            result["final_dataset_download_url"] = "/api/final-dataset/download"
            return result

        finally:
            for upload in files:
                await upload.close()
