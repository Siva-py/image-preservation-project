from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from PIL import Image, ImageOps
import json 
import os
import shutil
import time

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
JPEG_QUALITIES = (95, 90, 85, 80, 75, 70, 60, 50, 40, 30)
RESIZE_SCALES = (1.0, 0.9)


@dataclass(frozen=True)
class CompressionConfig:
    input_dir: Path
    output_dir: Path
    max_size_kb: float = 500.0
    max_workers: int = min(4, os.cpu_count() or 1)
    # True = lossy compression; False = lossless compression
    lossy_compression: bool = True
    jpeg_qualities: tuple[int, ...] = JPEG_QUALITIES
    resize_scales: tuple[float, ...] = RESIZE_SCALES
    background_rgb: tuple[int, int, int] = (255, 255, 255)

    def validate(self):
        if self.max_size_kb <= 0:
            raise ValueError("max_size_kb must be positive.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if not self.jpeg_qualities or any(q < 1 or q > 95 for q in self.jpeg_qualities):
            raise ValueError("JPEG qualities must be between 1 and 95.")
        if not self.resize_scales or any(s <= 0 or s > 1 for s in self.resize_scales):
            raise ValueError("Resize scales must be > 0 and <= 1.")
        if len(self.background_rgb) != 3 or any(c < 0 or c > 255 for c in self.background_rgb):
            raise ValueError("background_rgb must be an RGB triple from 0 to 255.")


def size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def source_key(path: Path) -> str:
    # Different formats can have the same stem, e.g. photo.png and photo.webp.
    return f"{path.stem}_{path.suffix.lower().lstrip('.')}"


def load_rgb_image(path: Path, background_rgb: tuple[int, int, int]):
    image = Image.open(path)
    image_format = image.format
    if image_format not in SUPPORTED_FORMATS:
        image.close()
        raise ValueError(f"Unsupported image content format: {image_format!r}")

    # Do not silently discard animation frames.
    if getattr(image, "n_frames", 1) > 1:
        frames = image.n_frames
        image.close()
        raise ValueError(f"Animated images ({frames} frames) are not supported.")

    image.load()
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*background_rgb, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")

    return image, image_format


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(
        buffer, format="JPEG", quality=quality,
        optimize=True, progressive=True, subsampling=0 if quality >= 90 else 2
    )
    return buffer.getvalue()


def resize_image(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image.copy()
    width, height = image.size
    dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(dimensions, Image.Resampling.LANCZOS)


def write_candidate(output_dir: Path, method: str, filename: str, data: bytes) -> Path:
    folder = output_dir / "methods" / method
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(data)
    return path


def candidate_record(source: Path, image: Image.Image, original_kb: float,
                     output_dir: Path, scale: float, quality: int,
                     target_kb: float) -> dict[str, Any]:
    resized = resize_image(image, scale)
    data = encode_jpeg(resized, quality)
    method = f"jpeg_scale_{scale:.2f}_q{quality}"
    path = write_candidate(output_dir, method, f"{source_key(source)}.jpg", data)
    output_kb = size_kb(path)

    with Image.open(path) as output:
        output.load()
        dimensions = [output.width, output.height]

    return {
        "method": method,
        "output_file": str(path.relative_to(output_dir)),
        "output_format": "JPEG",
        "output_kb": round(output_kb, 2),
        "size_savings_kb": round(original_kb - output_kb, 2),
        "size_reduction_percent": round((original_kb - output_kb) / original_kb * 100, 2)
            if original_kb else 0.0,
        "original_dimensions": [image.width, image.height],
        "output_dimensions": dimensions,
        "scale": scale,
        "jpeg_quality": quality,
        "size_target_met": output_kb <= target_kb,
        "pixel_values_preserved": False,
    }


def _process_worker(args: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    source = Path(args[0])
    output_dir = Path(args[1])
    settings = args[2]
    original_kb = size_kb(source)

    result = {
        "file": source.name,
        "source_format": None,
        "original_kb": round(original_kb, 2),
        "original_dimensions": None,
        "output_format": "JPEG",
        "target_size_kb": settings["max_size_kb"],
        "methods_tested": [],
        "selected_method": None,
        "selected_output_file": None,
        "selected_output_kb": None,
        "selected_dimensions": None,
        "size_target_met": False,
        "size_savings_kb": None,
        "size_reduction_percent": None,
        "pixel_values_preserved": False,
        "status": "",
        "notes": [
            "JPEG encoding is lossy, including when the source was PNG or WebP.",
            f"Transparency is composited onto RGB background {tuple(settings['background_rgb'])}.",
        ],
    }

    try:
        image, source_format = load_rgb_image(source, tuple(settings["background_rgb"]))
        result["source_format"] = source_format
        result["original_dimensions"] = [image.width, image.height]

        originals_dir = output_dir / "originals"
        originals_dir.mkdir(parents=True, exist_ok=True)
        original_copy = originals_dir / source.name
        shutil.copy2(source, original_copy)
        result["original_copy"] = str(original_copy.relative_to(output_dir))

        candidates = []
        for scale in settings["resize_scales"]:
            for quality in settings["jpeg_qualities"]:
                candidates.append(candidate_record(
                    source, image, original_kb, output_dir, scale, quality,
                    settings["max_size_kb"]
                ))

        fitting = [c for c in candidates if c["size_target_met"]]
        if fitting:
            # First prefer retaining the most pixels, then the higher quality setting.
            best = max(fitting, key=lambda c: (
                c["output_dimensions"][0] * c["output_dimensions"][1],
                c["jpeg_quality"], -c["output_kb"]
            ))
        else:
            best = min(candidates, key=lambda c: c["output_kb"])

        selected_dir = output_dir / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_path = selected_dir / f"{source_key(source)}.jpg"
        shutil.copy2(output_dir / best["output_file"], selected_path)

        result.update({
            "methods_tested": candidates,
            "selected_method": best["method"],
            "selected_output_file": str(selected_path.relative_to(output_dir)),
            "selected_output_kb": round(size_kb(selected_path), 2),
            "selected_dimensions": best["output_dimensions"],
            "size_target_met": best["size_target_met"],
            "size_savings_kb": round(original_kb - size_kb(selected_path), 2),
            "size_reduction_percent": round(
                (original_kb - size_kb(selected_path)) / original_kb * 100, 2
            ) if original_kb else 0.0,
            "status": "target_met" if best["size_target_met"] else "target_not_met",
        })
        if not best["size_target_met"]:
            result["notes"].append("No tested candidate met the requested size target.")
        result["notes"].append(
            "Pixel values are not exactly preserved by JPEG re-encoding; assess visual quality separately."
        )
        image.close()
    except Exception as exc:
        result["status"] = "error"
        result["notes"].append(f"{type(exc).__name__}: {exc}")

    return result


class CompressionService:
    """Reusable batch/single-image compression service; selected outputs are JPEG."""

    def __init__(self, config: CompressionConfig):
        config.validate()
        self.config = config

    def discover_images(self) -> list[Path]:
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")
        return sorted(
            p for p in self.config.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _settings(self) -> dict[str, Any]:
        return {
            "max_size_kb": self.config.max_size_kb,
            "jpeg_qualities": list(self.config.jpeg_qualities),
            "resize_scales": list(self.config.resize_scales),
            "background_rgb": list(self.config.background_rgb),
        }

    def compress_image(self, source: str | Path) -> dict[str, Any]:
        source = Path(source)
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Accepted input formats: PNG, JPG/JPEG, WebP.")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return _process_worker((str(source), str(self.config.output_dir), self._settings()))

    def compress_batch(self) -> dict[str, Any]:
        sources = self.discover_images()
        if not sources:
            raise ValueError(f"No PNG, JPEG or WebP images found in {self.config.input_dir}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        args = [(str(p), str(self.config.output_dir), self._settings()) for p in sources]
        workers = min(self.config.max_workers, len(sources))

        if workers == 1:
            images = [_process_worker(arg) for arg in args]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                images = list(pool.map(_process_worker, args))

        elapsed = time.perf_counter() - started
        completed = [r for r in images if r["selected_output_kb"] is not None]
        originals_total = sum(r["original_kb"] for r in completed)
        selected_total = sum(r["selected_output_kb"] for r in completed)
        targets_met = sum(r["size_target_met"] for r in completed)

        report = {
            "service": "image-compression-service",
            "input_formats": ["PNG", "JPEG", "WebP"],
            "output_format": "JPEG",
            "settings": {
                "max_size_kb": self.config.max_size_kb,
                "max_workers": workers,
                "jpeg_qualities": list(self.config.jpeg_qualities),
                "resize_scales": list(self.config.resize_scales),
                "background_rgb": list(self.config.background_rgb),
            },
            "summary": {
                "images_discovered": len(sources),
                "images_completed": len(completed),
                "images_with_errors": len(sources) - len(completed),
                "targets_met": targets_met,
                "targets_not_met": len(completed) - targets_met,
                "original_total_kb": round(originals_total, 2),
                "selected_total_kb": round(selected_total, 2),
                "overall_size_reduction_percent": round(
                    (originals_total - selected_total) / originals_total * 100, 2
                ) if originals_total else 0.0,
                "elapsed_seconds": round(elapsed, 3),
                "average_seconds_per_image": round(elapsed / len(sources), 3),
            },
            "images": images,
        }
        report_path = self.config.output_dir / "compression_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report


def main():
    root = Path(__file__).resolve().parent[2]
    input_dir = r"C:\DM_Drive\Projects\Dissertation1 - Image Preservation\image-preservation-project\experiments\data\input"
    output_dir = r"C:\DM_Drive\Projects\Dissertation1 - Image Preservation\image-preservation-project\experiments\data\output"

    print("=" * 60)
    print("IMAGE COMPRESSION SERVICE")
    print("Inputs: PNG, JPEG/JPG, WebP | Selected outputs: JPEG")
    print("Note: JPEG encoding is lossy; transparent images use a white background.")
    print("=" * 60)

    try:
        raw_size = input("\nMaximum output size in KB [500]: ").strip()
        max_size = float(raw_size or "500")
        if max_size <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a positive number.")
        return

    default_workers = min(4, os.cpu_count() or 1)
    try:
        raw_workers = input(f"CPU workers [{default_workers}]: ").strip()
        workers = int(raw_workers or default_workers)
        if workers < 1:
            raise ValueError
    except ValueError:
        print("Invalid worker count; using default.")
        workers = default_workers

    service = CompressionService(CompressionConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        max_size_kb=max_size,
        max_workers=workers,
        lossy_compression = True,
    ))

    try:
        report = service.compress_batch()
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nError: {exc}")
        return

    summary = report["summary"]
    print("\n" + "=" * 60)
    print("COMPRESSION SUMMARY")
    print("=" * 60)
    print(f"Images discovered: {summary['images_discovered']}")
    print(f"Images completed:  {summary['images_completed']}")
    print(f"Targets met:       {summary['targets_met']}/{summary['images_completed']}")
    print(f"Original total:    {summary['original_total_kb']:.2f} KB")
    print(f"Selected total:    {summary['selected_total_kb']:.2f} KB")
    print(f"Overall reduction:{summary['overall_size_reduction_percent']:>8.2f}%")
    print(f"Elapsed time:      {summary['elapsed_seconds']:.3f} seconds")
    print(f"Report:            {output_dir / 'compression_report.json'}")
    print(f"Selected images:   {output_dir / 'selected'}")


if __name__ == "__main__":
    main()
