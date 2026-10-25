import shutil
import tarfile
import tempfile
import zipfile

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

SUPPORTED_ARCHIVE_EXTENSIONS = {
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
}


@dataclass(frozen=True)
class InputConfig:
    max_archive_files: int = 10_000
    max_extracted_size_mb: int = 2048
    max_image_pixels: int = 100_000_000
    recursive: bool = True


@dataclass
class InputResult:
    image_paths: list[Path] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


class InputService:
    """Discover, extract, and validate supported image inputs."""

    def __init__(self, config: InputConfig | None = None):
        self.config = config or InputConfig()
        self._temp_dir = None

    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="msqfip_input_"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    @staticmethod
    def _is_supported_image(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    @staticmethod
    def _is_archive(path: Path) -> bool:
        name = path.name.lower()
        return any(
            name.endswith(ext)
            for ext in SUPPORTED_ARCHIVE_EXTENSIONS
        )

    def process(self, source: str | Path) -> InputResult:
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Input does not exist: {source}")

        if source.is_symlink():
            raise ValueError("Symbolic-link inputs are not supported.")

        source = source.resolve()

        if source.is_file() and self._is_supported_image(source):
            candidates = [source]

        elif source.is_dir():
            iterator = (
                source.rglob("*")
                if self.config.recursive
                else source.iterdir()
            )
            candidates = [
                p for p in iterator
                if p.is_file()
                and not p.is_symlink()
                and self._is_supported_image(p)
            ]

        elif source.is_file() and self._is_archive(source):
            candidates = self._extract_archive(source)

        else:
            raise ValueError(
                f"Unsupported input: {source}. "
                "Provide an image, directory, ZIP, or TAR archive."
            )

        result = InputResult()

        for path in sorted(candidates, key=lambda p: str(p).lower()):
            if self._validate_image(path):
                result.image_paths.append(path)
            else:
                result.skipped_files.append(str(path))

        if not result.image_paths:
            raise ValueError(
                "No valid supported images were found in the input."
            )

        return result

    def _validate_image(self, path: Path) -> bool:
        try:
            with Image.open(path) as image:
                if image.width * image.height > self.config.max_image_pixels:
                    return False

                if getattr(image, "is_animated", False):
                    return False

                image.verify()

            return True

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            return False

    def _extract_archive(self, archive: Path) -> list[Path]:
        if self._temp_dir is None:
            raise RuntimeError(
                "Use InputService as a context manager when processing "
                "archives: with InputService() as service:"
            )

        destination = Path(self._temp_dir.name) / "extracted"
        destination.mkdir(parents=True, exist_ok=True)

        if archive.name.lower().endswith(".zip"):
            return self._extract_zip(archive, destination)

        return self._extract_tar(archive, destination)

    def _safe_destination(
        self,
        name: str,
        destination: Path,
    ) -> Path | None:
        posix_path = PurePosixPath(name)
        windows_path = PureWindowsPath(name)

        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            return None

        target = (destination / Path(*posix_path.parts)).resolve()

        if not target.is_relative_to(destination.resolve()):
            return None

        return target

    def _extract_zip(
        self,
        archive: Path,
        destination: Path,
    ) -> list[Path]:
        extracted = []
        total_size = 0

        with zipfile.ZipFile(archive) as zip_file:
            members = [
                item for item in zip_file.infolist()
                if not item.is_dir()
            ]

            if len(members) > self.config.max_archive_files:
                raise ValueError("Archive contains too many files.")

            for item in members:
                target = self._safe_destination(
                    item.filename,
                    destination,
                )

                if target is None:
                    continue

                if item.file_size < 0:
                    continue

                total_size += item.file_size

                if total_size > self.config.max_extracted_size_mb * 1024**2:
                    raise ValueError(
                        "Archive exceeds the extraction size limit."
                    )

                # Reject Unix symbolic links stored in ZIP archives.
                unix_mode = item.external_attr >> 16
                if (unix_mode & 0o170000) == 0o120000:
                    continue

                if not self._is_supported_image(target):
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)

                with zip_file.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                extracted.append(target)

        return extracted

    def _extract_tar(
        self,
        archive: Path,
        destination: Path,
    ) -> list[Path]:
        extracted = []
        total_size = 0

        with tarfile.open(archive, mode="r:*") as tar_file:
            members = [
                item for item in tar_file.getmembers()
                if item.isfile()
            ]

            if len(members) > self.config.max_archive_files:
                raise ValueError("Archive contains too many files.")

            for item in members:
                target = self._safe_destination(
                    item.name,
                    destination,
                )

                if target is None:
                    continue

                if item.size < 0:
                    continue

                total_size += item.size

                if total_size > self.config.max_extracted_size_mb * 1024**2:
                    raise ValueError(
                        "Archive exceeds the extraction size limit."
                    )

                if not self._is_supported_image(target):
                    continue

                source_file = tar_file.extractfile(item)

                if source_file is None:
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)

                with source_file, target.open("wb") as dst:
                    shutil.copyfileobj(source_file, dst)

                extracted.append(target)

        return extracted
