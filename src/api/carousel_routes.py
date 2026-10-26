
"""API endpoints for discovering and serving carousel images."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAROUSEL_FOLDER = (
    PROJECT_ROOT / "frontend" / "assets" / "carousel"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".avif",
    ".bmp",
}


@router.get("/api/carousel-images", response_model=list[str])
def list_carousel_images() -> list[str]:
    """Return every supported image filename in the carousel folder."""

    if not CAROUSEL_FOLDER.is_dir():
        return []

    return sorted(
        path.name
        for path in CAROUSEL_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


@router.get("/api/carousel-images/{filename}")
def get_carousel_image(filename: str) -> FileResponse:
    """Serve an image from the carousel folder."""

    image_path = (
        CAROUSEL_FOLDER / filename
    ).resolve()

    # Prevent access to files outside the carousel folder.
    try:
        image_path.relative_to(CAROUSEL_FOLDER.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image path",
        )

    if (
        not image_path.is_file()
        or image_path.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise HTTPException(
            status_code=404,
            detail="Carousel image not found",
        )

    return FileResponse(image_path)
