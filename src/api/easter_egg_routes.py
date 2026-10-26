from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["Easter Egg"])

# Project root / frontend / assets / friends
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRIENDS_DIR = PROJECT_ROOT / "frontend" / "assets" / "friends"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
}


@router.get("/easter-egg/friends")
def get_friends_images():
    """Return all supported images inside the friends folder."""

    if not FRIENDS_DIR.exists():
        return {
            "status": "success",
            "count": 0,
            "images": [],
        }

    images = sorted(
        [
            file.name
            for file in FRIENDS_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=str.lower,
    )

    return {
        "status": "success",
        "count": len(images),
        "images": images,
    }