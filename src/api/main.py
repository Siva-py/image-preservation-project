from src.api.routes import router
from fastapi import FastAPI

app = FastAPI(
    title="Image Preservation API",
    description=(
        "API for the multi-stage image dataset reduction "
        "and information-preservation pipeline."
    ),
    version="1.0.0",
)

app.include_router(router, prefix="/api")

@app.get("/")
def home():
    """Check whether the API is running."""
    return {
        "message": "Image Preservation API is running!",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    """Return the API health status."""
    return {
        "status": "healthy",
    }
