from src.api.routes import router
from src.api.carousel_routes import router as carousel_router
from src.api.easter_egg_routes import router as easter_egg_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Image Preservation API",
    description=(
        "API for the multi-stage image dataset reduction "
        "and information-preservation pipeline."
    ),
    version="1.0.0",
)

# Local development only: supports VS Code Live Server and direct file:// previews.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_methods=["GET","POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(carousel_router)
app.include_router(easter_egg_router, prefix="/api")

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
