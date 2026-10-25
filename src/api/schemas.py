from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Customizable settings supported by the existing pipeline."""

    # Pipeline stages
    skip_compression: bool = False
    detect_ai: bool = False

    # Compression configuration
    lossless: bool = False
    max_size_kb: float = Field(default=500.0, gt=0)

    # Information preservation
    retention_rate: float = Field(
        default=0.80,
        gt=0.0,
        le=1.0,
        description="Fraction of readable quality-filtered images to retain.",
    )

    # AI detection configuration
    ai_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    real_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    ai_batch_size: int = Field(default=4, ge=1)

    # Processing configuration
    max_workers: int | None = Field(default=None, ge=1)
    num_workers: int | None = Field(default=None, ge=0)

    # Output directory
    output_dir: str = "data/output"


class PipelineResponse(BaseModel):
    """Standard response returned by the API."""

    status: str
    message: str
    input_count: int | None = None
    output_dir: str | None = None
    report_path: str | None = None
    details: dict | None = None
