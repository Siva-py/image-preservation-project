# Image Compression Service

## Accepted inputs / selected outputs
- Inputs: PNG, JPEG/JPG, WebP
- Selected outputs: JPEG
- Transparent images are composited onto a white background by default.
- JPEG re-encoding is lossy. Original inputs are copied into `data/output/originals/`.
- Animated images are reported as unsupported instead of silently dropping frames.

## Project layout

Place `compression_service.py` in your project root:

```text
image-preservation-project/
├── compression_service.py
└── data/
    ├── input/       # PNG, JPG/JPEG, WebP
    └── output/      # created automatically
```

## Run

Activate your existing virtual environment, then run:

```powershell
python compression_service.py
```

## Output folders
- `data/output/originals/`: unchanged source copies
- `data/output/methods/`: candidate JPEGs, grouped by scale and quality
- `data/output/selected/`: chosen JPEG per source
- `data/output/compression_report.json`: settings, candidates, outcomes, and runtime

## Use it from another Python module

```python
from pathlib import Path
from compression_service import CompressionConfig, CompressionService

service = CompressionService(CompressionConfig(
    input_dir=Path("data/input"),
    output_dir=Path("data/output"),
    max_size_kb=500,
    max_workers=4,
))
report = service.compress_batch()
print(report["summary"])
```

For a single image:

```python
result = service.compress_image(Path("data/input/example.png"))
print(result["selected_output_file"])
```

## Academic note
A common JPEG output makes size comparisons easier, but conversion from PNG/WebP to JPEG is itself lossy and removes transparency by compositing onto a background. Keep the original inputs and document this normalization step in your methodology. A strict lossless baseline should be evaluated separately.
