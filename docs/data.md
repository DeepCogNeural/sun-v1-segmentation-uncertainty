# Data and Reproducibility Notes

This repository intentionally does not include private neural recordings, internal experiment folders, generated pickles, or large image datasets.

## Data used by the full research pipeline

- **Natural images / segmentations:** BSDS-style natural-image assets and human segmentation labels.
- **Neural sessions:** V1 recording sessions stored as local `.mat` or `.pkl` files.
- **Generated artifacts:** segmentation maps, probability maps, entropy maps, analysis tables, figures, and intermediate pickles.

## Local path configuration

`src/import_utils.py` supports environment variables so a reviewer can import the code without editing source files:

```bash
export SUN_DATA_ROOT=/path/to/sun_workspace
export SUN_EXP_NAME=EXP150_NatImages_NeuroPixels
```

`SUN_DATA_ROOT` should point to the directory that contains experiment folders such as `EXP150_NatImages_NeuroPixels/` and the optional `data/BSR/BSDS500/...` tree. `SUN_SESSION_PATH` is still accepted as a backward-compatible alias. If neither variable is set, the code falls back to repo-local placeholder paths. Missing private files should not block lightweight documentation or demo usage.

## Reproducibility modes

| Mode | Command | Data needed |
| --- | --- | --- |
| Smoke demo | `python examples/demo_pipeline.py --out outputs/demo` | None; creates synthetic image and probability maps. |
| Segmentation model fit | `PYTHONPATH=src python -m seg.segment` or custom script | BSDS-style images and installed Torch/Torchvision. |
| Neural alignment | custom script using `Session`, `SegmentationMap`, and `Analysis` | Local neural session files and image/session index mapping. |

## What not to commit

- private neural data
- raw BSDS or internal image datasets
- generated `.pkl`, `.csv`, `.png`, `.pdf`, or `.html` outputs
- local virtual environments
- `__pycache__` and compiled Python artifacts
