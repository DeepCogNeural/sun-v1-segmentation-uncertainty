# Source Module Map

This directory keeps the original research-code layout to avoid breaking legacy imports. The public documentation explains the module boundaries instead of renaming modules aggressively.

## Top-level interfaces

- `SegmentationMap.py`: high-level wrapper for images, segmentation fits, probability maps, and crop/session state.
- `Session.py`: neural session loader and neuron-coordinate utilities for Utah-array or Neuropixels-style recordings.
- `Analysis.py`: aligns `SegmentationMap` outputs with neural-session tables and runs regression-style analyses.

## Model backend

- `seg/segment.py`: fitting entry points and segmentation-map generation.
- `seg/models_deep_seg.py`: CNN-feature segmentation models.
- `seg/gmm_prior.py`, `seg/smm_prior.py`: Gaussian and Student mixture-model priors.
- `seg/pyramid.py`: image pyramid / multi-scale helpers.

## Analysis utilities

- `dynamics.py`: uncertainty dynamics, convergence metrics, likelihood traces, and same-segment probabilities.
- `analysis/single_neuron_analysis.py`: joins segmentation probabilities to single-neuron coordinates.
- `analysis/pairwise_analysis.py`: pairwise-analysis helpers.
- `plot_utils.py`: lightweight plotting helpers for receptive fields and image overlays.
- `import_utils.py`: data loading and local path configuration.
