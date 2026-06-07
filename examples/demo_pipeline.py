#!/usr/bin/env python3
"""Lightweight SUN pipeline demo with no private data.

The full research code uses private neural sessions and heavier model dependencies.
This script creates a synthetic image, constructs soft segment-probability maps,
computes entropy, and writes the same kinds of artifacts that the full pipeline
uses downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_synthetic_image(size: int = 96) -> np.ndarray:
    """Create a simple RGB image with three regions and an uncertain boundary."""
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = xx / max(size - 1, 1)
    image[..., 1] = yy / max(size - 1, 1)
    image[..., 2] = ((xx - yy) ** 2) / max((size - 1) ** 2, 1)
    image[: size // 2, : size // 2, 0] += 0.35
    image[size // 2 :, size // 2 :, 1] += 0.35
    return np.clip(image, 0.0, 1.0)


def make_probability_map(size: int = 96, temperature: float = 8.0) -> np.ndarray:
    """Create a three-segment soft assignment map shaped like a toy segmentation."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xx / max(size - 1, 1)) * 2.0 - 1.0
    y = (yy / max(size - 1, 1)) * 2.0 - 1.0
    logits = np.stack(
        [
            -temperature * (x + 0.35),
            temperature * (x - 0.35),
            -temperature * (np.abs(y) - 0.18),
        ],
        axis=0,
    )
    logits -= logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=0, keepdims=True)


def entropy(probability_map: np.ndarray) -> np.ndarray:
    """Compute per-pixel entropy from a K x H x W probability map."""
    eps = 1e-12
    return -(probability_map * np.log(probability_map + eps)).sum(axis=0)


def maybe_write_png(image: np.ndarray, probability_map: np.ndarray, entropy_map: np.ndarray, out_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    hard_segments = probability_map.argmax(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Synthetic image")
    axes[1].imshow(hard_segments, cmap="tab10")
    axes[1].set_title("Most likely segment")
    axes[2].imshow(entropy_map, cmap="magma")
    axes[2].set_title("Segmentation entropy")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out_dir / "demo_overview.png", dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-private-data SUN smoke demo.")
    parser.add_argument("--out", default="outputs/demo", help="Output directory.")
    parser.add_argument("--size", type=int, default=96, help="Synthetic image size.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = make_synthetic_image(args.size)
    probability_map = make_probability_map(args.size)
    entropy_map = entropy(probability_map)

    np.save(out_dir / "synthetic_image.npy", image)
    np.save(out_dir / "segment_probability_map.npy", probability_map)
    np.save(out_dir / "entropy_map.npy", entropy_map)
    wrote_png = maybe_write_png(image, probability_map, entropy_map, out_dir)

    summary = {
        "synthetic_image_shape": list(image.shape),
        "probability_map_shape": list(probability_map.shape),
        "entropy_min": float(entropy_map.min()),
        "entropy_max": float(entropy_map.max()),
        "entropy_mean": float(entropy_map.mean()),
        "wrote_png": wrote_png,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("SUN smoke demo complete")
    print(f"output_dir: {out_dir}")
    print(f"synthetic_image: {image.shape}")
    print(f"probability_map: {probability_map.shape}")
    print(f"entropy_range: {summary['entropy_min']:.4f} - {summary['entropy_max']:.4f}")
    if wrote_png:
        print("preview: demo_overview.png")


if __name__ == "__main__":
    main()
