# Notice and Attribution

This repository contains collaborative lab code and third-party code with source-level license/provenance headers.

## Important provenance notes

- `src/seg/models_deep_seg.py` carries a GNU GPLv3 notice from Jonathan Vacher for the multilayer segmentation mixture-model implementation.
- `src/seg/gmm_prior.py` carries provenance notes for reused `sklearn.mixture.gmm` code and modifications by Jonathan Vacher and Luis Carlos Garcia-Peraza Herrera.
- `src/seg/smm_prior.py` carries provenance notes for reused `sklearn.mixture.gmm` code, a Student-mixture implementation reference, and modifications by Jonathan Vacher and Luis Carlos Garcia-Peraza Herrera.
- `src/toolbox.py` carries a CC BY-NC-SA 4.0 notice from Jonathan Vacher for image-processing utilities.
- Other files contain lab-specific glue code for image segmentation, neural session loading, and analysis.

This public repository does not remove or override those notices. Reuse should preserve the source headers and comply with the most restrictive applicable upstream terms. No blanket relicensing is implied by this README or by the repository-specific documentation changes.

## Repository-specific contribution

The repository-specific changes focus on external readability and reproducibility hygiene:

- expanded README and module map
- lightweight no-private-data demo
- data/reproducibility notes
- ignored generated artifacts
- removal of tracked Python bytecode
- safer local path configuration
