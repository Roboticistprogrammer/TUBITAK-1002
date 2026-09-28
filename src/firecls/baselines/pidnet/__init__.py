"""Re-implementation of Pesonen et al. (WACV 2025): SAM box-prompted pseudo-masks -> PIDNet-S.

Modules:

* ``model``      -- vendored PIDNet-S builder and the export-ready image-level scoring wrapper.
* ``masks``      -- bit-encoded pseudo-mask files and the SAM box-prompt segmenter.
* ``data``       -- image + mask dataset with joint augmentation and Canny edge targets.
* ``losses``     -- PIDNet's four-term loss adapted to independent sigmoid channels.
* ``metrics``    -- pseudo-label IoU used for model selection diagnostics.
* ``checkpoint`` -- checkpoint and ``thresholds.json`` I/O shared by scripts and the family loader.
"""
