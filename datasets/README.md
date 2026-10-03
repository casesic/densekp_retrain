# Datasets

Current scope and acceptance rules: [当前数据与来源划分](../docs/DATA_AND_MODELS.md) and [contracts](../docs/CONTRACTS.md).

This directory holds named, versioned training/validation datasets and existing
tiny regression fixtures. DenseKP consumes an annotation NPZ plus an image root;
neither must physically live here for the loader to work.

Keep candidate renders, reconstruction assets, visual reviews and experiment
results in `outputs/experiments/` while their quality and selection are unsettled.
The cycling pilot remains there, including its diagnostic loader NPZ. Moving it
here does not make it an accepted corpus or an independent train/validation split.

After acceptance, package the selected samples under `datasets/<name_version>/`,
freeze the source-person/image split and supervision policy, and record provenance.
Resolve all image, geometry and mask dependencies; rewrite active paths and check
the actual loader after packaging. Do not blindly move a directory with absolute
paths or symlinks. Preserve experiment evidence in its original location.

Dataset registration is in `densekp/core/configs/__init__.py`; use the corresponding
Hydra data configuration for a named training run. Local assets are not implied to
be available in a clean checkout. See `docs/SAMPLE_WALKTHROUGH.md` for the current
138-point and supervision contracts.
