# Datasets

No images or labels are staged. DenseKP consumes one NPZ per dataset plus an
image root. See `docs/DATA_FORMAT.md` and `configs/datasets.example.json`.

The copied upstream registry remains in `densekp/core/configs/__init__.py` for
semantic fidelity. Before running on a new machine, replace its dataset paths
with local paths or add a small manifest-backed registry adapter.
