# External/static assets

`densekp_regressor_R.npy` is intentionally included: it is only 5.6 MB and is
the audited 138-by-10475 SMPL-X surface operator. Large files are not included.

`downsample_mat_smplx.pkl` is the 17.5 MB, 437-by-10475 official BEDLAM
preprocessing operator downloaded from the BEDLAM GitHub repository. It is kept
only as an audited comparison/reference asset; DenseKP supervision must use
`densekp_regressor_R.npy`.

Expected external files:

- `vitpose_backbone.pth` (2.4 GB): required to construct a new training model.
- `densekp.ckpt` (7.5 GB): optional initialization/reference checkpoint.
- SMPL-X neutral model files: needed only for generating labels from SMPL-X
  parameters, not by the current NPZ-consuming training loop.
- `smplx2smpl.pkl` (552 MB): only needed if a conversion pipeline explicitly
  uses CameraHMR's SMPL-X-to-SMPL conversion; not needed by DenseKP training.

See `docs/MIGRATION.md` for audited source paths and sizes.
