"""Exercise an installed checkout without weights, existing data, or path injection.

Run with an absolute script path from the repository or a temporary cwd. Each
output directory must be new. --mode stdlib works with no third-party packages;
light checks CPU utility imports; data additionally exercises the real Dataset.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["stdlib", "light", "data"], default="data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import densekp
    import densekp_retrain
    from densekp_retrain.runtime import repo_root, load_appearance_baseline

    root = repo_root()
    report = dict(
        mode=args.mode,
        packages={
            p.__name__: str(Path(p.__file__).resolve())
            for p in (densekp, densekp_retrain)
        },
        root=str(root),
        cwd=str(Path.cwd()),
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
    )
    assert root.joinpath("pyproject.toml").is_file()
    assert Path(densekp.__file__).resolve() == root / "densekp/__init__.py"
    assert (
        Path(densekp_retrain.__file__).resolve()
        == root / "src/densekp_retrain/__init__.py"
    )
    forbidden = ("torch", "bpy", "smplx", "densekp.core.densekp_trainer")
    assert not any(name in sys.modules for name in forbidden)
    report["stdlib_imports_no_model_or_blender"] = True
    if args.mode != "stdlib":
        for name in [
            "densekp.core.evaluation",
            "densekp.core.keypoint_contract",
            "densekp_retrain.camera",
            "densekp_retrain.body",
            "densekp_retrain.io",
            "densekp_retrain.data.boxes",
            "densekp_retrain.imaging.ds",
            "densekp_retrain.imaging.omni",
            "densekp_retrain.rendering.jobs",
        ]:
            importlib.import_module(name)
        assert not any(name in sys.modules for name in forbidden)
        report["light_imports_no_torch_model_or_blender"] = True
    if args.mode == "data":
        import cv2
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from densekp.core.datasets.configuration import audit_cfg
        from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
        from densekp.core.keypoint_supervision import SUPERVISION_POLICY

        with initialize_config_dir(
            version_base="1.2", config_dir=str(root / "configs/hydra")
        ):
            cfg = compose(config_name="train", overrides=["trainer=cpu"])
        # Unresolved Hydra runtime fields remain unresolved, exactly as before.
        (args.output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=False))
        assert cfg.DATASETS.DATASETS_AND_RATIOS == ""
        assert cfg.MODEL.IMAGE_SIZE == 256
        assert cfg.trainer.accelerator == "cpu"
        h, w = 96, 128
        y, x = np.indices((h, w))
        image = np.stack(
            ((x * 3 + y) % 256, (y * 5) % 256, (x + y * 7) % 256), axis=-1
        ).astype(np.uint8)
        assert cv2.imwrite(str(args.output / "input.png"), image)
        points = np.column_stack(
            (np.linspace(-8, 140, 138), np.linspace(4, 90, 138), np.ones(138))
        ).astype(np.float32)
        in_view = np.arange(138) % 5 != 0
        visible = in_view & (np.arange(138) % 3 == 0)
        np.savez(
            args.output / "labels.npz",
            imgname=["input.png"],
            center=np.array([[62.25, 45.5]], np.float32),
            scale=np.array([0.48], np.float32),
            proj_verts=points[None],
            in_view=in_view[None],
            visible=visible[None],
            supervision_policy=SUPERVISION_POLICY,
        )
        # Exercise cwd-relative annotation/root paths, plus absolute paths.
        relative = os.path.relpath(args.output.resolve(), Path.cwd())
        config = audit_cfg("supervision_weight")
        results = []
        for prefix in (relative, str(args.output.resolve())):
            ds = DatasetKeypoints(
                config,
                "installation-fixture",
                annotation_path=str(Path(prefix) / "labels.npz"),
                image_root=prefix,
                is_train=False,
                use_augmentation=False,
            )
            item = ds[0]
            results.append({k: v for k, v in item.items() if isinstance(v, np.ndarray)})
        assert results[0].keys() == results[1].keys()
        for key in results[0]:
            assert results[0][key].dtype == results[1][key].dtype
            np.testing.assert_array_equal(results[0][key], results[1][key])
        np.savez(args.output / "sample.npz", **results[0])
        report["sample_arrays"] = {
            k: dict(shape=list(v.shape), dtype=str(v.dtype))
            for k, v in results[0].items()
        }
        mask = results[0]["proj_verts_cropped"][:, 2] > 0
        report["supervised_points"] = int(mask.sum())
        report["supervised_occluded_points"] = int((mask & ~visible).sum())
        assert report["supervised_occluded_points"] > 0
        assert "densekp.core.densekp_trainer" not in sys.modules
        assert "bpy" not in sys.modules
        report["relative_absolute_arrays_exact"] = True
        report["cv2_version"] = cv2.__version__
        report["torch_version"] = torch.__version__
        (args.output / "appearance.jsonl").write_text(
            json.dumps({"body_uv_path": "relative_uv.npz"}) + "\n"
        )
        first, path = load_appearance_baseline(
            os.path.relpath(args.output.resolve() / "appearance.jsonl", Path.cwd())
        )
        assert first == {"body_uv_path": "relative_uv.npz"}
        assert path == args.output.resolve() / "appearance.jsonl"
        report["appearance_cwd_relative_preserved"] = True
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
