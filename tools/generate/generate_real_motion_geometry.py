#!/usr/bin/env python3
"""从 canonical real motion 生成 P0 单一几何制品。"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.body import (
    compute_densekp_3d,
    load_smplx_model,
    validate_geometry,
    write_geometry_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--regressor",
        type=Path,
        default=Path("assets/densekp_regressor_R.npy"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--avatar-id")
    parser.add_argument("--motion-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with np.load(args.motion, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}

    source_frame_ids = motion["source_frame_ids"]
    frames = len(source_frame_ids)
    gender = str(motion["gender"].item())
    model = load_smplx_model(
        args.model_dir,
        gender=gender,
        num_betas=len(motion["betas"]),
        batch_size=frames,
    )

    parameter_names = [
        "global_orient",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "left_hand_pose",
        "right_hand_pose",
        "transl",
    ]
    inputs = {
        name: torch.from_numpy(np.asarray(motion[name], np.float32))
        for name in parameter_names
    }
    inputs["betas"] = torch.from_numpy(
        np.repeat(motion["betas"][None], frames, axis=0).astype(np.float32)
    )
    with torch.no_grad():
        vertices = (
            model(return_verts=True, **inputs)
            .vertices.cpu()
            .numpy()
            .astype(np.float32)
        )

    faces = np.asarray(model.faces, np.int32)
    regressor = np.load(args.regressor)
    validate_geometry(vertices, regressor, frames=frames)
    densekp_3d = compute_densekp_3d(regressor, vertices)
    write_geometry_arrays(
        args.output_dir,
        vertices=vertices,
        faces=faces,
        densekp_3d=densekp_3d,
    )

    # canonical 参数原样进入 geometry 目录，保留 source frame mapping 与未使用 translation。
    np.savez(args.output_dir / "motion_params.npz", **motion)
    manifest = {
        "artifact": "p0_real_smplx_geometry",
        "training_data": False,
        "source_motion": str(args.motion.resolve()),
        "avatar_id": args.avatar_id or f"smplx_{gender}_source_betas",
        "motion_id": args.motion_id or args.motion.stem,
        "gender": gender,
        "frames": frames,
        "source_total_frames": int(motion["source_total_frames"]),
        "source_frame_ids": source_frame_ids.tolist(),
        "source_translation_used": bool(motion["source_translation_used"].item()),
        "vertices_shape": list(vertices.shape),
        "faces_shape": list(faces.shape),
        "densekp_3d_shape": list(densekp_3d.shape),
        "dtype": "float32",
        "model_dir": str(args.model_dir.resolve()),
        "regressor": str(args.regressor.resolve()),
        "generation_seconds": time.perf_counter() - started,
        "max_vertex_change_test_first_to_last_m": float(
            np.max(np.abs(vertices[-1] - vertices[0]))
        ),
    }
    (args.output_dir / "geometry_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

r'''
# 方式 A: 从 BEDLAM canonical motion 直接生成 P0 几何制品
python tools/generate/generate_real_motion_geometry.py \
    --motion outputs/baselines/p0_real_motion/geometry/canonical_motion.npz \
    --model-dir /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx \
    --output-dir outputs/geometry/real_motion_test

# 方式 B: 从 BEDLAM 原始 motion 从头跑完整两步
python tools/generate/load_real_smplx_motion.py \
    /home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz \
    --output outputs/motions/aaron_1095_canonical.npz --test-frames 16
python tools/generate/generate_real_motion_geometry.py \
    --motion outputs/motions/aaron_1095_canonical.npz \
    --model-dir /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx \
    --output-dir outputs/geometry/aaron_1095
'''
