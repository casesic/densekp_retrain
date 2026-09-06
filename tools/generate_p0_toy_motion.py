#!/usr/bin/env python3
"""生成确定性的 16 帧 SMPL-X toy motion 和唯一几何制品。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import smplx
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, default=Path("assets/densekp_regressor_R.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p0_perspective/geometry"))
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    kwargs = dict(gender="neutral", ext="npz", num_betas=10, use_pca=False,
                  flat_hand_mean=True, batch_size=args.frames)
    if (args.model_dir / "SMPLX_NEUTRAL.npz").is_file():
        model = smplx.SMPLX(str(args.model_dir), **kwargs)
    else:
        model = smplx.create(str(args.model_dir), model_type="smplx", **kwargs)

    frames = args.frames
    phase = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    ease = (0.5 - 0.5 * np.cos(np.pi * phase)).astype(np.float32)
    global_orient = np.zeros((frames, 3), dtype=np.float32)
    global_orient[:, 1] = np.deg2rad(-10.0 + 20.0 * ease)
    body_pose = np.zeros((frames, 21, 3), dtype=np.float32)
    # SMPL-X body joint order: left_shoulder=15, left_elbow=17（pelvis 不在 body_pose 内）。
    body_pose[:, 15, 2] = np.deg2rad(35.0 * ease)
    body_pose[:, 17, 1] = np.deg2rad(-55.0 * ease)
    body_pose = body_pose.reshape(frames, 63)
    betas = np.zeros((1, 10), dtype=np.float32)
    transl = np.zeros((frames, 3), dtype=np.float32)

    with torch.no_grad():
        result = model(
            global_orient=torch.from_numpy(global_orient),
            body_pose=torch.from_numpy(body_pose),
            betas=torch.from_numpy(np.repeat(betas, frames, axis=0)),
            transl=torch.from_numpy(transl),
            return_verts=True,
        )
    vertices = result.vertices.cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    regressor = np.load(args.regressor)
    if vertices.shape != (frames, 10475, 3):
        raise RuntimeError(f"vertices shape 错误：{vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"faces shape 错误：{faces.shape}")
    if regressor.shape != (138, 10475):
        raise RuntimeError(f"DenseKP regressor shape 错误：{regressor.shape}")
    if not np.isfinite(vertices).all():
        raise RuntimeError("vertices 存在非有限值")
    densekp_3d = np.einsum("kv,tvc->tkc", regressor, vertices, optimize=True).astype(np.float32)
    if not np.isfinite(densekp_3d).all():
        raise RuntimeError("densekp_3d 存在非有限值")
    if float(np.max(np.abs(vertices[-1] - vertices[0]))) <= 1e-4:
        raise RuntimeError("toy motion 没有产生可检测的顶点变化")

    np.save(args.output_dir / "vertices.npy", vertices)
    np.save(args.output_dir / "faces.npy", faces)
    np.save(args.output_dir / "densekp_3d.npy", densekp_3d)
    np.savez(
        args.output_dir / "motion_params.npz",
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl,
        fps=np.float32(30.0),
    )
    manifest = {
        "artifact": "p0_procedural_smplx_geometry",
        "avatar_id": "smplx_neutral_zero_betas",
        "motion_id": "procedural_p0_yaw_left_arm",
        "training_data": False,
        "deterministic": True,
        "frames": frames,
        "vertices_shape": list(vertices.shape),
        "faces_shape": list(faces.shape),
        "densekp_3d_shape": list(densekp_3d.shape),
        "dtype": "float32",
        "model_dir": str(args.model_dir.resolve()),
        "regressor": str(args.regressor.resolve()),
        "num_betas": 10,
        "fps": 30.0,
        "max_vertex_change_frame0_to_last_m": float(np.max(np.abs(vertices[-1] - vertices[0]))),
        "generation_seconds": time.perf_counter() - started,
        "files": {
            "vertices": "vertices.npy",
            "faces": "faces.npy",
            "densekp_3d": "densekp_3d.npy",
            "motion_params": "motion_params.npz",
        },
    }
    (args.output_dir / "geometry_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
