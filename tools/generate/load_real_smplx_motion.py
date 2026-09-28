#!/usr/bin/env python3
"""把真实 SMPL-X motion NPZ 适配为 renderer 的 canonical 参数。"""

import argparse, json
from pathlib import Path
import numpy as np


def pick_frames(total, count):
    ids = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    ids[np.argmin(np.abs(ids - total // 2))] = total // 2
    ids = np.unique(ids)
    if len(ids) != count:
        raise RuntimeError(f"帧选择去重后不足 {count}: {ids}")
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--test-frames", type=int, default=16)
    p.add_argument("--frame-ids", type=str)
    p.add_argument("--use-source-translation", action="store_true")
    a = p.parse_args()
    with np.load(a.input, allow_pickle=False) as z:
        keys = z.files
        raw = {k: z[k] for k in keys}
    if "poses" not in raw or raw["poses"].ndim != 2 or raw["poses"].shape[1] != 165:
        raise RuntimeError("当前 adapter 要求 poses=[T,165] 的完整 SMPL-X axis-angle")
    poses = np.asarray(raw["poses"], np.float32)
    total = len(poses)
    ids = (
        np.asarray([int(x) for x in a.frame_ids.split(",")], np.int64)
        if a.frame_ids
        else pick_frames(total, a.test_frames)
    )
    if (
        ids.ndim != 1
        or len(ids) == 0
        or len(np.unique(ids)) != len(ids)
        or ids.min() < 0
        or ids.max() >= total
    ):
        raise RuntimeError(f"frame ids 无效: {ids}")
    if "global_ori" in raw and not np.allclose(
        poses[:, :3], raw["global_ori"], atol=1e-7
    ):
        raise RuntimeError("poses[:,:3] 与 global_ori 冲突")
    betas = np.asarray(raw.get("betas"), np.float32)
    if betas.ndim == 2:
        if not np.allclose(betas, betas[0], atol=1e-7):
            raise RuntimeError("逐帧 betas 非静态，当前 P0.5 不接受")
        betas = betas[0]
    if betas.ndim != 1:
        raise RuntimeError(f"betas shape 无法解释: {betas.shape}")
    source_trans = np.asarray(
        raw.get("trans", raw.get("transl", np.zeros((total, 3)))), np.float32
    )
    if source_trans.shape != (total, 3):
        raise RuntimeError(f"translation shape 错误: {source_trans.shape}")
    if (
        not np.isfinite(poses).all()
        or not np.isfinite(betas).all()
        or not np.isfinite(source_trans).all()
    ):
        raise RuntimeError("motion 参数存在非有限值")
    gender = str(np.asarray(raw.get("gender", "neutral")).item()).lower()
    fps = float(np.asarray(raw.get("mocap_frame_rate", 30)).item())
    selected = poses[ids]
    canonical = {
        "global_orient": selected[:, :3],
        "body_pose": selected[:, 3:66],
        "jaw_pose": selected[:, 66:69],
        "leye_pose": selected[:, 69:72],
        "reye_pose": selected[:, 72:75],
        "left_hand_pose": selected[:, 75:120],
        "right_hand_pose": selected[:, 120:165],
        "betas": betas,
        "transl": (
            source_trans[ids]
            if a.use_source_translation
            else np.zeros((len(ids), 3), np.float32)
        ),
        "source_transl": source_trans[ids],
        "source_frame_ids": ids,
        "source_total_frames": np.int64(total),
        "fps": np.float32(fps),
        "gender": np.asarray(gender),
        "source_translation_used": np.asarray(a.use_source_translation),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **canonical)
    report = {
        "source": str(a.input.resolve()),
        "source_keys": keys,
        "schema": "SMPL-X full axis-angle poses[165]",
        "total_frames": total,
        "test_frames": len(ids),
        "source_frame_ids": ids.tolist(),
        "gender": gender,
        "fps": fps,
        "betas_shape": list(betas.shape),
        "translation_key": (
            "trans" if "trans" in raw else ("transl" if "transl" in raw else None)
        ),
        "source_translation_used": a.use_source_translation,
        "source_translation_range_m": {
            "min": source_trans.min(0).tolist(),
            "max": source_trans.max(0).tolist(),
        },
        "canonical_shapes": {
            k: list(v.shape)
            for k, v in canonical.items()
            if isinstance(v, np.ndarray) and v.ndim > 0
        },
        "output": str(a.output.resolve()),
    }
    a.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    a.output.with_name("frame_mapping.csv").write_text(
        "render_frame_id,source_frame_id\n"
        + "".join(f"{i},{source}\n" for i, source in enumerate(ids)),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

r'''
python tools/generate/load_real_smplx_motion.py \
    /home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz \
    --output outputs/motions/aaron_1095_canonical.npz --test-frames 16
'''
