"""Serialization for the tiny BEDLAM DenseKP audit dataset."""

import shutil
from pathlib import Path

import numpy as np


def write_bedlam_tiny_dataset(records: list[dict], output: Path) -> tuple[Path, Path]:
    image_dir = output / "images"
    annotation_dir = output / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    relative_names = []
    for output_index, record in enumerate(records):
        suffix = record["source_image"].suffix.lower() or ".png"
        output_name = f"sample_{output_index:03d}{suffix}"
        shutil.copy2(record["source_image"], image_dir / output_name)
        relative_names.append(output_name)

    output_npz = annotation_dir / "bedlam_densekp_tiny.npz"
    np.savez(
        output_npz,
        imgname=np.asarray(relative_names),
        center=np.stack([record["center"] for record in records]).astype(np.float32),
        scale=np.asarray([record["scale"] for record in records], dtype=np.float32),
        proj_verts=np.stack([record["proj_verts"] for record in records]).astype(np.float32),
    )
    debug_npz = annotation_dir / "bedlam_densekp_tiny_debug.npz"
    np.savez(
        debug_npz,
        source_index=np.asarray([record["source_index"] for record in records], dtype=np.int64),
        source_imgname=np.asarray([record["source_imgname"] for record in records]),
        joints2d=np.stack([record["joints2d"] for record in records]),
        valid_joint_count=np.asarray([record["valid_joint_count"] for record in records]),
        tight_bbox=np.stack([record["tight_bbox"] for record in records]),
        image_size=np.stack([record["image_size"] for record in records]),
        densekp_in_image_count=np.asarray([record["in_image_count"] for record in records]),
        densekp_in_bbox_count=np.asarray([record["in_bbox_count"] for record in records]),
        surface_operator_max_abs_error=np.asarray(
            [record["surface_error"] for record in records], dtype=np.float64
        ),
    )
    readme = output / "README.md"
    readme.write_text(
        "# Tiny BEDLAM DenseKP 数据\n\n"
        f"本目录由 `tools/annotation/generate_bedlam_densekp.py` 生成，共 {len(records)} 个 landscape 样本。\n\n"
        "- `images/`：复制后的 RGB 图像；\n"
        "- `annotations/bedlam_densekp_tiny.npz`：供 `DatasetKeypoints` 读取的主标注；\n"
        "- `annotations/bedlam_densekp_tiny_debug.npz`：审计字段，不参与训练。\n\n"
        "主标注中的 `proj_verts` 为 `[N,138,3]`；前两维是原图像素坐标，第三维是齐次坐标常数 1。\n",
        encoding="utf-8",
    )
    return output_npz, debug_npz

