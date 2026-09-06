#!/usr/bin/env python3
"""只读检查 CameraHMR DenseKP annotation NPZ 的结构与数值。"""

import argparse
from pathlib import Path

import numpy as np


REQUIRED = ("imgname", "center", "scale", "proj_verts")
RELATED = ("part", "gtkps")


def numeric_summary(name: str, value: np.ndarray) -> None:
    """输出数值数组的有限值比例和基本统计。"""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return
    finite = np.isfinite(array)
    finite_count = int(finite.sum())
    total = int(array.size)
    print(
        f"  {name}: finite={finite_count}/{total} "
        f"({finite_count / total:.9f}), nan={int(np.isnan(array).sum())}, "
        f"inf={int(np.isinf(array).sum())}"
    )
    if finite_count:
        valid = array[finite]
        print(
            f"  {name}: min={float(valid.min()):.9g}, "
            f"max={float(valid.max()):.9g}, mean={float(valid.mean()):.9g}, "
            f"std={float(valid.std()):.9g}"
        )


def print_examples(name: str, value: np.ndarray, indices: list[int]) -> None:
    print(f"  {name} examples:")
    for index in indices:
        print(f"    [{index}] {np.asarray(value[index]).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--samples", type=int, default=3, help="默认显示前几个样本")
    parser.add_argument("--index", type=int, help="额外详细显示指定样本")
    parser.add_argument(
        "--percentile-sample",
        type=int,
        default=1_000_000,
        help="第三通道 percentile 最多使用多少个值（确定性均匀采样）",
    )
    args = parser.parse_args()

    if not args.npz.is_file():
        raise SystemExit(f"file not found: {args.npz}")

    with np.load(args.npz, allow_pickle=True) as data:
        print(f"file: {args.npz.resolve()}")
        print("keys:", sorted(data.files))
        missing = [key for key in REQUIRED if key not in data]
        for key in data.files:
            value = data[key]
            marker = " (related)" if key in RELATED else ""
            print(f"{key}: shape={value.shape} dtype={value.dtype}{marker}")
        if missing:
            raise SystemExit(f"missing required fields: {missing}")

        n = len(data["imgname"])
        print(f"sample_count: {n}")
        if any(len(data[key]) != n for key in REQUIRED):
            counts = {key: len(data[key]) for key in REQUIRED}
            raise SystemExit(f"required fields have inconsistent sample counts: {counts}")
        if args.index is not None and not 0 <= args.index < n:
            raise SystemExit(f"--index must be in [0, {n - 1}]")

        indices = list(range(min(max(args.samples, 0), n)))
        if args.index is not None and args.index not in indices:
            indices.append(args.index)

        imgname = data["imgname"]
        print_examples("imgname", imgname, indices)
        path_kinds = [Path(str(item)).is_absolute() for item in imgname[: min(n, 1000)]]
        if path_kinds:
            if all(path_kinds):
                kind = "absolute"
            elif any(path_kinds):
                kind = "mixed"
            else:
                kind = "relative"
            print(f"  imgname path_kind (first {len(path_kinds)}): {kind}")

        center = np.asarray(data["center"])
        numeric_summary("center", center)
        print_examples("center", center, indices)

        scale = np.asarray(data["scale"])
        numeric_summary("scale", scale)
        if scale.ndim == 1:
            scale_kind = "scalar per sample"
        elif scale.ndim == 2 and scale.shape[1] == 2:
            scale_kind = "[sx, sy] per sample"
        else:
            scale_kind = "other"
        print(f"  scale representation: {scale_kind}")
        print_examples("scale", scale, indices)

        proj_verts = np.asarray(data["proj_verts"])
        strict_shape = proj_verts.ndim == 3 and proj_verts.shape == (n, 138, 3)
        print(f"  proj_verts strict [N,138,3]: {strict_shape}")
        if not strict_shape:
            raise SystemExit(f"expected proj_verts [N,138,3], got {proj_verts.shape}")
        numeric_summary("proj_verts.xy", proj_verts[..., :2])
        numeric_summary("proj_verts.channel_2", proj_verts[..., 2])

        channel_2 = proj_verts[..., 2].reshape(-1)
        finite_channel_2 = channel_2[np.isfinite(channel_2)]
        if finite_channel_2.size:
            limit = max(args.percentile_sample, 1)
            if finite_channel_2.size > limit:
                positions = np.linspace(0, finite_channel_2.size - 1, limit, dtype=np.int64)
                percentile_values = finite_channel_2[positions]
            else:
                percentile_values = finite_channel_2
            percentiles = np.percentile(
                percentile_values, [0, 1, 5, 25, 50, 75, 95, 99, 100]
            )
            print(
                "  channel_2 percentiles "
                "[0,1,5,25,50,75,95,99,100]:",
                [float(value) for value in percentiles],
            )
            unique, counts = np.unique(percentile_values, return_counts=True)
            if unique.size <= 32:
                print(
                    "  channel_2 sampled unique/counts:",
                    list(zip(unique.tolist(), counts.tolist())),
                )
            else:
                print(
                    f"  channel_2 sampled_unique_count: {unique.size} "
                    f"(sample_size={percentile_values.size})"
                )

        for sentinel in (-1.0, 0.0, 1.0):
            count = int(np.count_nonzero(proj_verts == sentinel))
            print(f"  proj_verts exact sentinel {sentinel:g}: {count}")

        if args.index is not None:
            print(f"sample[{args.index}] proj_verts:")
            print(proj_verts[args.index])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
