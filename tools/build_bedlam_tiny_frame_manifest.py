#!/usr/bin/env python3
"""按 BEDLAM v1 官方状态机为已解压的 tiny PNG 建立 person-level 映射。"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re


IMAGE_RE = re.compile(r"^(seq_\d{6})_(\d{4})\.png$")
MANIFEST_FIELDS = (
    "scene",
    "archive_member",
    "seq_name",
    "image_name",
    "img_ind",
    "camera_csv",
    "camera_row",
    "start_frame",
    "person_id",
    "sequence_id",
    "motion_relpath",
    "smplx_param_ind",
    "body_X",
    "body_Y",
    "body_Z",
    "body_yaw",
)


def first_comment_value(comment: str, key: str) -> str:
    first = comment.split(";")[0]
    prefix = f"{key}="
    if not first.startswith(prefix):
        raise ValueError(f"expected first comment field {prefix!r}, got {comment!r}")
    return first.split("=")[-1]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def parse_state_machine(be_seq: Path):
    """逐行复现 df_full_body.py 中 sequence_name/start_frame 状态机。"""
    columns, rows = read_csv(be_seq)
    bodies_by_sequence = defaultdict(list)
    seq_name = ""
    for row_index, row in enumerate(rows):
        comment = row["Comment"]
        if "sequence_name" in comment:
            seq_name = first_comment_value(comment, "sequence_name")
            continue
        if "start_frame" not in comment:
            continue
        if not seq_name:
            raise ValueError(f"body row {row_index} appeared before sequence_name")
        start_frame = int(first_comment_value(comment, "start_frame"))
        body = row["Body"]
        person_id_parts = body.split("_")
        if len(person_id_parts) < 2:
            raise ValueError(f"invalid Body at row {row_index}: {body!r}")
        person_id = "_".join(person_id_parts[:-1])
        sequence_id = person_id_parts[-1]
        bodies_by_sequence[seq_name].append(
            {
                "start_frame": start_frame,
                "person_id": person_id,
                "sequence_id": sequence_id,
                "motion_relpath": f"{person_id}/{sequence_id}/motion_seq.npz",
                "body_X": row["X"],
                "body_Y": row["Y"],
                "body_Z": row["Z"],
                "body_yaw": row["Yaw"],
            }
        )
    return columns, rows, bodies_by_sequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--tiny-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene = args.scene_root.name
    be_seq = args.scene_root / "be_seq.csv"
    camera_root = args.scene_root / "ground_truth" / "camera"
    be_columns, be_rows, bodies_by_sequence = parse_state_machine(be_seq)

    images = sorted(args.tiny_image_root.glob("seq_*/*.png"))
    if len(images) != 24:
        raise SystemExit(f"expected 24 tiny PNGs, found {len(images)}")

    camera_cache = {}
    camera_columns = {}
    camera_stats = {}
    manifest = []
    motion_indices = defaultdict(set)
    image_records = []

    for image_path in images:
        match = IMAGE_RE.fullmatch(image_path.name)
        if match is None:
            raise SystemExit(f"invalid BEDLAM image name: {image_path.name}")
        seq_name, frame_text = match.groups()
        if image_path.parent.name != seq_name:
            raise SystemExit(f"directory/name sequence mismatch: {image_path}")
        img_ind = int(frame_text)
        camera_path = camera_root / f"{seq_name}_camera.csv"
        if seq_name not in camera_cache:
            columns, rows = read_csv(camera_path)
            camera_columns[seq_name] = columns
            camera_cache[seq_name] = rows
        camera_rows = camera_cache[seq_name]
        if img_ind >= len(camera_rows):
            raise SystemExit(
                f"camera index out of range: {seq_name} img_ind={img_ind}, rows={len(camera_rows)}"
            )
        camera_row = camera_rows[img_ind]
        if camera_row["name"] != image_path.name:
            raise SystemExit(
                f"camera name mismatch: image={image_path.name}, camera={camera_row['name']}"
            )
        if seq_name not in bodies_by_sequence:
            raise SystemExit(f"no bodies found for {seq_name}")

        archive_member = f"{scene}/png/{seq_name}/{image_path.name}"
        image_records.append((archive_member, seq_name, image_path.name, img_ind))
        for body in bodies_by_sequence[seq_name]:
            smplx_param_ind = img_ind + body["start_frame"]
            record = {
                "scene": scene,
                "archive_member": archive_member,
                "seq_name": seq_name,
                "image_name": image_path.name,
                "img_ind": img_ind,
                "camera_csv": str(camera_path.relative_to(args.scene_root)),
                "camera_row": img_ind,
                **body,
                "smplx_param_ind": smplx_param_ind,
            }
            manifest.append(record)
            motion_indices[body["motion_relpath"]].add(smplx_param_ind)

    required_camera_columns = {
        "name", "x", "y", "z", "yaw", "pitch", "roll", "focal_length"
    }
    for seq_name, columns in camera_columns.items():
        missing = required_camera_columns - set(columns)
        if missing:
            raise SystemExit(f"camera schema missing {sorted(missing)} in {seq_name}")

    for seq_name, rows in camera_cache.items():
        numeric = {key: [float(row[key]) for row in rows]
                   for key in ("x", "y", "z", "yaw", "pitch", "roll", "focal_length")}
        camera_stats[seq_name] = {
            "row_count": len(rows),
            **{f"{key}_min": min(values) for key, values in numeric.items()},
            **{f"{key}_max": max(values) for key, values in numeric.items()},
        }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)

    motion_path = args.output / "required_motion_files.txt"
    motion_path.write_text("".join(f"{path}\n" for path in sorted(motion_indices)), encoding="utf-8")
    indices_path = args.output / "required_motion_indices.txt"
    with indices_path.open("w", encoding="utf-8") as handle:
        for path in sorted(motion_indices):
            values = ", ".join(str(value) for value in sorted(motion_indices[path]))
            handle.write(f"{path}\nrequired indices: {values}\n\n")

    print("BE_SEQ_COLUMNS", be_columns)
    print("BE_SEQ_ROW_COUNT", len(be_rows))
    print("TINY_IMAGES", len(images))
    print("TINY_PERSON_SAMPLES", len(manifest))
    print("TINY_SEQUENCES", len(camera_cache))
    print("REQUIRED_CAMERA_FILES", len(camera_cache))
    print("REQUIRED_MOTION_FILE_COUNT", len(motion_indices))
    print("CAMERA_COLUMNS", next(iter(camera_columns.values())))
    for seq_name in sorted(camera_stats):
        print("CAMERA", seq_name, camera_stats[seq_name])
    for archive_member, seq_name, image_name, img_ind in image_records:
        print("IMAGE", archive_member, seq_name, image_name, img_ind)
    print("MANIFEST", manifest_path)
    print("MOTION_FILES", motion_path)
    print("MOTION_INDICES", indices_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
