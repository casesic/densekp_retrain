"""Freeze box-only derivatives of accepted images. No rendering or model inference."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import cv2
import numpy as np
from PIL import Image, ImageDraw
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, supervision_for_crop
from densekp_retrain.data.source_split import validate_split
from densekp_retrain.data.loading import load_box_dataset
from densekp_retrain.io.json import write_json

# Historical import compatibility. Active callers use the public definitions.
load_dataset = load_box_dataset
write = write_json


from densekp_retrain.data.boxes import (
    change_box,
    box_rejection,
    trial_crop_from_box,
    support_rejection,
)
from densekp_retrain.data.packing import pack_derivative_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--parent-trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "loader").mkdir()
    (out / "inputs").mkdir()
    root = a.data_root.resolve()
    indices = {s: json.loads((root / f"{s}_index.json").read_text()) for s in ("train", "val")}
    split = json.loads((root / "split_v2.json").read_text())
    write_json(out / "parent_isolation.json", validate_split(split, indices))
    utc = datetime.now(timezone.utc).isoformat()
    config = dict(
        frozen_utc=utc,
        seed=930,
        parent_data=str(root),
        parent_trial=str(a.parent_trial.resolve()),
        target_order="Replay the previous seed-930 shuffled 400 exposures, without using its errors",
        intended_exposures=dict(
            reference_visible_instance=200, center=100, scale=50, single_edge=50
        ),
        reference="Original reference boxes already equal visible target-instance boxes; not a new independent box type",
        center="One axis, +/- uniform(0.10,0.20) of original 1.2-expanded crop side",
        scale="Alternate 1/1.2 deployment-tight crop and uniform(0.75,1.20), around original center",
        single_edge="One edge moved inward uniform(0.20,0.40) of its original dimension",
        rejection="Nonfinite or sides <2 ERP pixels; <10 supervised GT; <16 visible target pixels in full crop. Fall back to unchanged reference; record rejection; never prediction-based.",
        challenge="First two val rows per action/layout; reference, left/right 15%, up/down 12%, tight 1/1.2, loose 1.2, longest-axis high edge shortened 35%",
        range_evidence="Center/edge/random scale are synthetic stress tests, not measured YOLO errors; tight factor matches read-only current delivery crop code",
        severe_failure="Common-GT mean error >32 pixels at fixed original reference crop scale, or any nonfinite prediction. Common set required >=10 points; otherwise null.",
        robustness_success="C vs A and B: lower challenge common-GT ERP error and lower mean prediction drift, fewer severe failures; report all individually and original-val costs",
        scope="box-only; no RGB augmentation, rendering, identity or pose changes; historical boxes never train/val",
        units="ERP 1536x768; periodic x error; reference units ERP error *256/original crop side",
        supervision="in_view & inside_full_crop; visibility never cancels reliable XY supervision",
    )
    write_json(out / "config.json", config)
    # Freeze policy before constructing or scoring any new input.
    sequence = [
        json.loads(s)["row_index"]
        for s in (a.parent_trial / "completed_updates.jsonl").read_text().splitlines()
    ]
    assert len(sequence) == 400 and set(sequence) == set(range(198))
    rng = np.random.default_rng(930)
    kinds = np.array(
        ["reference_visible_instance"] * 200
        + ["center"] * 100
        + ["scale"] * 50
        + ["single_edge"] * 50
    )
    rng.shuffle(kinds)
    archives = {s: dict(np.load(root / "loader" / f"{s}.npz")) for s in indices}
    mask_cache = {}
    neighbor_cache = {}
    rejected = []
    checks = []
    max_error = 0.0

    def append_derivative(side, parent, name, box, records, arrays):
        nonlocal max_error
        parent_record = indices[side]["rows"][parent]
        parent_arrays = archives[side]
        key = (parent_record["scene"], parent_record["camera"], parent_record["person_id"])
        if key not in mask_cache:
            camera = Path(parent_record["scene"]) / parent_record["camera"]
            target = (
                np.array(Image.open(camera / f"person_{parent_record['person_id']}_mask.png")) > 0
            )
            valid = np.array(Image.open(camera / "valid_mask.png")) > 0
            target &= valid
            yy, xx = np.where(target)
            actual = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
            np.testing.assert_array_equal(actual, parent_record["given_bbox_xyxy"])
            # in_view also contains geometric-domain validity. All in_view points must lie in explicit mask.
            xy = parent_arrays["proj_verts"][parent, :, :2]
            # Frozen renderer uses floor, not delivery's nearest-pixel prediction diagnostic.
            ix = np.floor(xy[:, 0]).astype(int) % valid.shape[1]
            iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, valid.shape[0] - 1)
            assert np.all(~parent_arrays["in_view"][parent] | valid[iy, ix])
            mask_cache[key] = target
            neighbor = np.zeros_like(target)
            for path in camera.glob("person_*_mask.png"):
                if int(path.stem.split("_")[1]) != parent_record["person_id"]:
                    neighbor |= (np.array(Image.open(path)) > 0) & valid
            neighbor_cache[key] = neighbor
        box = np.asarray(box, np.float32)
        failure = box_rejection(box)
        if failure:
            return failure
        sample = {k: v[parent].copy() for k, v in parent_arrays.items() if v.ndim}
        sample["center"], sample["scale"] = trial_crop_from_box(box)
        # Use actual Dataset rather than a second image preprocessing implementation.
        probe_path = out / "loader" / "candidate_probe.npz"
        np.savez(
            probe_path,
            **{k: np.asarray([v]) for k, v in sample.items()},
            supervision_policy=SUPERVISION_POLICY,
        )
        dataset = load_box_dataset(probe_path)
        item = dataset[0]
        target_xy = item["proj_verts_cropped"][:, :2]
        supervision = supervision_for_crop(parent_arrays["in_view"][parent], target_xy)
        np.testing.assert_array_equal(supervision, item["proj_verts_cropped"][:, 2] > 0)
        roundtrip_error = float(
            np.abs(
                restore_crop_xy(target_xy, item["_trans"], (256, 256)) - sample["proj_verts"][:, :2]
            ).max()
        )
        if roundtrip_error > 2e-4:
            raise ValueError(f"Inverse affine error {roundtrip_error}")
        max_error = max(max_error, roundtrip_error)
        target_support = cv2.warpAffine(
            mask_cache[key].astype(np.uint8), item["_trans"], (256, 256), flags=cv2.INTER_NEAREST
        )
        neighbor = cv2.warpAffine(
            neighbor_cache[key].astype(np.uint8),
            item["_trans"],
            (256, 256),
            flags=cv2.INTER_NEAREST,
        )
        failure = support_rejection(supervision, target_support, min_points=10, min_pixels=16)
        if failure:
            return failure
        index = len(records)
        path = out / "inputs" / f"{side}_{index:03d}"
        rgb = cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(path) + "_square.png")
        Image.fromarray(rgb[:, 32:-32]).save(str(path) + "_center.png")
        np.savez(
            str(path) + "_geometry.npz",
            affine=item["_trans"],
            supervised=supervision,
            in_view=sample["in_view"],
            visible=sample["visible"],
            target_crop_xy=target_xy,
            target_erp_xy=sample["proj_verts"][:, :2],
            target_support_crop=target_support,
            network_input=item["img"][:, :, 32:-32],
        )
        sample["proj_verts"][:, 2] = supervision
        records.append(
            dict(
                parent_record,
                parent_row=parent,
                box_variant=name,
                parent_index=str(root / f"{side}_index.json"),
                given_bbox_xyxy=box.tolist(),
                parent_bbox_xyxy=parent_record["given_bbox_xyxy"],
                reference_crop_side=float(parent_arrays["scale"][parent] * 200),
                loader_input=str(out / "loader" / f"{side}.npz"),
                loader_row=index,
                affine_flat=item["_trans"].ravel().tolist(),
                supervised_points=int(supervision.sum()),
                supervised_occluded_points=int((supervision & ~sample["visible"]).sum()),
                in_view_outside_full_crop=int((sample["in_view"] & ~supervision).sum()),
                parent_neighbor_crop_pixels=parent_record["neighbor_crop_pixels"],
                neighbor_crop_pixels=int(neighbor.sum()),
                target_crop_pixels=int(target_support.sum()),
                square=str(path) + "_square.png",
                network_center=str(path) + "_center.png",
                geometry=str(path) + "_geometry.npz",
            )
        )
        arrays.append(sample)
        checks.append(
            dict(
                split=side,
                row=index,
                parent_row=parent,
                variant=name,
                roundtrip_max_px=roundtrip_error,
            )
        )
        return None

    output_indices = {}
    for side in ("train", "val"):
        rows = []
        arrays = []
        proposals = []
        if side == "train":
            for exposure, (parent, kind) in enumerate(zip(sequence, kinds)):
                box = indices[side]["rows"][parent]["given_bbox_xyxy"]
                params = {}
                if kind == "center":
                    params[rng.choice(["dx", "dy"])] = float(
                        rng.choice([-1, 1]) * rng.uniform(0.10, 0.20)
                    )
                elif kind == "scale":
                    params["factor"] = (
                        1 / 1.2 if exposure % 2 == 0 else float(rng.uniform(0.75, 1.20))
                    )
                elif kind == "single_edge":
                    params.update(
                        edge=int(rng.integers(4)), fraction=float(rng.uniform(0.20, 0.40))
                    )
                proposals.append((parent, str(kind), change_box(box, **params), params))
        else:
            grouped = defaultdict(list)
            for i, row in enumerate(indices[side]["rows"]):
                grouped[(row["action"], row["category"])].append(i)
            for parent in sorted(i for values in grouped.values() for i in values[:2]):
                box = indices[side]["rows"][parent]["given_bbox_xyxy"]
                edge = 2 + int(np.argmax(np.asarray(box[2:]) - box[:2]))
                variants = [
                    ("reference", {}),
                    ("left", dict(dx=-0.15)),
                    ("right", dict(dx=0.15)),
                    ("up", dict(dy=-0.12)),
                    ("down", dict(dy=0.12)),
                    ("deployment_tight", dict(factor=1 / 1.2)),
                    ("loose", dict(factor=1.2)),
                    ("single_edge", dict(edge=edge, fraction=0.35)),
                ]
                proposals.extend(
                    (parent, name, change_box(box, **params), params) for name, params in variants
                )
        for parent, name, box, params in proposals:
            failure = append_derivative(side, parent, name, box, rows, arrays)
            if failure:
                rejected.append(
                    dict(
                        split=side,
                        parent_row=parent,
                        variant=name,
                        box=box.tolist(),
                        reason=failure,
                    )
                )
                if side == "train":
                    failure = append_derivative(
                        side,
                        parent,
                        "reference_fallback",
                        indices[side]["rows"][parent]["given_bbox_xyxy"],
                        rows,
                        arrays,
                    )
                    if failure:
                        raise ValueError("Original accepted training sample failed support check")
            if not failure:
                rows[-1]["box_parameters"] = params
        np.savez(
            out / "loader" / f"{side}.npz",
            **pack_derivative_rows(arrays),
            supervision_policy=SUPERVISION_POLICY,
            split=side,
        )
        output_indices[side] = dict(
            split=side,
            rows=rows,
            frozen_utc=utc,
            parent_index=str(root / f"{side}_index.json"),
            scope="training_exposure_schedule" if side == "train" else "fixed_box_challenge",
            excluded_rows=[r for r in rejected if r["split"] == side],
        )
        write_json(out / f"{side}_index.json", output_indices[side])
    write_json(out / "source_isolation.json", validate_split(split, output_indices))
    # Preview by first example of each variant; no model-dependent choice.
    for side in ("train", "val"):
        selected = {}
        rows = output_indices[side]["rows"]
        for row in rows:
            selected.setdefault(row["box_variant"], row)
        sheet = Image.new("RGB", (512, 280 * len(selected)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (name, row) in enumerate(selected.items()):
            sheet.paste(Image.open(row["square"]), (0, 280 * i + 24))
            sheet.paste(Image.open(row["network_center"]), (288, 280 * i + 24))
            draw.text(
                (4, 280 * i + 4),
                f'{name}: parent {row["parent_row"]}, GT {row["supervised_points"]}',
                fill="black",
            )
        sheet.save(out / f"{side}_input_review.jpg")
    write_json(
        out / "acceptance.json",
        dict(
            status="passed",
            max_roundtrip_error_px=max_error,
            tolerance_px=2e-4,
            rows={s: len(x["rows"]) for s, x in output_indices.items()},
            actual_variants={
                s: dict(Counter(r["box_variant"] for r in x["rows"]))
                for s, x in output_indices.items()
            },
            rejected=rejected,
            checks=checks,
            seconds=time.monotonic() - start,
            masks="Explicit valid_mask + target instance, actual loader crop recomputes supervision",
            unchanged="Original images, geometry, indices, identities, poses, source split",
        ),
    )
    print(json.dumps({s: len(x["rows"]) for s, x in output_indices.items()}))


if __name__ == "__main__":
    main()
