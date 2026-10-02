"""Independent CPU Dataset/crop baseline; no model, rendering or GPU computation."""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from tests.dataset_crop_fixture import make_inputs, specifications, run_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["capture", "verify"], required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.acceptance.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture = args.phase == "capture"
    from densekp.core.datasets.dataset_keypoints import DatasetKeypoints

    if capture:
        if "annotation_path" in inspect.signature(DatasetKeypoints).parameters:
            raise RuntimeError("Capture requires the independent pre-migration Dataset")
        make_inputs(root / "inputs")
        records = specifications(root / "inputs")
        for entry in json.loads((root / "selection.json").read_text()):
            record = dict(
                id=entry["id"],
                annotation_path=entry["loader_input"],
                image_root=entry["scene"],
                row=entry["loader_row"],
                name="refactor-readback",
                seed=930,
                augment=False,
                settings={},
            )
            records.append(record)
            records.append(
                dict(
                    record,
                    id=entry["id"] + "_augmented",
                    augment=True,
                    settings=dict(
                        SCALE_FACTOR=0.2,
                        TRANS_FACTOR=0.15,
                        TRANS_AUG_RATE=1.0,
                        ROT_FACTOR=25.0,
                        ROT_AUG_RATE=1.0,
                        COLOR_SCALE=0.2,
                    ),
                )
            )
        (root / "specifications.json").write_text(json.dumps(records, indent=2) + "\n")
        for name in ("tests/dataset_crop_fixture.py", "tools/validate/verify_dataset_crop.py"):
            destination = root / "before/source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    else:
        records = json.loads((root / "specifications.json").read_text())
    results = []
    for record in records:
        for route in (["old_id"] if capture else ["old_id", "explicit"]):
            print(record["id"], route, flush=True)
            metadata, arrays = run_dataset(record, explicit=route == "explicit", original=capture)
            label = record["id"] if capture else record["id"] + "_" + route
            (output / (label + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
            np.savez_compressed(output / (label + ".npz"), **arrays)
            if not capture:
                expected = json.loads((root / "before/data" / (record["id"] + ".json")).read_text())
                assert metadata == expected, label + " metadata changed"
                with np.load(
                    root / "before/data" / (record["id"] + ".npz"), allow_pickle=False
                ) as old:
                    assert list(arrays) == old.files, label + " array order changed"
                    for key, value in arrays.items():
                        assert value.dtype == old[key].dtype and value.shape == old[key].shape, (
                            label + "/" + key
                        )
                        np.testing.assert_array_equal(value, old[key], err_msg=label + "/" + key)
            results.append(
                dict(id=record["id"], route=route, arrays=len(arrays), error=metadata["error"])
            )
            if metadata["error"] and record["id"] != "flip_rejected":
                raise AssertionError(
                    "Unexpected baseline/verification failure: " + str(metadata["error"])
                )
    result = dict(
        status="passed",
        phase=args.phase,
        cases=results,
        arrays=sum(row["arrays"] for row in results),
        comparison="exact values, dtype, shape, field order, sampled parameters and Python/NumPy/Torch RNG",
        baseline_explicit_paths="old registry binding of identical annotation/root is the independent oracle",
        gpu=False,
        model_inference=False,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
