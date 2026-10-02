"""Independent old source/box results and explicit packing/supervision contracts."""

import base64
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from densekp.core.keypoint_supervision import (
    legacy_weight_targets,
    in_view_targets,
    supervision_for_crop,
)
from densekp_retrain.data.source_split import validate_split
from densekp_retrain.data.boxes import (
    change_box,
    official_bbox,
    box_rejection,
    trial_crop_from_box,
    support_rejection,
)
from densekp_retrain.data.packing import pack_index_rows, pack_derivative_rows
from tests.source_packing_fixture import source_boundaries, contract_cases, pure_boxes


class SourcePackingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(
            (Path(__file__).parent / "fixtures/source_packing_v1.json").read_text()
        )

    def assert_arrays(self, actual, name):
        with np.load(
            io.BytesIO(base64.b64decode(self.fixture[name + "_npz_base64"])), allow_pickle=False
        ) as expected:
            self.assertEqual(actual.files, expected.files)
            for key in expected.files:
                self.assertEqual(actual[key].dtype, expected[key].dtype, key)
                self.assertEqual(actual[key].shape, expected[key].shape, key)
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_declared_sources_all_scene_people_and_pair_failures_match_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                source_boundaries(Path(tmp), validate_split), self.fixture["source_boundaries"]
            )

    def test_boxes_rejections_and_bedlam_policy_match_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                pure_boxes(root, SimpleNamespace(change_box=change_box), official_bbox),
                self.fixture["boxes"],
            )
            with np.load(root / "boxes.npz") as actual:
                self.assert_arrays(actual, "boxes")

    def test_annotation_contract_masks_and_empty_cases_match_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(contract_cases(root), self.fixture["contract"])
            with np.load(root / "contract_arrays.npz") as actual:
                self.assert_arrays(actual, "contract_arrays")

    def test_packing_keeps_index_order_field_dtypes_and_legacy_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.npz"
            fields = dict(
                imgname=np.array(["a", "b"]),
                center=np.array([[1, 2], [3, 4]], np.float32),
                scale=np.array([0.1, 0.2], np.float32),
                proj_verts=np.ones((2, 138, 3), np.float32),
                in_view=np.ones((2, 138), bool),
                visible=np.zeros((2, 138), bool),
                person_id=np.array([5, 7], np.int32),
                camera_id=np.array(["cam2", "cam4"]),
            )
            indices = dict(
                rows=[
                    dict(loader_input=str(path), loader_row=i, image=str(root / f"{i}.png"))
                    for i in [1, 0, 1]
                ]
            )

            def save(data, policy="in_view_full_crop"):
                np.savez(path, **data, supervision_policy=policy)

            save(fields)
            packed = pack_index_rows(indices)
            self.assertEqual(list(packed), list(fields))
            for key in fields:
                if key != "imgname":
                    self.assertEqual(packed[key].dtype, fields[key].dtype)
                    np.testing.assert_array_equal(packed[key], fields[key][[1, 0, 1]])
            self.assertEqual(
                packed["imgname"].tolist(), [str(root / f"{i}.png") for i in [1, 0, 1]]
            )
            for name, key in [("missing_center", "center"), ("missing_visible", "visible")]:
                broken = dict(fields)
                broken.pop(key)
                save(broken)
                with self.assertRaises(KeyError) as error:
                    pack_index_rows(indices)
                self.assertEqual(
                    str(error.exception), self.fixture["packing_errors"][name]["error"]["message"]
                )
            save(fields, "wrong")
            with self.assertRaisesRegex(ValueError, "Supervision mismatch"):
                pack_index_rows(indices)
            # The packer itself does not reject empty rows; Dataset does on readback.
            self.assertTrue(all(a.size == 0 for a in pack_index_rows({"rows": []}).values()))
            with self.assertRaises(IndexError):
                pack_derivative_rows([])
            rows = [
                dict(value=np.array([2, 3], np.int32), label=np.array("second")),
                dict(value=np.array([0, 1], np.int32), label=np.array("first")),
            ]
            derivative = pack_derivative_rows(rows)
            self.assertEqual(list(derivative), ["value", "label"])
            np.testing.assert_array_equal(derivative["value"], [[2, 3], [0, 1]])
            self.assertEqual(derivative["value"].dtype, np.int32)

    def test_box_gate_thresholds_and_float32_center_scale(self):
        for box in ([0, 0, 1, 20], [0, 0, np.nan, 20], [20, 0, 10, 20]):
            self.assertEqual(box_rejection(np.array(box, np.float32)), "invalid_box")
        box = np.array([100, 100, 200, 300], np.float32)
        self.assertIsNone(box_rejection(box))
        center, scale = trial_crop_from_box(box)
        np.testing.assert_array_equal(center, [150, 200])
        self.assertEqual(center.dtype, np.float32)
        self.assertEqual(scale, np.float32(1.2))
        self.assertEqual(scale.dtype, np.float32)
        self.assertIsNone(support_rejection(np.ones(10, bool), np.ones(16, np.uint8), 10, 16))
        for n, p in [(9, 16), (10, 15), (0, 0)]:
            self.assertEqual(
                support_rejection(np.ones(n, bool), np.ones(p, np.uint8), 10, 16),
                "empty_or_insufficient_GT_or_visible_target_support",
            )

    def test_legacy_weight_and_in_view_adapters_keep_separate_semantics(self):
        xy = np.array([[0, 0], [0.5, 0], [-0.5, 0], [np.nan, 0]], np.float32)
        weights = np.array([0.25, 1, 0, 0], np.float32)
        legacy = legacy_weight_targets(xy, weights)
        np.testing.assert_array_equal(legacy[:, 2], weights)
        in_view = np.array([True, True, True, False])
        raw = in_view_targets(xy, in_view)
        np.testing.assert_array_equal(raw[:, 2], in_view.astype(np.float32))
        np.testing.assert_array_equal(supervision_for_crop(in_view, xy), [True, False, True, False])
        self.assertEqual(raw.dtype, np.float32)

    def test_old_imports_remain_compatible(self):
        from tools.validate.validate_source_isolated_split import validate_split as old_split
        from tools.experiments.prepare_box_robustness import change_box as old_box
        from tools.annotation.generate_bedlam_densekp import official_bbox as old_official

        self.assertIs(old_split, validate_split)
        self.assertIs(old_box, change_box)
        self.assertIs(old_official, official_bbox)
