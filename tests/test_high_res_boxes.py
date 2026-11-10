"""CPU-only sized boxes, actual Dataset crops and historical command construction."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from densekp.core.keypoint_supervision import SUPERVISION_POLICY
from densekp_retrain.data.boxes import (
    BoxRejected, change_box_for_image, crop_from_box, rejection_record,
    validate_box_context,
)
from densekp_retrain.data.loading import load_box_dataset
from tools.experiments.prepare_box_robustness import prepare_sized_derivatives

ROOT = Path(__file__).resolve().parents[1]


def sized(box, **kwargs):
    options = dict(image_size_wh=(5760, 2880), boundary_policy="erp_reject_seam",
                   min_side_px=2, shift_reference_expansion=1.2)
    options.update(kwargs)
    return change_box_for_image(box, **options)


class HistoricalCommands(unittest.TestCase):
    def test_only_final_size_flags_added_and_commands_evaluate(self):
        before = json.loads((ROOT / "tests/fixtures/historical_box_commands_v1.json").read_text())
        for name, expected in before.items():
            with self.subTest(entry=name):
                tree = ast.parse((ROOT / "tools/experiments" / name).read_text())
                expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "command" for t in n.targets))
                self.assertEqual([n.value for n in expr.elts[-4:]],
                                 ["--final-width", "1536", "--final-height", "768"])
                job = dict(characters=["F01", "M01"], pose_source_ids=["p1", "p2"],
                           slots=[0, 2], scene="scene", control_distance_m=3.8)
                context = dict(sys=SimpleNamespace(executable="python"), ROOT=Path("/repo"),
                               root=Path("/data"), out=Path("/out"), output=Path("/out"),
                               base=Path("/base"), config=Path("/config"), job=job,
                               repair=dict(job, reuse_layout="/base"),
                               keep=[dict(character_id="F01", pose_source_id="p1")])
                command = eval(compile(ast.Expression(expr), name, "eval"), context)
                self.assertEqual(command[command.index("--width") + 1], "1536")
                self.assertEqual(command[command.index("--output-mode") + 1], "full")
                expr.elts = expr.elts[:-4]
                self.assertEqual(ast.dump(expr), expected)

    def test_new_defaults_still_c_compact(self):
        for name in ("render_cycling_ds.py", "render_cycling_characters_ds.py",
                     "render_adult_candidates_ds.py"):
            tree = ast.parse((ROOT / "tools/experiments" / name).read_text())
            defaults = {n.args[0].value: next(k.value.value for k in n.keywords if k.arg == "default")
                        for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                        and n.args and isinstance(n.args[0], ast.Constant)
                        and any(k.arg == "default" for k in n.keywords)
                        and n.args[0].value in ("--width", "--final-width", "--final-height", "--output-mode")}
            self.assertEqual(defaults, {"--width": 5760, "--final-width": 5760,
                                       "--final-height": 2880, "--output-mode": "compact"})


class SizedBoxTests(unittest.TestCase):
    def test_explicit_size_required_and_high_coordinates_retained(self):
        with self.assertRaises(TypeError):
            change_box_for_image([0, 0, 10, 10])
        box = [4800, 2300, 5700, 2850]
        np.testing.assert_array_equal(sized(box), box)
        np.testing.assert_allclose(sized(box, dy=.1)[[1, 3]], [2408, 2880])

    def test_invalid_and_pixel_gate(self):
        for box, reason in [([0, 0, np.nan, 10], "nonfinite"),
                            ([0, 0, np.inf, 10], "nonfinite"),
                            ([10, 0, 9, 10], "reversed"), ([0, 0, 0, 10], "empty"),
                            ([0, 0, 1.99, 10], "below_min")]:
            with self.subTest(box=box), self.assertRaisesRegex(BoxRejected, reason):
                sized(box)
        np.testing.assert_array_equal(sized([0, 0, 2, 10]), [0, 0, 2, 10])
        for params in (dict(factor=np.nan), dict(dx=np.inf), dict(edge=5), dict(edge=1.0),
                       dict(edge=True), dict(fraction=1.1),
                       dict(factor=0), dict(image_size_wh=(0, 20)), dict(image_size_wh=(20.1, 20))):
            with self.assertRaises(BoxRejected):
                sized([1, 1, 10, 10], **params)

    def test_seams_distinct_from_rectangular_clipping(self):
        for box, params in [([5750, 50, 5770, 100], {}), ([-5, 50, 10, 100], {}),
                            ([5700, 50, 5750, 100], dict(dx=.2)),
                            ([0, 50, 50, 100], dict(dx=-.2)),
                            ([5700, 50, 5750, 100], dict(factor=2))]:
            with self.assertRaisesRegex(BoxRejected, "erp_seam_crossing"):
                sized(box, **params)
        np.testing.assert_array_equal(sized([-5, -10, 20, 30], boundary_policy="raster_clip"),
                                      [0, 0, 20, 30])
        np.testing.assert_array_equal(sized([100, -10, 120, 30]), [100, 0, 120, 30])
        with self.assertRaisesRegex(BoxRejected, "reversed_or_empty"):
            sized([0, 2900, 20, 2950])
        with self.assertRaisesRegex(BoxRejected, "erp_crop_seam_crossing"):
            crop_from_box([1, 0, 20, 100], image_size_wh=(5760, 2880),
                          boundary_policy="erp_reject_seam", crop_policy="deployment", min_side_px=2)

    def context(self, box, **overrides):
        target = np.zeros((60, 80), bool)
        target[10:50, 20:50] = True
        args = dict(bbox_kind="visible_instance_tight", bbox_source="fixture mask", person_id=7,
                    mask_person_id=7, annotation_person_id=7, image_size_wh=(80, 60),
                    manifest_size_wh=(80, 60), valid_mask=np.ones_like(target), person_mask=target,
                    boundary_policy="erp_reject_seam", min_side_px=2)
        args.update(overrides)
        return validate_box_context(box, **args)

    def test_kinds_keep_identity_provenance_and_support_checks(self):
        self.context([20, 10, 50, 50])
        with self.assertRaisesRegex(BoxRejected, "tight_mismatch"):
            self.context([15, 5, 55, 55])
        for kind in ("fixed_reference", "detection_tracking", "manual", "synthetic_perturbation"):
            self.context([15, 5, 55, 55], bbox_kind=kind)
            for extra, reason in [(dict(mask_person_id=8), "identity"),
                                  (dict(annotation_person_id=8), "identity"),
                                  (dict(bbox_source=""), "source"),
                                  (dict(manifest_size_wh=(1536, 768)), "size_mismatch"),
                                  (dict(valid_mask=np.ones((10, 10))), "size_mismatch"),
                                  (dict(person_mask=np.zeros((60, 80))), "empty_visible")]:
                with self.subTest(kind=kind, extra=list(extra)), self.assertRaisesRegex(BoxRejected, reason):
                    self.context([15, 5, 55, 55], bbox_kind=kind, **extra)
            with self.assertRaisesRegex(BoxRejected, "no_valid_target_support"):
                self.context([0, 0, 10, 10], bbox_kind=kind)
        with self.assertRaisesRegex(BoxRejected, "unknown_bbox_kind"):
            self.context([20, 10, 50, 50], bbox_kind="unknown")
        target = np.ones((60, 80), bool)
        with self.assertRaisesRegex(BoxRejected, "both_seam_edges"):
            self.context([0, 0, 80, 60], person_mask=target)

    def test_safe_rejections(self):
        record = rejection_record("nonfinite", box=np.array([np.nan, np.inf, -np.inf, 2]),
                                  parameters={"dx": np.float32(np.nan)})
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False))["box"], [None, None, None, 2])

    def test_shift_normalization_independent_of_crop_expansion(self):
        changed = sized([100, 100, 200, 300], dx=.1, shift_reference_expansion=1.2)
        np.testing.assert_array_equal(changed, [124, 100, 224, 300])
        for policy, expansion in (("deployment", 1.0), ("historical_trial", 1.2)):
            center, scale = crop_from_box(changed, image_size_wh=(5760, 2880),
                                         boundary_policy="erp_reject_seam", crop_policy=policy, min_side_px=2)
            np.testing.assert_array_equal(center, [174, 200])
            self.assertEqual(scale, np.float32(expansion))


class RealConsumerTests(unittest.TestCase):
    def test_actual_dataset_affine_packing_policies_and_rejected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Black RGB is intentional: only explicit masks determine support.
            image = np.zeros((60, 80, 3), np.uint8)
            image[20:30, 30:40] = [10, 20, 30]
            Image.fromarray(image).save(root / "image.png")
            target = np.zeros((60, 80), np.uint8)
            target[10:50, 20:50] = 255
            Image.fromarray(target).save(root / "person.png")
            Image.fromarray(np.full((60, 80), 255, np.uint8)).save(root / "valid.png")
            (root / "manifest.json").write_text(json.dumps(dict(final_erp_resolution=[80, 60])))
            points = np.tile([35, 30, 1], (1, 138, 1)).astype(np.float32)
            np.savez(root / "original.npz", imgname=[str(root / "image.png")], center=[[35, 30]], scale=[.2],
                     proj_verts=points, in_view=np.ones((1, 138), bool), visible=np.zeros((1, 138), bool),
                     person_id=[7], camera_id=["cam2"], supervision_policy=SUPERVISION_POLICY)
            parent = dict(image=str(root / "image.png"), manifest=str(root / "manifest.json"),
                          valid_mask=str(root / "valid.png"), person_mask=str(root / "person.png"),
                          person_id=7, mask_person_id=7, camera="cam2", bbox_kind="visible_instance_tight",
                          bbox_source="fixture", given_bbox_xyxy=[20, 10, 50, 50],
                          loader_input=str(root / "original.npz"), loader_row=0)
            variants = [dict(name=p, parameters={}, crop_policy=p) for p in ("deployment", "historical_trial")]
            variants.append(dict(name="nan", parameters={"dx": float("nan")}, crop_policy="deployment"))
            variants.append(dict(name="seam", parameters={"dx": 2}, crop_policy="deployment"))
            result = prepare_sized_derivatives([parent], variants, root / "new", min_side_px=2,
                     min_points=10, min_target_crop_pixels=16, shift_reference_expansion=1.2,
                     boundary_policy="erp_reject_seam")
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(len(result["excluded_rows"]), 2)
            reasons = json.loads((root / "new/rejections.json").read_text())["reasons"]
            self.assertEqual(reasons, {"invalid_perturbation_parameters": 1, "erp_seam_crossing": 1})
            data = load_box_dataset(root / "new/packed.npz")
            for i, expansion in enumerate((1.0, 1.2)):
                item = data[i]
                expected = np.array([[256 / (40 * expansion), 0, 128 - 35 * 256 / (40 * expansion)],
                                     [0, 256 / (40 * expansion), 128 - 30 * 256 / (40 * expansion)]])
                np.testing.assert_allclose(item["_trans"], expected, atol=1e-5)
                warped = cv2.warpAffine(image[:, :, ::-1], item["_trans"], (256, 256))
                np.testing.assert_array_equal(item["img_disp"], warped)
                np.testing.assert_array_equal(item["proj_verts_cropped"][:, :2], np.zeros((138, 2)))
                self.assertEqual(result["rows"][i]["supervised_occluded_points"], 138)
            # The finite-invalid parent path also writes diagnostics without a loader.
            bad = dict(parent, given_bbox_xyxy=[20, 10, float("nan"), 50])
            result = prepare_sized_derivatives([bad], variants[:1], root / "bad", min_side_px=2,
                     min_points=10, min_target_crop_pixels=16, shift_reference_expansion=1.2,
                     boundary_policy="erp_reject_seam")
            self.assertEqual(result["rows"], [])
            self.assertIsNone(json.loads((root / "bad/rejections.json").read_text())["rows"][0]["box"][2])
