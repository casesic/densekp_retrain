"""Artificial sequence fixtures only; no real motion or parameter updates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction

import numpy as np
from PIL import Image
from densekp_retrain.data.sequence_index import (
    SCHEMA, SequenceIndexError, exact_time, row_context, validate_sequence_index,
)
from densekp_retrain.data.boxes import trial_crop_from_box
from densekp_retrain.data.loading import load_box_dataset
from densekp.core.keypoint_supervision import SUPERVISION_POLICY


def sequence_fixture():
    """Three artificial clock ticks; images/poses are not a motion quality fixture."""
    ep = dict(schema=SCHEMA, episode_id="test_episode", sequence_id="test_sequence",
              data_kind="synthetic_test_only", synthetic_test_only=True, temporal_kind="sequence",
              time_base=[1, 30], time_semantics="synthetic_clock", time_origin="fixture zero",
              timeline=dict(first_frame_index=0, frame_step=1, frame_count=3, expected_tick_step=1),
              synchronization=dict(status="synthetic_same_state", basis="fixture shared state",
                                   evidence_ref="evidence.json"),
              source_refs=["evidence.json"], seed=12,
              cameras=[dict(camera_id=c, rig_ref="evidence.json", calibration_ref="evidence.json",
                            image_size_wh=[80, 60]) for c in ("cam2", "cam4")],
              assets=[dict(asset_id="test_asset", asset_ref="evidence.json", motion_source_ref="evidence.json")],
              people=[dict(person_id=7, source_ref="evidence.json", character_family_id="test_family",
                           asset_id="test_asset", identity_status="synthetic_instance")])
    table = dict(schema=SCHEMA, episode_id=ep["episode_id"], sequence_id=ep["sequence_id"],
                 frames=[], observations=[], boxes=[], packed=dict(path="packed.npz", row_count=6))
    for frame in range(3):
        table["frames"].append(dict(frame_index=frame, time_tick=frame, status="available"))
        for camera in ("cam2", "cam4"):
            oid = f"frame{frame}_{camera}_person7"
            table["observations"].append(dict(
                observation_id=oid, frame_index=frame, time_tick=frame, person_id=7, camera_id=camera,
                source_ref="evidence.json", character_family_id="test_family", asset_id="test_asset",
                asset_frame_index=frame, asset_frame_status="known", status="observed",
                image="image.png", annotations="annotations.npz", annotation_person_index=0,
                valid_mask="valid.png", person_mask="person.png", raster_manifest="raster.json",
                image_size_wh=[80, 60]))
            i = len(table["boxes"])
            table["boxes"].append(dict(box_id=f"box{i}", observation_id=oid, packed_row_id=i,
                bbox_xyxy=[20, 10, 50, 50], bbox_kind="visible_instance_tight", bbox_source="test mask",
                crop_policy="historical_trial", parent_box_id=None, origin_ref="origins.json", origin_row=i))
    return ep, table


def write_fixture_files(root, ep, table):
    """Tiny RGB and explicit masks, marked by synthetic_test_only in episode."""
    image = np.zeros((60, 80, 3), np.uint8)
    image[10:50, 20:50] = [80, 90, 100]
    Image.fromarray(image).save(root / "image.png")
    Image.fromarray(np.full((60, 80), 255, np.uint8)).save(root / "valid.png")
    Image.fromarray((image[:, :, 0] > 0).astype(np.uint8)*255).save(root / "person.png")
    (root / "evidence.json").write_text(json.dumps(dict(synthetic_test_only=True)))
    (root / "raster.json").write_text(json.dumps(dict(final_erp_resolution=[80, 60])))
    origins = [dict(person_id=7, given_bbox_xyxy=b["bbox_xyxy"], bbox_kind=b["bbox_kind"])
               for b in table["boxes"]]
    (root / "origins.json").write_text(json.dumps(dict(rows=origins)))
    xy = np.tile([35, 30], (1, 138, 1)).astype(np.float32)
    np.savez(root / "annotations.npz", person_ids=[7], xy=xy)
    count = table["packed"]["row_count"]
    center, scale = trial_crop_from_box(np.array([20, 10, 50, 50], np.float32))
    np.savez(root / "packed.npz", imgname=[str(root / "image.png")]*count,
             person_id=[7]*count, camera_id=[o["camera_id"] for o in table["observations"]],
             center=np.tile(center, (count, 1)), scale=np.full(count, scale, dtype=np.float32),
             proj_verts=np.tile([35., 30., 1.], (count, 138, 1)).astype(np.float32),
             in_view=np.ones((count, 138), bool), visible=np.zeros((count, 138), bool),
             supervision_policy=SUPERVISION_POLICY)


def mark_missing(table, observation, state):
    observation["status"] = state
    observation["reason"] = "synthetic_test_only missing observation"
    if state == "camera_missing":
        for k in ("image", "annotations", "valid_mask", "person_mask", "raster_manifest",
                  "annotation_person_index", "image_size_wh"):
            observation[k] = None
    elif state == "person_absent":
        observation["person_mask"] = observation["annotation_person_index"] = None
    table["boxes"] = [b for b in table["boxes"] if b["observation_id"] != observation["observation_id"]]
    for i, b in enumerate(table["boxes"]):
        b["packed_row_id"] = i
    table["packed"]["row_count"] = len(table["boxes"])


class SequenceIndexTests(unittest.TestCase):
    def validate(self, ep, table, **kw):
        return validate_sequence_index(ep, table, base_dir="/fixture", check_files=False, **kw)

    def rejected(self, ep, table, code):
        with self.assertRaises(SequenceIndexError) as ctx:
            self.validate(ep, table)
        self.assertEqual(ctx.exception.code, code)

    def test_normal_dual_camera_integer_time_and_context(self):
        ep, table = sequence_fixture()
        result = self.validate(ep, table)
        self.assertEqual(result["packed_rows"], 6)
        self.assertFalse(result["exposure_sync_validated"])
        context = row_context(ep, table, 3)
        self.assertEqual(context["observation"]["asset_frame_index"], 1)
        self.assertEqual(context["observation"]["camera_id"], "cam4")
        self.assertEqual(exact_time(2**54+1, [1001, 30000]), Fraction((2**54+1)*1001, 30000))
        for tick in (1.0, True):
            with self.assertRaises(SequenceIndexError):
                exact_time(tick, [1, 30])
        with self.assertRaises(SequenceIndexError):
            exact_time(0, [2, 60])
        for invalid_row in (-1, None, 6):
            with self.assertRaises(SequenceIndexError):
                row_context(ep, table, invalid_row)

    def test_frame_order_time_conflicts_and_gaps(self):
        cases = [
            (lambda e,t: t["frames"].pop(1), "missing_frame_record"),
            (lambda e,t: t["frames"].reverse(), "frame_order_or_gap"),
            (lambda e,t: t["frames"][1].update(frame_index=0), "duplicate_frame_index"),
            (lambda e,t: t["frames"][1].update(time_tick=0), "nonmonotonic_time"),
            (lambda e,t: t["frames"][1].update(time_tick=9), "time_step_conflict"),
            (lambda e,t: t["observations"][1].update(time_tick=1), "observation_time_conflict"),
            (lambda e,t: e.update(temporal_kind="static"), "static_must_have_one_frame"),
        ]
        for mutate, code in cases:
            with self.subTest(code=code):
                ep, table = sequence_fixture(); mutate(ep, table); self.rejected(ep, table, code)
        ep, table = sequence_fixture()
        table["frames"][1].update(status="missing", reason="explicit dropped frame")
        for obs in table["observations"][2:4]:
            mark_missing(table, obs, "camera_missing")
        result = self.validate(ep, table)
        self.assertEqual(result["missing_frame_count"], 1)

    def test_missing_camera_absence_empty_and_unexplained_omission(self):
        for state in ("camera_missing", "person_absent", "empty"):
            ep, table = sequence_fixture()
            mark_missing(table, table["observations"][1], state)
            result = self.validate(ep, table)
            self.assertEqual(result["observation_counts"][state], 1)
            table["observations"][1]["reason"] = ""
            self.rejected(ep, table, "unexplained_missing_observation")
        ep, table = sequence_fixture(); table["observations"].pop()
        self.rejected(ep, table, "missing_observation_slot")

    def test_duplicate_observation_and_packed_mapping(self):
        for change, code in [
            (lambda t: t["observations"].append(copy.deepcopy(t["observations"][0])), "duplicate_observation_id"),
            (lambda t: t["observations"].append(dict(t["observations"][0], observation_id="other")), "duplicate_observation"),
            (lambda t: t["boxes"][1].update(packed_row_id=0), "duplicate_or_invalid_packed_row"),
            (lambda t: t["boxes"][1].update(packed_row_id=9), "packed_row_coverage_mismatch"),
        ]:
            ep, table = sequence_fixture(); change(table); self.rejected(ep, table, code)

    def test_identity_source_and_dual_camera_asset_frame(self):
        for key, value, code in [("source_ref", "other.json", "person_source_conflict"),
                                 ("character_family_id", "other", "person_source_conflict"),
                                 ("asset_id", "other", "person_source_conflict"),
                                 ("asset_frame_index", 2, "cross_camera_asset_frame_conflict"),
                                 ("asset_frame_index", None, "unexplained_null_asset_frame")]:
            ep, table = sequence_fixture(); table["observations"][1][key] = value
            self.rejected(ep, table, code)

    def test_multiple_derivatives_keep_parent_and_sequence(self):
        ep, table = sequence_fixture()
        parent = table["boxes"][0]
        parent["packed_row_id"] = None
        for name, packed_row in (("child1", 0), ("child2", 6)):
            table["boxes"].append(dict(parent, box_id=name, parent_box_id="box0",
                bbox_kind="synthetic_perturbation", packed_row_id=packed_row))
        table["packed"]["row_count"] = 7
        self.assertEqual(self.validate(ep, table)["packed_rows"], 7)
        table["boxes"][-1]["observation_id"] = table["observations"][2]["observation_id"]
        self.rejected(ep, table, "derivative_sequence_conflict")
        table["boxes"][-1]["observation_id"] = parent["observation_id"]
        table["boxes"][-1]["parent_box_id"] = "gone"
        self.rejected(ep, table, "missing_or_self_parent_box")
        table["boxes"][-1]["parent_box_id"] = "child1"
        table["boxes"][-2]["parent_box_id"] = "child2"
        self.rejected(ep, table, "parent_cycle")

    def test_real_nominal_frames_and_tracks_do_not_prove_sync_or_identity(self):
        ep, table = sequence_fixture()
        ep.update(data_kind="real", time_semantics="nominal_index", assets=[])
        ep["synchronization"] = dict(status="unverified", basis="frame_index_only",
                                     evidence_ref=None, reason="dataset3 exposure synchronization unconfirmed")
        ep["people"][0].update(asset_id=None, identity_status="track_only", identity_reason="unreviewed tracker")
        for obs in table["observations"]:
            obs.update(asset_id=None, asset_frame_index=None, asset_frame_status="not_applicable",
                       asset_frame_reason="real observation has no rendered asset", track_id=16)
        result = self.validate(ep, table)
        self.assertEqual(result["declared_synchronization"], "unverified")
        self.assertFalse(result["identity_confirmed_by_validator"])
        context = row_context(ep, table, 0)
        self.assertEqual(context["person"]["identity_status"], "track_only")
        self.assertEqual(context["synchronization"]["status"], "unverified")
        ep["synchronization"]["status"] = "synthetic_same_state"
        self.rejected(ep, table, "real_cannot_claim_synthetic_sync")
        ep["synchronization"]["status"] = "verified_exposure"
        self.rejected(ep, table, "unsupported_exposure_sync_claim")
        ep["time_semantics"] = "measured_pts"
        self.rejected(ep, table, "unsupported_exposure_sync_claim")

    def test_temporary_dependencies_even_if_files_not_checked(self):
        for bad in ("source/cam2.exr", "_temporary_ds_source/x.png", "x/../source/x.png"):
            ep, table = sequence_fixture(); table["observations"][0]["image"] = bad
            self.rejected(ep, table, "temporary_dependency")
        ep, table = sequence_fixture()
        ep["ephemeral_roots"] = ["custom_scratch"]
        table["observations"][0]["image"] = "custom_scratch/image.png"
        self.rejected(ep, table, "temporary_dependency")

    def test_file_headers_origin_rows_and_npz_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep, table = sequence_fixture(); write_fixture_files(root, ep, table)
            def validate():
                return validate_sequence_index(ep, table, base_dir=root, check_image_headers=True, check_packed=True)
            self.assertEqual(validate()["image_headers_read"], 3)
            (root / "person.png").unlink()
            with self.assertRaisesRegex(SequenceIndexError, "missing_long_term_file"):
                validate()
            Image.fromarray(np.zeros((10, 10), np.uint8)).save(root / "person.png")
            with self.assertRaisesRegex(SequenceIndexError, "file_header_size_conflict"):
                validate()
            write_fixture_files(root, ep, table)
            table["observations"][0]["annotation_person_index"] = 1
            with self.assertRaisesRegex(SequenceIndexError, "original_annotation_identity_mismatch"):
                validate()
            table["observations"][0]["annotation_person_index"] = 0
            table["boxes"][0]["bbox_xyxy"] = [21, 10, 50, 50]
            with self.assertRaisesRegex(SequenceIndexError, "origin_box_mismatch"):
                validate()
            table["boxes"][0]["bbox_xyxy"] = [20, 10, 50, 50]
            table["boxes"][0]["crop_policy"] = "deployment"
            with self.assertRaisesRegex(SequenceIndexError, "packed_crop_mismatch"):
                validate()

    def test_legacy_npz_and_dataset_without_sidecar_still_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep, table = sequence_fixture(); write_fixture_files(root, ep, table)
            dataset = load_box_dataset(root / "packed.npz")
            before = dataset[0]
            validate_sequence_index(ep, table, base_dir=root, check_packed=True)
            after = dataset[0]
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                if isinstance(before[key], np.ndarray):
                    np.testing.assert_array_equal(before[key], after[key])
                else:
                    self.assertEqual(before[key], after[key])
            self.assertNotIn("sequence_id", after)
            self.assertEqual(int((after["proj_verts_cropped"][:, 2] > 0).sum()), 138)
