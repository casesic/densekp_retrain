"""Independent old Dataset/crop outputs plus explicit-source API boundaries."""

import base64
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from densekp.core.configs import DATASET_FILES, DATASET_FOLDERS
from densekp.core.datasets.configuration import audit_cfg
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.loading import resolve_source, resolve_image_path
from tests.dataset_crop_fixture import make_inputs, run_dataset, run_lowlevel


class DatasetCropContractTests(unittest.TestCase):
    def setUp(self):
        self.states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "inputs"
        make_inputs(self.root)
        text = (Path(__file__).parent / "fixtures/dataset_crop_v1.json").read_text()
        self.fixture = json.loads(text.replace("<INPUTS>", str(self.root)))

    def tearDown(self):
        random.setstate(self.states[0])
        np.random.set_state(self.states[1])
        torch.set_rng_state(self.states[2])
        self.temporary.cleanup()

    def assert_frozen(self, metadata, arrays, fixture):
        self.assertEqual(metadata, fixture["metadata"])
        with np.load(
            io.BytesIO(base64.b64decode(fixture["arrays_npz_base64"])), allow_pickle=False
        ) as old:
            self.assertEqual(list(arrays), old.files)
            for key, value in arrays.items():
                expected = old[key]
                if key == 'item_imgname':
                    # A portable fixture changes the temporary directory's text
                    # and hence the Unicode width, never numeric data/dtypes.
                    expected = np.asarray(expected.item().replace('<INPUTS>', str(self.root)))
                self.assertEqual(value.dtype, expected.dtype, key)
                self.assertEqual(value.shape, expected.shape, key)
                np.testing.assert_array_equal(value, expected, err_msg=key)

    def test_old_id_and_explicit_sources_match_independent_augmented_outputs(self):
        folders = dict(DATASET_FOLDERS)
        files = [dict(value) for value in DATASET_FILES]
        for fixture in self.fixture["dataset"]:
            for explicit in (False, True):
                with self.subTest(case=fixture["spec"]["id"], explicit=explicit):
                    self.assert_frozen(*run_dataset(fixture["spec"], explicit=explicit), fixture)
        self.assertEqual(DATASET_FOLDERS, folders)
        self.assertEqual(DATASET_FILES, files)

    def test_named_outputs_mutations_failures_and_rng(self):
        for fixture in self.fixture["lowlevel"]:
            with self.subTest(case=fixture["spec"]["id"]):
                self.assert_frozen(*run_lowlevel(fixture["spec"], self.root), fixture)

    def test_explicit_fields_override_independently_without_registry_writes(self):
        with (
            patch.dict(DATASET_FOLDERS, {"test-source": "old-root"}),
            patch.dict(DATASET_FILES[False], {"test-source": "old-annotation"}),
        ):
            self.assertEqual(
                resolve_source("test-source", False, annotation_path=Path("new.npz")).image_root,
                "old-root",
            )
            self.assertEqual(
                resolve_source("test-source", False, image_root="").annotation_path,
                "old-annotation",
            )
            self.assertEqual(resolve_source("test-source", False, image_root="").image_root, "")
            self.assertEqual(DATASET_FOLDERS["test-source"], "old-root")
        source = resolve_source(
            "unregistered", False, annotation_path=Path("a.npz"), image_root=Path("images")
        )
        self.assertEqual((source.annotation_path, source.image_root), ("a.npz", "images"))
        with self.assertRaises(KeyError):
            resolve_source("unregistered", False, annotation_path="a.npz")
        self.assertEqual(
            resolve_image_path("relative/root", "../image.png"), "relative/root/../image.png"
        )
        self.assertEqual(resolve_image_path("unused", "/absolute/image.png"), "/absolute/image.png")

    def test_dataset_name_default_and_existing_img_dir_attribute(self):
        cfg = audit_cfg("supervision_weight")
        dataset = DatasetKeypoints(
            cfg,
            annotation_path=self.root / "relative.npz",
            image_root="/unused",
            is_train=False,
            use_augmentation=False,
        )
        dataset.img_dir = str(self.root)
        sample = dataset[0]
        self.assertEqual(sample["dataset"], "explicit")
        self.assertEqual(sample["imgname"], str(self.root / "image.png"))

    def test_evaluation_disables_requested_augmentation(self):
        cfg = audit_cfg("supervision_weight")
        cfg.DATASETS.CONFIG.SCALE_FACTOR = 0.5
        cfg.DATASETS.CONFIG.ROT_FACTOR = 30.0
        cfg.DATASETS.CONFIG.ROT_AUG_RATE = 1.0
        dataset = DatasetKeypoints(
            cfg,
            "validation",
            annotation_path=self.root / "relative.npz",
            image_root=self.root,
            is_train=False,
            use_augmentation=True,
        )
        with patch(
            "densekp.core.datasets.crop_pipeline.sample_augmentation",
            side_effect=AssertionError("validation must not sample"),
        ):
            sample = dataset[0]
        disabled = DatasetKeypoints(
            cfg,
            "validation",
            annotation_path=self.root / "relative.npz",
            image_root=self.root,
            is_train=True,
            use_augmentation=False,
        )[0]
        for key in ("img", "proj_verts_cropped", "_trans", "box_center"):
            np.testing.assert_array_equal(sample[key], disabled[key])

    def test_flip_rejection_survives_postconstruction_config_changes(self):
        cfg = audit_cfg("supervision_weight")
        dataset = DatasetKeypoints(
            cfg, annotation_path=self.root / "relative.npz", image_root=self.root
        )
        cfg.DATASETS.CONFIG.DO_FLIP = True
        cfg.DATASETS.CONFIG.FLIP_AUG_RATE = 1.0
        with patch(
            "densekp.core.datasets.dataset_keypoints.read_image",
            side_effect=AssertionError("must reject before image IO"),
        ):
            with self.assertRaisesRegex(ValueError, "verified 138-point semantic permutation"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
