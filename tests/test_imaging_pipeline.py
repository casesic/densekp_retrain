"""Independent pre-migration fixtures, plus separate compatibility checks."""
import base64
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from densekp_retrain.imaging import ds, omni, remap, source
from densekp_retrain.camera import OmniRadtanIntrinsics
from tests.imaging_pipeline_fixture import capture
from tools.validate.verify_imaging_pipeline import compare


class ImagingPipelineTests(unittest.TestCase):
    def test_full_small_artifacts_and_partial_failure_outputs_match_old_fixture(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/imaging_pipeline_v1.json').read_text())
        with TemporaryDirectory() as temporary:
            root=Path(temporary);before=root/'before'; before.mkdir()
            with zipfile.ZipFile(BytesIO(base64.b64decode(fixture['zip_base64']))) as archive:
                archive.extractall(before)
            capture(root/'after',migrated=True)
            result=compare(before,root/'after')
            self.assertTrue(result['exact'])
            self.assertEqual(result['counts'],fixture['counts'])

    def test_ds_source_keeps_preview_rgb_and_original_erp_range(self):
        rgb=np.zeros((3,4,3),np.float32);depth=np.full((3,4),7,np.float32);ids=np.ones((3,4),np.float32)
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            Image.fromarray(np.full((3,4,3),150,np.uint8)).save(root/'cam2_preview.png')
            with patch.object(source,'read_combined_depth_index',return_value=(rgb,depth,ids,[])):
                loaded=source.read_ds_source(root,'cam2')
            self.assertIs(loaded.ray_range_m,depth)
            self.assertIs(loaded.instance,ids)
            self.assertEqual(loaded.rgb.dtype,np.uint8)
            self.assertTrue((loaded.rgb==150).all())

    def test_omni_clears_only_remapped_depth_and_keeps_source_range(self):
        k=OmniRadtanIntrinsics(width=32,height=32,cx=16,cy=16,f=5)
        depth=np.full((24,48),np.inf,np.float32)
        raw=source.SourceERP(np.ones((24,48,3),np.float32),depth,np.ones((24,48),np.float32),[])
        mapping=omni.build_omni_source_map(raw,k)
        buffers=omni.remap_omni_source(raw,mapping)
        omni.apply_omni_domain(buffers,mapping)
        self.assertTrue(np.isinf(raw.ray_range_m).all())
        self.assertTrue(np.isfinite(buffers.ray_range_m).all())
        self.assertFalse(buffers.ray_range_m.any())
        self.assertFalse(mapping.valid.all())

    def test_periodic_sampler_wraps_longitude_without_wrapping_poles(self):
        rgb=np.array([[10,20,30,40],[50,60,70,80]],np.float32)
        x=np.array([[3.75,3.75]],np.float32); y=np.array([[0,2]],np.float32)
        periodic=remap.remap_periodic_erp(rgb,x,y,cv2.INTER_LINEAR)
        constant=cv2.remap(rgb,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        np.testing.assert_array_equal(periodic,[[17.5,0]])
        np.testing.assert_array_equal(constant,[[10,0]])

    def test_historical_exports_are_compatibility_only(self):
        from tools.fisheye import run_ds_scene_pair_pilot as old_ds
        from tools.fisheye import run_fisheye_camera_p0 as old_omni
        from tools.fisheye import build_ds_multiperson_pilot as old_remap
        self.assertIs(old_ds.process_scene,ds.process_scene)
        self.assertIs(old_omni.warp_case,omni.warp_case)
        self.assertIs(old_omni.scalar_reference,omni.scalar_reference)
        self.assertIs(old_remap.remap_periodic_erp,remap.remap_periodic_erp)
        self.assertIs(old_remap.lens_mask_from_reference,remap.lens_mask_from_reference)
