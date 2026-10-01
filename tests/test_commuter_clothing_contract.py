"""A rigidly posed loose garment must reconstruct; unreviewed assets cannot reach DS."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.render.commuter_clothing_contract import validate_binding


class CommuterBindingTests(unittest.TestCase):
    def test_rotation_provenance_and_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rest=np.array([[1.,0,0],[0,1,0],[0,0,0]],np.float32)
            rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32)
            translation=np.array([.5,-.2,.3],np.float32)
            offsets=np.tile([.03,0,0],(3,1)).astype(np.float32)
            reference=(rest@rotation.T+translation)[None]
            final=((rest+offsets)@rotation.T+translation)[None]
            for name,value in dict(reference_body_vertices=reference,rest_offsets=offsets,
                                   binding_rotations=np.tile(rotation,(3,1,1)),
                                   collision_correction=np.zeros((3,3),np.float32)).items():
                np.save(root/f'{name}.npy',value)
            info=dict(contract='commuter_surface_binding_v1',status='numeric_candidate_visual_pending',
                      collision=dict(negative_vertex_count=0,negative_face_center_count=0))
            path=root/'clothing_manifest.json';path.write_text(json.dumps(info))
            self.assertLess(validate_binding(root,reference,final)['binding_reconstruction_max_m'],2e-6)
            with self.assertRaisesRegex(ValueError,'studio acceptance'):
                validate_binding(root,reference,final,require_studio=True)
            with self.assertRaisesRegex(ValueError,'posed body'):
                validate_binding(root,reference+.001,final)
            with self.assertRaisesRegex(ValueError,'reconstruct'):
                validate_binding(root,reference,final+.001)
            info['status']='studio_accepted';path.write_text(json.dumps(info))
            validate_binding(root,reference,final,require_studio=True)


if __name__=='__main__':unittest.main()
