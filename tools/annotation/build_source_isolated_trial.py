"""Pack accepted index rows into actual loaders without copying them to datasets/."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp.core.configs import DATASET_FILES,DATASET_FOLDERS
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.keypoint_supervision import SUPERVISION_POLICY
from densekp.core.crop_geometry import restore_crop_xy
from tools.validate.validate_bedlam_densekp_tiny import audit_cfg
from tools.validate.validate_source_isolated_split import validate_split


def main():
    root=Path(sys.argv[1]).resolve();split=json.loads((root/'split_v2.json').read_text())
    indices={s:json.loads((root/f'{s}_index.json').read_text()) for s in ['train','val']}
    isolation=validate_split(split,indices);out=root/'loader';out.mkdir(exist_ok=False);records=[]
    for side,index in indices.items():
        fields={k:[] for k in ['imgname','center','scale','proj_verts','in_view','visible','person_id','camera_id']}
        cache={}
        for row in index['rows']:
            path=row['loader_input']
            if path not in cache:
                with np.load(path) as z:cache[path]={k:z[k] for k in z.files}
            z=cache[path];i=row['loader_row']
            if str(z['supervision_policy'])!=SUPERVISION_POLICY:raise ValueError('Supervision mismatch')
            for key in fields:fields[key].append(str(Path(row['image']).resolve()) if key=='imgname' else z[key][i])
        target=out/f'{side}.npz'
        np.savez(target,**{k:np.asarray(v) for k,v in fields.items()},supervision_policy=SUPERVISION_POLICY,split=side)
        # Test the true training and validation DatasetKeypoints branches. No aug
        # in this bounded experiment, so saved baseline inputs must match exactly.
        train=side=='train'
        with patch.dict(DATASET_FOLDERS,{'source-trial':str(root)}),patch.dict(DATASET_FILES[train],{'source-trial':str(target)}):
            dataset=DatasetKeypoints(audit_cfg('supervision_weight'),'source-trial',is_train=train,use_augmentation=False)
        supervised=occluded=0
        for i,row in enumerate(index['rows']):
            item=dataset[i];mask=item['proj_verts_cropped'][:,2]>0
            np.testing.assert_array_equal(cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB),np.array(Image.open(row['square'])))
            np.testing.assert_array_equal(item['img'][:,:,32:-32],np.load(Path(row['square']).with_name(Path(row['square']).name.replace('_square.png','_input.npy'))))
            np.testing.assert_allclose(item['_trans'].ravel(),row['affine_flat'],atol=1e-7,rtol=0)
            np.testing.assert_allclose(restore_crop_xy(item['proj_verts_cropped'][:,:2],item['_trans'],(256,256)),
                                       dataset.proj_verts[i,:,:2],atol=2e-4,rtol=0)
            if int(mask.sum())!=row['supervised_points']:raise ValueError('Merged loader supervision mismatch')
            supervised+=int(mask.sum());occluded+=int((mask&~item['visible']).sum())
        records.append(dict(split=side,path=str(target),rows=len(dataset),supervised=supervised,occluded=occluded,
            actual_training_branch=train,augmentation=False,input_and_affine_exact=True))
    (out/'acceptance.json').write_text(json.dumps(dict(status='passed',source_isolation=isolation,loaders=records),indent=2)+'\n')
    print(records)

if __name__=='__main__':main()
