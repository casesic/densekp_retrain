"""Actual DenseKP preprocessing of studio diagnostics; no inference or DS claim."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.keypoint_supervision import make_targets
from densekp.core.datasets.configuration import audit_cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review',type=Path)
    args=parser.parse_args()
    output=args.review/'loader_diagnostic';output.mkdir(exist_ok=False)
    views=json.loads((args.review/'review_manifest.json').read_text())['views']
    rows=[]
    # Explicit valid rectangle for this orthographic studio, not a DS lens mask.
    Image.new('L',(640,640),255).save(output/'studio_valid_mask.png')
    for view in views:
        if 'xy' not in view:continue
        xy=np.asarray(view['xy'],np.float32)
        box=np.asarray(view['bbox_xyxy']);center=(box[:2]+box[2:])/2
        scale=float((box[2:]-box[:2]).max()*1.2/200)
        in_view=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<640)
        rows.append(dict(view,center=center,scale=scale,target=make_targets(xy,in_view,center,scale)))
    path=output/'loader_input.npz'
    # Visibility is deliberately absent: this artifact tests image preprocessing only.
    # Targets retain explicit in-view/full-crop membership; no augmentation is used.
    np.savez(path,imgname=np.array([r['path'] for r in rows]),
             center=np.array([r['center'] for r in rows],np.float32),
             scale=np.array([r['scale'] for r in rows],np.float32),
             proj_verts=np.stack([r['target'] for r in rows]),split=np.array('integration_only'))
    cfg=audit_cfg('supervision_weight');dataset_name='commuter-studio-diagnostic'
    records=[];panels=[]
    dataset = DatasetKeypoints(
        cfg,
        dataset_name,
        is_train=False,
        use_augmentation=False,
        annotation_path=str(path),
        image_root="",
    )
    for i, row in enumerate(rows):
        item = dataset[i]
        restored = restore_crop_xy(item['proj_verts_cropped'][:, :2], item['_trans'], (256, 256))
        error = float(np.abs(restored - item['proj_verts'][:, :2]).max())
        if error > 0.0002:
            raise ValueError(f'Existing crop gate failed: {error}')
        if item['img'].shape != (3, 256, 256) or not np.isfinite(item['img']).all():
            raise ValueError('Invalid loader input')
        np.testing.assert_array_equal(item['proj_verts_cropped'][:, 2], row['target'][:, 2])
        rgb = cv2.cvtColor(item['img_disp'], cv2.COLOR_BGR2RGB)
        name = f"{row['character_id']}_{row['pose_source_id']}_{row['view']}"
        Image.fromarray(rgb).save(output / (name + '_square.png'))
        Image.fromarray(rgb[:, 32:-32]).save(output / (name + '_network_center.png'))
        np.save(output / (name + '_network_input.npy'), item['img'][:, :, 32:-32])
        record = dict(character_id=row['character_id'], pose_source_id=row['pose_source_id'], view=row['view'], roundtrip_max_px=error, square_shape=[256, 256], network_shape=[256, 192])
        records.append(record)
        if row['pose_source_id'] != 'reference' and row['view'] == 'front':
            panel = Image.new('RGB', (448, 280), (35, 35, 35))
            panel.paste(Image.fromarray(rgb), (0, 0))
            panel.paste(Image.fromarray(rgb[:, 32:-32]), (256, 0))
            ImageDraw.Draw(panel).text((4, 260), name, fill='white')
            panels.append(panel)
    sheet=Image.new('RGB',(896,280*max(1,(len(panels)+1)//2)),(35,35,35))
    for i,panel in enumerate(panels):sheet.paste(panel,(448*(i%2),280*(i//2)))
    sheet.save(output/'square_and_network_center.jpg')
    (output/'report.json').write_text(json.dumps(dict(status='preprocessing_passed',split='integration_only',
        rows=len(rows),max_roundtrip_px=max(r['roundtrip_max_px'] for r in records),
        visibility='not computed; not a supervised training dataset',
        limitation='studio camera only; DS projection, masks, visibility and instances remain untested for these assets',
        network_crop_source='densekp/core/densekp_trainer.py forward_step: x[:,:,:,32:-32]',
        cases=records),indent=2)+'\n')


if __name__=='__main__':main()
