"""Small metadata fixtures; these are not rendered or qualified scene assets."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from densekp_retrain.data.mamma_export import export_existing
from densekp_retrain.data.mamma_crop import mask_native_support
from densekp_retrain.data.sequence_packing import optional_mamma_512_refs


def _json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data)+'\n')


def test_export_fixture(tmp_path):
    run=tmp_path/'run';run.mkdir()
    people=[];qualified={}
    for pid in (7,9):
        manifest=tmp_path/f'asset{pid}.json';body=tmp_path/f'body{pid}.npy'
        _json(manifest,dict(body_vertices=str(body),body_faces=str(tmp_path/'faces.npy')))
        people.append(dict(person_id=pid,asset_id=f'a{pid}',asset_manifest=str(manifest)))
        qualified[str(body)]='fixture_only'
    registry=tmp_path/'registry.json';_json(registry,dict(schema='fixture_qualification_registry_v1',qualified_assets=qualified))
    rig_rows=[]
    for rid in ('rig_alpha','rig_beta'):
        root=run/'rigs'/rid;root.mkdir(parents=True)
        rig_rows.append(dict(rig_id=rid,root=str(root)))
        _json(root/'recipe.json',dict(rig_id=rid,scene_episode_id='fixture_ep',people=people,frame_count=31,
            scene_matrices_ref=str(tmp_path/'scene.npy'),person_timeline_refs={'7':'timeline7','9':'timeline9'}))
        _json(root/'rig.json',dict(cameras=[dict(id='cam2'),dict(id='cam4')]))
        _json(root/'label_policy.json',dict(visibility_policy='fixture_unknown',sidecar_policy='mask_native_v2'))
        for camera in ('cam2','cam4'):
            dest=root/'frame_005'/camera;dest.mkdir(parents=True)
            _json(dest/'raster_storage.json',dict(roi_xyxy=[0,0,100,100],full_image_size_wh=[100,100],
                stored_size_wh=[100,100],physical_mask_ref=str(tmp_path/'physical.png')))
            _json(dest/'boxes.json',[dict(person_id=pid,bbox_xyxy_exclusive=[40,40,60,60],source='fixture') for pid in (7,9)])
            for name in ('roi.png','valid_mask.png','person_7_mask.png','person_9_mask.png'):
                array=np.zeros((100,100),np.uint8)
                if name.startswith('person_'):array[40:61,40:61]=255
                Image.fromarray(array).save(dest/name)
            np.savez(dest/'annotations.npz',person_ids=np.array([7,9]),frame_index=5,time_tick=5,time_base=[1,10],
                person_asset_frame=[5,5],person_source_frame=[5,5],person_source_time_tick=[5,5],
                person_source_time_base=np.array([[1,10],[1,10]]))
            np.savez(dest/'mamma_512.npz',schema='mamma_512_evidence_v1',sidecar_policy='mask_native_v2',
                person_ids=np.array([7,9]),frame_index=5,time_tick=5,camera_id=camera,rig_id=rid,
                scene_episode_id='fixture_ep',xy_full_erp=np.zeros((2,512,2)),visibility_value=np.full((2,512),-1),
                visibility_known=np.zeros((2,512),bool),ordered_vertex_ids=np.arange(512),template_id='fixture',
                asset_frame_index=np.array([5,5]),source_frame_index=np.array([5,5]),time_base=np.array([1,10]))
    _json(run/'run_config.json',dict(selected_frames=[5],preview_only=False))
    _json(run/'multirig.json',dict(scene_episode_id='fixture_ep',rigs=rig_rows))
    export=lambda:export_existing(run,registry,allow_fixture_registry=True)
    result=export()
    assert len(result['observations'])==8 and result['frame_scope']=='selected_frames_only'
    assert {r['person_id'] for r in result['observations']}=={7,9}
    assert {r['rig_id'] for r in result['observations']}=={'rig_alpha','rig_beta'}
    assert all('box_support' not in r['files'] for r in result['observations'])
    cd=run/'rigs/rig_alpha/frame_005/cam2'
    with np.load(cd/'annotations.npz') as ann:
        refs=optional_mamma_512_refs(cd,annotation_person_ids=[7,9],person_index=0,frame_index=5,
            camera_id='cam2',label_policy_path=run/'rigs/rig_alpha/label_policy.json',annotation=ann,
            rig_id='rig_alpha',scene_episode_id='fixture_ep')
    assert 'mamma_box_support_ref' not in refs
    def rejects(reason,fn):
        try:fn()
        except ValueError as exc:assert reason in str(exc),exc
        else:raise AssertionError('expected rejection '+reason)
    _json(registry,dict(schema='fixture_qualification_registry_v1',qualified_assets={str(tmp_path/'body7.npy'):'fixture_only'}))
    rejects('unqualified_asset',export)
    _json(registry,dict(schema='fixture_qualification_registry_v1',qualified_assets=qualified))
    labels=cd/'mamma_512.npz';labels.rename(cd/'held.npz')
    rejects('missing_mamma_512_label',export)
    (cd/'held.npz').rename(labels)
    boxes=_json(cd/'boxes.json',[dict(person_id=pid,bbox_xyxy_exclusive=[40,40,60,60],source='fixture') for pid in (9,7)])
    rejects('box_person_order_conflict',export)
    _json(cd/'boxes.json',[dict(person_id=pid,bbox_xyxy_exclusive=[40,40,60,60],source='fixture') for pid in (7,9)])
    with np.load(labels) as z:payload={k:z[k] for k in z.files}
    payload['camera_id']=np.array('cam4');np.savez(labels,**payload)
    rejects('mamma_annotation_identity_time_conflict',export)
    payload['camera_id']=np.array('cam2');payload['time_tick']=np.array(6);np.savez(labels,**payload)
    rejects('mamma_annotation_identity_time_conflict',export)
    payload['time_tick']=np.array(5);np.savez(labels,**payload)
    _json(cd/'mamma_box_support.json',dict(camera_id='cam2',frame_index=5,people=[dict(person_id=pid,
        native_box_xyxy=[40,40,60,60],variants=[]) for pid in (7,9)]))
    rejects('mamma_sidecar_policy_conflict',export)
    payload.pop('sidecar_policy');np.savez(labels,**payload)
    assert optional_mamma_512_refs(cd,annotation_person_ids=[7,9],person_index=1,frame_index=5,
        camera_id='cam2',label_policy_path=run/'rigs/rig_alpha/label_policy.json')['mamma_box_support_ref']
    assert len(export()['observations'])==8


def test_crop_support():
    mask=np.zeros((100,100),np.uint8);mask[40:61,40:61]=255
    assert mask_native_support(mask,[0,0,100,100],[100,100])['box_xyxy']==[40,40,60,60]
    mask[:]=0;mask[2:23,2:23]=255
    try:mask_native_support(mask,[0,0,100,100],[100,100])
    except ValueError as exc:assert 'crop_filter_support_outside_roi' in str(exc)
    else:raise AssertionError('expected crop rejection')
