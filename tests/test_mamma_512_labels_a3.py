import numpy as np
from pathlib import Path

from densekp_retrain.camera.visibility import erp_visibility,erp_visibility_points
from densekp_retrain.data.mamma_labels import fixed_seven_boxes,ray_mesh_first_range,seven_box_support
from densekp_retrain.data.occlusion_events import occlusion_evidence,point_occlusion_evidence
from densekp_retrain.data.sequence_packing import optional_mamma_512_refs


def test_arbitrary_points_preserve_legacy_r138_policy():
    rng=np.random.default_rng(4);xy=np.column_stack([rng.uniform(3,28,276),rng.uniform(3,12,276)])
    distance=rng.uniform(1,4,276);depth=np.full((16,32),9.0);instance=np.zeros((16,32))
    for j,(x,y) in enumerate(xy):
        depth[int(y),int(x)]=distance[j];instance[int(y),int(x)]=1+j//138
    valid=np.isfinite(xy).all(1)&(xy[:,1]>=0)&(xy[:,1]<depth.shape[0]);reference=np.zeros(276,bool)
    for j,(x,y) in enumerate(xy):
        if not valid[j]:continue
        ix,iy=int(np.floor(x))%depth.shape[1],int(np.floor(y));rows=np.arange(max(0,iy-2),min(depth.shape[0],iy+3));cols=np.arange(ix-2,ix+3)%depth.shape[1]
        sd=depth[np.ix_(rows,cols)];si=instance[np.ix_(rows,cols)];matched=(sd>0)&(np.rint(si)==[1,2][j//138])
        reference[j]=bool(matched.any() and np.min(np.abs(sd[matched]-distance[j]))<.04)
    old=erp_visibility(xy,distance,depth,instance,[1,2])
    new=erp_visibility_points(xy,distance,depth,instance,np.repeat([1,2],138))
    np.testing.assert_array_equal(old[0],valid);np.testing.assert_array_equal(old[1],reference)
    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])


def test_general_evidence_wrapper_keeps_legacy_arrays():
    xy=np.full((2,138,2),[8.2,8.2]);distance=np.full((2,138),2.0)
    depth=np.full((20,20),1.0);instance=np.full((20,20),2.0)
    visible=np.zeros((2,138),bool);inside=np.ones((2,138),bool)
    legacy=occlusion_evidence(xy,distance,visible,inside,[1,2],depth,instance)
    flat=point_occlusion_evidence(xy.reshape(-1,2),distance.reshape(-1),visible.reshape(-1),inside.reshape(-1),
                                  np.repeat([1,2],138),[1,2],depth,instance)
    for key,value in legacy.items():np.testing.assert_array_equal(value,flat[key].reshape(2,138))


def test_fixed_seven_boxes_are_independent_variants_and_can_reject():
    rows=fixed_seven_boxes([400,300,600,700]);assert [r['name'] for r in rows]==[
        'native','center_x_minus','center_x_plus','center_y_minus','center_y_plus','scale_0p9','scale_1p1']
    np.testing.assert_allclose(rows[1]['center_xy'],[494,500]);np.testing.assert_allclose(rows[3]['center_xy'],[500,488])
    assert all(r['supported'] for r in seven_box_support([400,300,600,700],[0,0,1000,1000],[1000,1000]))
    assert any(not r['supported'] for r in seven_box_support([400,300,600,700],[480,350,800,800],[1000,1000]))


def test_body_ray_distinguishes_front_and_back_surface():
    vertices=np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],float)
    faces=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6]],int)
    first,target=ray_mesh_first_range([0,0,0],np.array([[0,0,1],[0,0,2]],float),vertices,faces,ray_chunk=2)
    np.testing.assert_allclose(first,[1,1]);np.testing.assert_allclose(target,[1,2])


def test_sequence_pack_optional_mamma_refs_and_old_compatibility(tmp_path):
    assert optional_mamma_512_refs(tmp_path,annotation_person_ids=[1,2],person_index=0,
        frame_index=0,camera_id='cam2',label_policy_path=tmp_path/'missing.json')=={}
    root=Path('outputs/experiments/mamma_512_label_pilot_a3_20261015/run/rigs/rig_back')
    camera_dir=root/'frame_000'/'cam2'
    if not camera_dir.is_dir():return
    refs=optional_mamma_512_refs(camera_dir,annotation_person_ids=[1,2],person_index=1,
        frame_index=0,camera_id='cam2',label_policy_path=root/'label_policy.json')
    assert refs['mamma_512_person_index']==1
    assert Path(refs['mamma_512_ref']).is_file()
    assert Path(refs['mamma_box_support_ref']).is_file()
