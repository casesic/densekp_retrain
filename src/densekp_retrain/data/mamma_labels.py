"""MAMMA 512 geometry, evidence and fixed rectangular crop support.

This module produces model-independent labels.  It does not import or run the
MAMMA network.  Visibility is a conservative project policy while the official
production definition remains pending.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.visibility import erp_visibility_points
from densekp_retrain.data.dynamic_scene import frame_record, geometry_paths
from densekp_retrain.data.occlusion_events import point_occlusion_evidence


SCHEMA='mamma_512_evidence_v1'
POLICY='mamma512_conservative_visibility_v1_compatible_definition_pending'
REASON={
    'physical_domain_outside':0,
    'visible_body_and_render_surface':1,
    'occluded_by_other_person':2,
    'occluded_by_scene':3,
    'own_instance_ambiguous':4,
    'contour_or_depth_ambiguous':5,
    'render_coverage_missing':6,
    'body_ray_unresolved':7,
}


def ordered_vertex_ids(template_path):
    matrix=joblib.load(str(Path(template_path).resolve()))
    if hasattr(matrix,'detach'):matrix=matrix.detach().cpu().numpy()
    matrix=np.asarray(matrix)
    if (matrix.shape!=(512,10475) or not np.isfinite(matrix).all() or
            not np.all((matrix==0)|(matrix==1)) or not np.all(matrix.sum(1)==1)):
        raise ValueError('invalid_mamma_512_template')
    ids=matrix.argmax(1).astype(np.int32)
    if len(np.unique(ids))!=512:raise ValueError('duplicate_mamma_template_vertex')
    return ids


def load_world_body(recipe,root,frame,template_path):
    """Load exact asset frames and apply scene_from_model exactly once."""
    root=Path(root);ids=ordered_vertex_ids(template_path);timing=frame_record(recipe,frame)
    matrices=np.load(recipe.get('scene_matrices_ref',root/'scene_matrices.npy'),mmap_mode='r')
    if matrices.shape[:2]!=(recipe['frame_count'],len(recipe['people'])):raise ValueError('scene_matrix_shape_conflict')
    selected=[];full=[];faces=[];asset_frames=[];source_frames=[]
    records=timing.get('person_records')
    for j,person in enumerate(recipe['people']):
        asset=json.loads(Path(person['asset_manifest']).read_text());paths=geometry_paths(asset)
        record=records[j] if records is not None else None
        af=int(record['asset_frame'] if record else person['asset_frames'][frame])
        sf=int(record['source_frame'] if record else person['source_frames'][frame])
        vertices=np.asarray(np.load(paths['body_vertices'],mmap_mode='r')[af],dtype=np.float64)
        triangles=np.asarray(np.load(paths['body_faces'],mmap_mode='r'),dtype=np.int32)
        if vertices.shape!=(10475,3) or triangles.ndim!=2 or triangles.shape[1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('unqualified_body_geometry')
        M=np.asarray(matrices[frame,j],dtype=np.float64)
        world=vertices@M[:3,:3].T+M[:3,3]
        full.append(world);selected.append(world[ids]);faces.append(triangles)
        asset_frames.append(af);source_frames.append(sf)
    return dict(vertex_ids=ids,world=np.stack(selected),full_body=full,faces=faces,
                person_ids=np.asarray([p['person_id'] for p in recipe['people']],np.int32),
                asset_frames=np.asarray(asset_frames,np.int64),source_frames=np.asarray(source_frames,np.int64),
                timing=timing)


def ray_mesh_first_range(camera_origin,targets,vertices,faces,*,ray_chunk=24,epsilon=1e-7):
    """Bounded dependency-free Moller-Trumbore ray test for body self-occlusion."""
    origin=np.asarray(camera_origin,dtype=np.float64);targets=np.asarray(targets,dtype=np.float64)
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int32)
    tri=vertices[faces];v0=tri[:,0];e1=tri[:,1]-v0;e2=tri[:,2]-v0
    delta=targets-origin;target_range=np.linalg.norm(delta,axis=1);directions=delta/target_range[:,None]
    first=np.full(len(targets),np.inf,dtype=np.float64)
    tvec=origin-v0
    for begin in range(0,len(targets),ray_chunk):
        direction=directions[begin:begin+ray_chunk]
        pvec=np.cross(direction[:,None,:],e2[None,:,:]);det=np.einsum('fj,cfj->cf',e1,pvec)
        usable=np.abs(det)>epsilon;inv=np.zeros_like(det);inv[usable]=1.0/det[usable]
        u=np.einsum('fj,cfj->cf',tvec,pvec)*inv
        qvec=np.cross(tvec,e1);v=np.einsum('cj,fj->cf',direction,qvec)*inv
        distance=np.einsum('fj,fj->f',e2,qvec)[None,:]*inv
        hit=usable&(u>=-epsilon)&(v>=-epsilon)&(u+v<=1+epsilon)&(distance>epsilon)
        distance[~hit]=np.inf;first[begin:begin+len(direction)]=distance.min(axis=1)
    return first,target_range


def build_evidence(world_body,full_body,faces,person_ids,camera_to_world,camera_id,physical_mask,depth,instance,coverage_region):
    """Project 512 ordered points and retain raw range/ID evidence."""
    world=np.asarray(world_body);shape=world.shape[:2];flat=world.reshape(-1,3)
    xy,fish,distance,domain,physical=point_in_view(flat,np.asarray(camera_to_world),camera_id,physical_mask)
    owners=np.repeat(np.asarray(person_ids),shape[1])
    _,surface_match=erp_visibility_points(xy,distance,depth,instance,owners)
    raw=point_occlusion_evidence(xy,distance,surface_match,physical,owners,person_ids,depth,instance,strict_coverage=False)
    roi=np.asarray(coverage_region,int);pixel=np.floor(xy).astype(int)
    coverage=(np.isfinite(xy).all(1)&(pixel[:,0]-2>=roi[0])&(pixel[:,1]-2>=roi[1])&
              (pixel[:,0]+2<roi[2])&(pixel[:,1]+2<roi[3]))
    first=[];target=[]
    for j in range(shape[0]):
        a,b=ray_mesh_first_range(np.asarray(camera_to_world)[:3,3],world[j],full_body[j],faces[j])
        first.append(a);target.append(b)
    first=np.stack(first).reshape(-1);target=np.stack(target).reshape(-1)
    body_resolved=np.isfinite(first);body_front=body_resolved&(np.abs(first-target)<=.003)
    code=raw['occlusion_evidence_code'];front_id=raw['front_instance_id'];gap=raw['gt_minus_front_range_m']
    value=np.full(len(flat),-1,np.int8);known=np.zeros(len(flat),bool)
    reason=np.full(len(flat),REASON['contour_or_depth_ambiguous'],np.uint8)
    reason[~physical]=REASON['physical_domain_outside'];reason[~coverage]=REASON['render_coverage_missing']
    unresolved=physical&coverage&~body_resolved;reason[unresolved]=REASON['body_ray_unresolved']
    own_ambiguous=physical&coverage&(code==3);reason[own_ambiguous]=REASON['own_instance_ambiguous']
    positive=(physical&coverage&body_front&(front_id==owners)&np.isfinite(gap)&(np.abs(gap)<=.015)&
              raw['evidence_neighborhood_complete'])
    neighbor=physical&coverage&(code==2);scene=physical&coverage&(code==4)
    value[positive]=1;known[positive]=True;reason[positive]=REASON['visible_body_and_render_surface']
    value[neighbor|scene]=0;known[neighbor|scene]=True
    reason[neighbor]=REASON['occluded_by_other_person'];reason[scene]=REASON['occluded_by_scene']
    def shaped(a,*tail):return np.asarray(a).reshape(*shape,*tail)
    result=dict(schema=np.array(SCHEMA),policy=np.array(POLICY),person_ids=np.asarray(person_ids,np.int32),
        world_points_m=world.astype(np.float64),xy_full_erp=shaped(xy,2).astype(np.float32),fisheye_xy=shaped(fish,2).astype(np.float32),
        geometry_defined=np.isfinite(world).all(2),physical_domain=shaped(physical),render_coverage=shaped(coverage),
        gt_ray_range_m=shaped(distance).astype(np.float32),surface_id_range_match=shaped(surface_match),
        front_instance_id=shaped(front_id).astype(np.int32),front_range_m=shaped(raw['front_range_m']).astype(np.float32),
        gt_minus_front_range_m=shaped(gap).astype(np.float32),nearer_same_front_id_3x3_count=shaped(raw['nearer_same_front_id_3x3_count']),
        evidence_neighborhood_complete=shaped(raw['evidence_neighborhood_complete']),body_first_range_m=shaped(first).astype(np.float32),
        body_self_frontmost=shaped(body_front),visibility_value=shaped(value),visibility_known=shaped(known),visibility_reason_code=shaped(reason),
        contact_value=np.full(shape,-1,np.int8),contact_known=np.zeros(shape,bool),
        floor_contact_value=np.full(shape,-1,np.int8),floor_contact_known=np.zeros(shape,bool),
        camera_to_world_erp=np.asarray(camera_to_world,dtype=np.float64))
    return result


def fixed_seven_boxes(native_box):
    box=np.asarray(native_box,dtype=np.float64)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):raise ValueError('invalid_native_box')
    center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
    rows=[]
    for name,offset,scale in [
        ('native',(0,0),1.0),('center_x_minus',(-.03,0),1.0),('center_x_plus',(.03,0),1.0),
        ('center_y_minus',(0,-.03),1.0),('center_y_plus',(0,.03),1.0),
        ('scale_0p9',(0,0),.9),('scale_1p1',(0,0),1.1)]:
        c=center+np.asarray(offset)*size;s=size*scale;b=np.r_[c-s/2,c+s/2]
        expanded=s*1.2;target=.75
        if expanded[0]/expanded[1]<target:expanded[0]=expanded[1]*target
        else:expanded[1]=expanded[0]/target
        downsampling=expanded.max()/384/2;sigma=(downsampling-1)/2 if downsampling>1.1 else 0.0
        margin=int(np.ceil(4*sigma))+3
        rect=np.r_[np.floor(c-expanded/2).astype(int)-margin,np.ceil(c+expanded/2).astype(int)+margin]
        rows.append(dict(name=name,box_xyxy=b.tolist(),center_xy=c.tolist(),expanded_size_wh=expanded.tolist(),
                         gaussian_sigma=float(sigma),filter_margin_px=margin,sampling_rect_xyxy=rect.tolist()))
    return rows


def seven_box_support(native_box,coverage_region,full_size_wh=(5760,2880)):
    roi=np.asarray(coverage_region,int);full=np.asarray(full_size_wh,int);rows=[]
    for row in fixed_seven_boxes(native_box):
        rect=np.asarray(row['sampling_rect_xyxy']);inside_full=bool(np.all(rect[:2]>=0)&np.all(rect[2:]<=full))
        supported=inside_full and bool(np.all(rect[:2]>=roi[:2])&np.all(rect[2:]<=roi[2:]))
        rows.append(dict(row,supported=supported,rejection_reason=None if supported else ('outside_full_erp' if not inside_full else 'crop_filter_support_outside_roi')))
    return rows


def policy_record(sidecar_policy='seven_box_probe_v1'):
    if sidecar_policy not in ('seven_box_probe_v1','mask_native_v2'):raise ValueError('unknown_mamma_sidecar_policy')
    record=dict(schema='mamma_512_label_policy_v1',visibility_policy=POLICY,official_compatibility='compatible_definition_pending',
        positive='body BVH first hit within 3mm; rendered same-person surface within 15mm; complete 3x3 evidence',
        negative='other registered person or scene is >4cm nearer with center and at least 5/9 agreeing 3x3 rays',
        unknown='own-instance foreground, contour/ID mixture, clothing or groom ambiguity, unresolved body ray, or missing coverage',
        ray_quantity='Euclidean camera ray range metres, not camera Z',neighborhood='3x3 evidence plus historical 5x5 surface match',
        contact=dict(known=False,valid=False,reason='out_of_scope'),floor_contact=dict(known=False,valid=False,reason='out_of_scope'),
        crop_policy=dict(variants=['native','center_x_minus','center_x_plus','center_y_minus','center_y_plus','scale_0p9','scale_1p1'],
            center_delta_fraction=.03,scale_factors=[.9,1.1],expansion=1.2,target_wh=[384,512],
            gaussian='sigma=(max(expanded_wh)/384/2-1)/2 when factor>1.1 else 0',filter_margin='ceil(4*sigma)+3'))
    if sidecar_policy=='mask_native_v2':
        record['sidecar_policy']=sidecar_policy
        record['crop_policy']=dict(version='mask_bbox_inclusive_v1',box_source='stored_target_person_mask',
            expansion=1.2,target_wh=[384,512],filter_margin='ceil(4*sigma)+3')
    return record
