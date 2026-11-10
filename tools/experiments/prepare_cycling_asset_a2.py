"""A2 frozen cycling delivery and bounded BEDLAM offset-transfer experiment.

No scene generator, pose fitting, source mutation or implicit resampling.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import csv
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
OUT = ROOT/'outputs/experiments/continuous_cycling_asset_a2_20261009'
BODY = ROOT/'outputs/experiments/apd_6526_smplx_pilot_20261008'
DRESSED = ROOT/'outputs/experiments/apd_6526_dressed_pilot_20261008'
FLICKER = ROOT/'outputs/experiments/apd_groom_flicker_20261009'
REF = ROOT/'outputs/baselines/p0_dressed_dataset/geometry'
PROBE = [0, 6, 16, 38, 75]
KEYS = ['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']

def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def stats(paths): return {str(p): dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths)) if p.is_file()}
def link(p, source): p.symlink_to(Path(source).resolve(strict=True))

def freeze():
    OUT.mkdir(exist_ok=False)
    recipe=read(DRESSED/'frozen_recipe.json');char=read(recipe['character_manifest'])
    paths=[]
    for folder in [ROOT/'src',ROOT/'tests',ROOT/'docs',BODY,DRESSED,FLICKER,Path(recipe['character_manifest']).parent,REF]:
        paths += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    paths += list((ROOT/'tools').rglob('*.py'))
    for k in ['geometry','clothing_geometry','groom_geometry']:paths+=list(Path(char['reference'][k]).glob('*'))
    write(OUT/'protected_before.json',stats(paths))
    (OUT/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    (OUT/'time_mapping.csv').write_bytes((DRESSED/'time_mapping.csv').read_bytes())
    write(OUT/'frozen_recipe.json',dict(integration_only=True,date='2026-10-09',body_authority=recipe['authoritative_body'],
        character='cycling_M01_v2',frame_count=76,source_frames_half_open=[30,106],asset_frames_half_open=[0,76],
        accepted_user_decision='User accepted 64-sample dressed playback, bright hair top, fast left arm and minor intersections; no further appearance changes',
        accepted_preview=str(FLICKER/'full_preview/frames/index.html'),body_updates=0,groom_updates=0,
        time_policy='original observations, no resampling, speed change or looping',
        bedlam_candidate='rp_aaron_posed_009/1095',bedlam_reference=dict(cached_frame=0,clothing_frame=0,motion_local_frame=0,source_frame_30fps=101),
        bedlam_reference_reason='first already paired body/clothing cache frame; selected before target fit, no best-frame search',
        probe_asset_frames=PROBE,probe_source_frames=[x+30 for x in PROBE],
        adaptation='source posed body surface barycentric binding plus inverse blended joint rotation of cloth offsets; target shape body anchor and target blended rotation; approximate non-rest reference',
        limits=dict(candidates_max=3,selected_outfits_max=1,initial_bindings=1,local_corrections_max=1,preview_seconds_max=1800,ds_seconds_max=900,ds_times_max=3),
        compute=dict(cpu_threads=2,cpu_budget_seconds=600,preview_resolution_each=[512,512],studio_samples=64,ds_sampling='A1 recipe only; no DS job until native time/scene support exists'),
        decision_gates='preserve failures; inspect five poses before full candidate. Full accepted studio is referenced, never rerendered',
        lifecycle='all dependencies are durable source/accepted outputs; no full groom copies; no cleanup of old artifacts'))
    write(OUT/'resource_before.json',dict(gpu_query_scope='host query after sandbox NVIDIA driver inaccessible',
        gpus=[dict(index=0,uuid='GPU-d481acc7-106f-29e5-1298-8c0d398261b2',used_mib=18,free_mib=24144,utilization_percent=0),dict(index=1,uuid='GPU-6f046d56-acbe-d49b-774a-9d463e768cac',used_mib=407,free_mib=23754,utilization_percent=1)],
        existing_gpu_process=dict(pid=43491,gpu_index=1,used_mib=118,command='Blender UI; not touched'),disk_available='repo approximately12TiB; /tmp333GiB',
        current_phase='CPU only, CUDA hidden',render_requires_fresh_gpu_check=True))
    print('FROZEN',OUT)

def accepted():
    import torch
    from smplx.lbs import batch_rodrigues
    from densekp_retrain.body.geometry import load_smplx_model,compute_densekp_3d
    from densekp_retrain.body.sequence_binding import bound_garment,head_transform
    began=time.monotonic();torch.set_num_threads(2)
    recipe=read(DRESSED/'frozen_recipe.json');m=read(recipe['character_manifest']);c=Path(recipe['source_clothing']);g=Path(recipe['source_groom'])
    p=dict(np.load(recipe['authoritative_body']));v=np.load(recipe['body_cache'],mmap_mode='r');f=np.load(BODY/'faces.npy');R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    d=np.load(BODY/'fit_attempt_2/densekp_3d.npy');cl=np.load(DRESSED/'full/clothing_vertices.npy');H=np.load(DRESSED/'full/head_transforms.npy')
    ids=np.load(c/'body_barycentric_indices.npy');w=np.load(c/'body_barycentric_weights.npy');hp=np.load(g/'points.npy',mmap_mode='r')
    active=np.load(g/'active_curve_mask.npy');off=np.load(g/'curve_offsets.npy');assert len(active)==63656 and active.sum()==63652 and np.diff(off)[active].sum()==1591300
    assert np.array_equal(p['source_frame_ids'],np.arange(30,106)) and np.array_equal(p['betas'],np.array(m['betas'],np.float32))
    assert np.all(p['transl']==p['transl'][0]);rows=list(csv.DictReader((OUT/'time_mapping.csv').open()))
    model=load_smplx_model(Path(m['model_dir']),gender='male',num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    np.testing.assert_array_equal(f,model.faces)
    # Arbitrary fixed CPU algebra test placement, NOT a generated road scene.
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float);yaw=.23;Z=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    M=np.eye(4);M[:3,:3]=Z@C;M[:3,3]=[1.25,-.8,1.1]
    records=[]
    with torch.no_grad():
        rest=model(betas=torch.tensor(p['betas'][None]),transl=torch.tensor(p['transl'][:1]));restj=rest.joints[0].numpy()
        for i in range(76):
            assert int(rows[i]['asset_frame'])==i and int(rows[i]['apd_source_frame'])==int(p['source_frame_ids'][i]) and int(rows[i]['timestamp_original'])==int(p['timestamps'][i])
            assert abs(float(rows[i]['clip_seconds'])-p['clip_seconds'][i])<1e-12
            kw={k:torch.tensor(p[k][i:i+1]) for k in KEYS};kw['betas']=torch.tensor(p['betas'][None]);ev=model(**kw);vv=ev.vertices[0].numpy();j=ev.joints[0].numpy()
            cc=bound_garment(vv,f,ids,w,offset_m=.004)
            local=batch_rodrigues(torch.tensor(np.r_[p['global_orient'][i],p['body_pose'][i]].reshape(-1,3))).numpy()
            hh=head_transform(local,model.parents,j[15],restj[15]);hair=hp@hh[:3,:3].T+hh[:3,3]
            assert np.isfinite(vv).all() and np.isfinite(cc).all() and np.isfinite(hair).all()
            dd=compute_densekp_3d(R,vv[None])[0]
            bodyerr=float(abs(vv-v[i]).max());derr=float(abs(dd-d[i]).max());cerr=float(abs(cc-cl[i]).max());herr=float(abs(hh-H[i]).max())
            assert bodyerr<2e-5 and derr<2e-6 and cerr<2e-6 and herr<2e-6
            world=v[i]@M[:3,:3].T+M[:3,3];wd=compute_densekp_3d(R,world[None])[0]
            we=float(abs(wd-(d[i]@M[:3,:3].T+M[:3,3])).max());assert we<2e-6
            sample=hp[::1000];sequential=(sample@H[i,:3,:3].T+H[i,:3,3])@M[:3,:3].T+M[:3,3];mh=M@H[i]
            he=float(abs(sequential-(sample@mh[:3,:3].T+mh[:3,3])).max());assert he<1e-12
            records.append(dict(asset_frame=i,source_frame=i+30,timestamp_original=int(p['timestamps'][i]),time_tick=int(p['timestamps'][i]-p['timestamps'][0]),body_error_m=bodyerr,dense_error_m=derr,clothing_error_m=cerr,head_matrix_error=herr,world_R_error_m=we,scene_head_composition_error_m=he))
    if '--check-only' in sys.argv:
        result=dict(status='passed',frames=76,records=records,elapsed_seconds=time.monotonic()-began,road_pass_consumed=False)
        if args.validation_output:
            assert not args.validation_output.exists(), 'Never overwrite a validation record'
            write(args.validation_output,result)
        print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2));return
    folder=OUT/'accepted';folder.mkdir(exist_ok=False);geom=folder/'geometry';geom.mkdir()
    for name,source in dict(vertices=BODY/'fit_attempt_2/vertices.npy',densekp_3d=BODY/'fit_attempt_2/densekp_3d.npy',joints=BODY/'fit_attempt_2/joints.npy',faces=BODY/'faces.npy',clothing_vertices=DRESSED/'full/clothing_vertices.npy',head_transforms=DRESSED/'full/head_transforms.npy').items():link(geom/(name+'.npy'),source)
    np.save(geom/'source_frames.npy',p['source_frame_ids']);np.save(geom/'asset_frames.npy',np.arange(76));np.save(folder/'source_timestamps.npy',p['timestamps'])
    write(folder/'timeline.json',[dict(asset_frame=i,source_frame=i+30,source_time_tick=int(t),episode_time_tick=int(t-p['timestamps'][0]),time_base=[1,1000000]) for i,t in enumerate(p['timestamps'])])
    validation=dict(status='passed',frames=76,body_max_error_m=max(x['body_error_m'] for x in records),R138_max_error_m=max(x['dense_error_m'] for x in records),clothing_max_error_m=max(x['clothing_error_m'] for x in records),head_max_error=max(x['head_matrix_error'] for x in records),scene_R_error_m=max(x['world_R_error_m'] for x in records),fixed_shape=True,fixed_topology=True,finite_all_frames=True,no_min_y_grounding=True,head_applied_once=True,scene_algebra_test_matrix=M.tolist(),scene_algebra_test_is_road_pass=False,public_helpers_actually_consumed=['bound_garment','head_transform'],road_pass_consumed=False,elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(folder/'validation.json',validation);write(folder/'per_frame.json',records)
    manifest=dict(schema='densekp_dynamic_asset_a2_extension_proposal_v1',core_field_compatibility='common names from densekp_dynamic_asset_v1; temporal/root/material extensions require A1 implementation',
        asset_id='cycling_M01_v2_APD6526_30_106_simple_v1',character_manifest=recipe['character_manifest'],character_family='cycling_M01',geometry=str(geom),clothing_reference=str(c),groom_reference=str(g),parameters=recipe['authoritative_body'],
        source_ref=str(BODY/'delivery_manifest.json'),motion_source_id='APD/5724486/6526/source30_106',source_actor_identity='unknown',source_group='APD_5724486_v1_identity_session_unknown',
        model_reference=str(BODY/'model_reference.json'),fixed_target_betas=p['betas'].tolist(),source_betas_used=False,model_units='metres',topology=dict(vertices=10475,faces=20908,vertex_order='unchanged local SMPL-X male model'),
        available_source_frames=[30,105],source_start_reference=30,available_asset_frames_half_open=[0,76],nominal_source_rate=[25,1],nominal_rate_is_exact_clock=False,
        time_mapping=str(OUT/'time_mapping.csv'),timeline=str(folder/'timeline.json'),source_timestamps=str(folder/'source_timestamps.npy'),time_base=[1,1000000],time_unit_status='microseconds inferred from APD25Hz; not official unit confirmation',
        root_mode='fixed_pelvis_model_origin_scene_trajectory_owned_by_scene',root_baked='fixed transl=-shaped_rest_pelvis; fitted global orientation already included in model vertices and head matrix',source_root_removed=True,removed_source_motion_ref=str(BODY/'alignment.json'),
        scene_transform_contract='body/cloth/138: scene_from_model[t]; groom: scene_from_model[t] @ head[t]; axis conversion exactly once inside scene_from_model',support_policy='fixed_scene_placement_height_no_pedestrian_foot_contact',
        clothing_offset_m=.004,head_joint=15,garment=dict(representation='fixed_body_barycentric_normal_offset',faces=str(c/'clothing_faces.npy'),binding_indices=str(c/'body_barycentric_indices.npy'),binding_weights=str(c/'body_barycentric_weights.npy'),material_kind='GarmentRest_continuous_shell',uv=None,material_reference=str(c/'garment_material.json'),rest_coordinates=str(c/'clothing_rest_vertices.npy')),
        groom=dict(representation='fixed_reference_curves_rigid_head',active_mask=str(g/'active_curve_mask.npy'),curve_offsets=str(g/'curve_offsets.npy'),points=str(g/'points.npy'),radius=str(g/'radius.npy'),head_transforms=str(geom/'head_transforms.npy'),dynamic_physics=False),
        available_combinations_verified=True,binding_reference_reproduced=True,dynamic_numeric_checked_frames=76,human_dynamic_review='accepted_by_user_2026-10-09_for_this_example_only',acceptance_scope='body + simple clothes + groom studio playback; not real reconstruction or dataset',
        known_limits=read(DRESSED/'delivery_manifest.json')['local_issues'][:3]+['accepted left-arm fast swing','accepted bright top hair; no new appearance work'],
        preview_html=str(FLICKER/'full_preview/frames/index.html'),preview_mp4=str(FLICKER/'full_preview/frames/lossless.mp4'),preview_samples=64,ds_sampling_decided=False,
        integration_only=True,independent_validation=False,split_group_policy='APD source group and cycling_M01 family propagate through neighbours/resampling/loops/outfits/seeds',
        provenance_refs=[str(BODY/'ATTRIBUTION.md'),str(DRESSED/'ATTRIBUTION.md'),recipe['character_manifest']],numeric_validation=str(folder/'validation.json'),public_scene_consumption='blocked_pending_A1_time_root_support',extensions_effective_in_A1=False)
    write(folder/'manifest.json',manifest);print(json.dumps(validation,indent=2))

def bedlam():
    import torch
    from smplx.lbs import batch_rodrigues
    from densekp_retrain.body.geometry import load_smplx_model
    from densekp_retrain.clothing.binding import bind_new_points,closest_surface
    began=time.monotonic();torch.set_num_threads(2)
    out=OUT/'bedlam_attempt_1'
    resume='--resume-metadata' in sys.argv
    if resume:
        assert (out/'binding.npz').is_file() and not (out/'manifest.json').exists()
    else:out.mkdir(exist_ok=False)
    recipe=read(DRESSED/'frozen_recipe.json');m=read(recipe['character_manifest']);p=dict(np.load(recipe['authoritative_body']))
    source=read(REF/'clothing_manifest.json');srcbody=np.load(REF/'vertices.npy')[0];srccloth=np.load(REF/'clothing_vertices.npy')[0];faces=np.load(REF/'faces.npy');cf=np.load(REF/'clothing_faces.npy');sp=dict(np.load(REF/'motion_params.npz'))
    model=load_smplx_model(Path(m['model_dir']),gender='male',num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    def rotations(par,i):
        aa=np.concatenate([par[k][i].ravel() for k in ['global_orient','body_pose','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose']])
        q=batch_rodrigues(torch.tensor(aa.reshape(-1,3),dtype=torch.float32)).numpy()
        for j in range(1,55):q[j]=q[int(model.parents[j])]@q[j]
        return q
    # Verify corresponding source parameters under the available same-topology model.
    with torch.no_grad():
        kw={k:torch.tensor(sp[k][:1]) for k in KEYS if k in sp};sb=np.pad(sp['betas'],(0,30-len(sp['betas'])));kw['betas']=torch.tensor(sb[None],dtype=torch.float32)
        sv=model(**kw).vertices[0].numpy()
    source_err=float(abs(sv-srcbody).max());assert source_err<2e-5
    np.testing.assert_array_equal(faces,model.faces)
    raw=np.load(source['clothing_source'],allow_pickle=True)
    np.testing.assert_array_equal(raw['vertices_seq'][0].astype(np.float32),srccloth);np.testing.assert_array_equal(raw['faces'].astype(np.int32),cf)
    metadata=dict(raw['metadata'].item());assert metadata==dict(fps=6,source_fps=30,start_frame=101,end_frame=292)
    if resume:
        saved=dict(np.load(out/'binding.npz'));ids=saved['body_vertex_indices'];w=saved['barycentric_weights']
    else:ids,w=bind_new_points(srccloth,srcbody,faces)
    anchor=(srcbody[ids]*w[...,None]).sum(1)
    skin=(model.lbs_weights.numpy()[ids]*w[...,None]).sum(1);source_rot=rotations(sp,0);B=np.einsum('nj,jab->nab',skin,source_rot)
    cond=np.linalg.cond(B);assert cond.max()<100
    delta=saved['unposed_offset'] if resume else np.linalg.solve(B,(srccloth-anchor)[...,None])[...,0]
    reconstruction=anchor+np.einsum('nij,nj->ni',B,delta);err=float(abs(reconstruction-srccloth).max());assert err<2e-6
    if not resume:
        np.savez(out/'binding.npz',body_vertex_indices=ids,barycentric_weights=w,skin_weights=skin,unposed_offset=delta)
        for n in ['clothing_faces','clothing_vertices_uv','clothing_faces_uv']:link(out/(n+'.npy'),REF/(n+'.npy'))
    tex=Path(source['clothing_source']).parents[2]/'1095/textures/texture_01'
    textures={k:str(tex/f'texture_01_{k}_1001.png') for k in ['diffuse','normal']}
    assert all(Path(x).is_file() for x in textures.values())
    manifest=dict(integration_only=True,status='probe_numeric_visual_pending',candidate_id='BEDLAM_rp_aaron_posed_009_1095',
        source_clothing=source['clothing_source'],source_body_parameters=str(REF/'motion_params.npz'),source_body_cache=str(REF/'vertices.npy'),source_body_original=read(REF/'canonical_motion.json')['source'],source_reference=read(OUT/'frozen_recipe.json')['bedlam_reference'],
        source_body_reproduction_error_m=source_err,source_clothing_reproduction_error_m=err,reference_is_official_rest=False,
        approximation='inverse blended source skeletal rotations applied to offsets from posed corresponding body; preserves source folds and offsets, no cloth physics or official rest recovery',
        binding=str(out/'binding.npz'),recipe='target = sum(w*target_body[ids]) + sum(skin_w*global_joint_rotation_target) @ unposed_offset',target_shape='fixed cycling_M01_v2, unchanged',
        topology=dict(vertices=len(srccloth),faces=len(cf),fixed=True),uv=dict(vertices=str(out/'clothing_vertices_uv.npy'),faces=str(out/'clothing_faces_uv.npy'),domain='face_corner',vflip=False),
        material=dict(kind='uv_textured_principled',textures=textures,diffuse_colorspace='sRGB',normal_colorspace='Non-Color',normal_strength=1,roughness=.58,metallic=0),
        license=dict(url='https://bedlam.is.tuebingen.mpg.de/license.html',checked='2026-10-09',scope='existing local research resources; internal non-commercial adaptation only; no redistribution or commercial authorization inferred',modification_clause='license restricts modification/third-party availability; internal derivative here not a redistribution permission',citation='Black, Patel, Tesch and Yang. BEDLAM. CVPR2023'),
        source_group='APD_5724486_v1_identity_session_unknown',character_family='cycling_M01',additional_derivation_source='BEDLAM/rp_aaron_posed_009/1095',independent_validation=False,
        initial_bindings=1,local_corrections=0,probe_asset_frames=PROBE,condition_number_max=float(cond.max()))
    write(out/'manifest.json',manifest)
    v=np.load(recipe['body_cache'],mmap_mode='r');records=[];cloth=[]
    edges=np.unique(np.sort(np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]),axis=1),axis=0);original_length=np.linalg.norm(srccloth[edges[:,0]]-srccloth[edges[:,1]],axis=1)
    for i in PROBE:
        Q=rotations(p,i);target=(v[i][ids]*w[...,None]).sum(1)+np.einsum('nj,jab,nb->na',skin,Q,delta);target=target.astype(np.float32);assert np.isfinite(target).all()
        length=np.linalg.norm(target[edges[:,0]]-target[edges[:,1]],axis=1);ratio=length/np.maximum(original_length,1e-8)
        _,_,signed,_=closest_surface(target,v[i],faces)
        records.append(dict(asset_frame=i,source_frame=i+30,max_edge_m=float(length.max()),edge_ratio_p99=float(np.quantile(ratio,.99)),edge_ratio_max=float(ratio.max()),edge_over5_fraction=float((ratio>5).mean()),signed_distance_min_m=float(signed.min()),signed_below_minus2cm_fraction=float((signed<-.02).mean()),signed_distance_diagnostic='approximate nearest triangle normal, not exhaustive collision'))
        cloth.append(target)
    np.save(out/'probe_clothing_vertices.npy',np.array(cloth));np.save(out/'probe_asset_frames.npy',PROBE)
    write(out/'probe_validation.json',dict(status='finite_topology_time_pass_visual_review_required',records=records,elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(records,indent=2))

def correction():
    """One fixed local binding-field correction, never a body or topology edit."""
    import torch
    from scipy import sparse
    from densekp_retrain.body.geometry import load_smplx_model
    torch.set_num_threads(2);start=time.monotonic()
    previous=OUT/'bedlam_attempt_1';out=OUT/'bedlam_attempt_2';out.mkdir(exist_ok=False)
    b=dict(np.load(previous/'binding.npz'));ids=b['body_vertex_indices'];w=b['barycentric_weights'];skin=b['skin_weights'];offset=b['unposed_offset']
    cm=read(previous/'manifest.json');a=read(OUT/'accepted/manifest.json');m=read(a['character_manifest']);p=dict(np.load(a['parameters']))
    cf=np.load(previous/'clothing_faces.npy');src=np.load(REF/'clothing_vertices.npy')[0];raw=np.load(previous/'probe_clothing_vertices.npy')
    edges=np.unique(np.sort(np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]),axis=1),axis=0)
    length=np.linalg.norm(src[edges[:,0]]-src[edges[:,1]],axis=1)
    ratios=np.linalg.norm(raw[:,edges[:,0]]-raw[:,edges[:,1]],axis=2)/np.maximum(length,1e-8)
    selected=np.unique(edges[(ratios>4).any(0)])
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(src),len(src))).tocsr()
    mask=np.zeros(len(src),bool);mask[selected]=True
    for _ in range(2):mask|=(adjacency@mask)>0
    weights=mask.astype(float)*.5;L=sparse.diags(1-weights)+sparse.diags(weights/np.maximum(np.asarray(adjacency.sum(1)).ravel(),1))@adjacency
    assert np.allclose(np.asarray(L.sum(1)).ravel(),1)
    sparse.save_npz(out/'local_binding_operator.npz',L)
    model=load_smplx_model(Path(m['model_dir']),gender='male',num_betas=30,batch_size=1).eval()
    with torch.no_grad():rest=model(betas=torch.tensor(p['betas'][None]),transl=torch.tensor(p['transl'][:1])).vertices[0].numpy()
    cr=(rest[ids]*w[...,None]).sum(1)+offset;np.save(out/'approximate_target_rest.npy',cr)
    def correct(x):
        change=x-cr
        for _ in range(12):change=L@change
        return (cr+change).astype(np.float32)
    corrected=np.array([correct(x) for x in raw]);np.save(out/'probe_clothing_vertices.npy',corrected);np.save(out/'probe_asset_frames.npy',PROBE)
    for n in ['binding.npz','clothing_faces.npy','clothing_vertices_uv.npy','clothing_faces_uv.npy']:link(out/n,previous/n)
    cm.update(status='corrected_probe_visual_review_pending',local_corrections=1,initial_binding_reference=str(previous/'binding.npz'),binding=str(out/'binding.npz'),local_correction=dict(reason='visible jagged upper sleeve and sparse edges stretching >4x at the five frozen probes; fixed local binding-field smoothing',selection='union endpoints of >4x source edge length across fixed5 probes, expanded2 adjacency rings',selected_vertices=int(mask.sum()),total_vertices=len(src),operator=str(out/'local_binding_operator.npz'),iterations=12,formula='target_rest + L^12 @ (raw_posed_cloth-target_rest)',target_rest=str(out/'approximate_target_rest.npy'),body_unchanged=True,topology_unchanged=True,no_temporal_filter=True),uv={**cm['uv'],'vertices':str(out/'clothing_vertices_uv.npy'),'faces':str(out/'clothing_faces_uv.npy')})
    write(out/'manifest.json',cm)
    records=[]
    for i,c in zip(PROBE,corrected):
        ratio=np.linalg.norm(c[edges[:,0]]-c[edges[:,1]],axis=1)/np.maximum(length,1e-8)
        records.append(dict(asset_frame=i,source_frame=i+30,edge_ratio_max=float(ratio.max()),edge_ratio_p99=float(np.quantile(ratio,.99)),max_correction_m=float(np.linalg.norm(c-raw[PROBE.index(i)],axis=1).max())))
    write(out/'probe_validation.json',dict(status='finite_fixed_binding_topology_pending_visual',selected_vertices=int(mask.sum()),records=records,elapsed_seconds=time.monotonic()-start))
    print(json.dumps(records,indent=2))

def full():
    import torch
    from scipy import sparse
    from smplx.lbs import batch_rodrigues
    from densekp_retrain.body.geometry import load_smplx_model
    from densekp_retrain.clothing.binding import closest_surface
    torch.set_num_threads(2);start=time.monotonic();out=OUT/'bedlam_attempt_2';assert read(out/'probe_decision.json')['proceed_full']
    b=dict(np.load(out/'binding.npz'));ids=b['body_vertex_indices'];w=b['barycentric_weights'];skin=b['skin_weights'];delta=b['unposed_offset'];L=sparse.load_npz(out/'local_binding_operator.npz');cr=np.load(out/'approximate_target_rest.npy')
    a=read(OUT/'accepted/manifest.json');m=read(a['character_manifest']);p=dict(np.load(a['parameters']));v=np.load(Path(a['geometry'])/'vertices.npy');faces=np.load(Path(a['geometry'])/'faces.npy');cf=np.load(out/'clothing_faces.npy')
    model=load_smplx_model(Path(m['model_dir']),gender='male',num_betas=30,batch_size=1).eval();clothes=[];records=[]
    edges=np.unique(np.sort(np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]),axis=1),axis=0);src=np.load(REF/'clothing_vertices.npy')[0];length=np.linalg.norm(src[edges[:,0]]-src[edges[:,1]],axis=1)
    for i in range(76):
        aa=np.concatenate([p[k][i].ravel() for k in ['global_orient','body_pose','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose']]);Q=batch_rodrigues(torch.tensor(aa.reshape(-1,3),dtype=torch.float32)).numpy()
        for j in range(1,55):Q[j]=Q[int(model.parents[j])]@Q[j]
        raw=(v[i][ids]*w[...,None]).sum(1)+np.einsum('nj,jab,nb->na',skin,Q,delta)
        # Float32 raw matches the frozen probe before local correction.
        change=raw.astype(np.float32)-cr
        for _ in range(12):change=L@change
        target=(cr+change).astype(np.float32);assert np.isfinite(target).all();clothes.append(target)
        _,_,signed,_=closest_surface(target,v[i],faces);ratio=np.linalg.norm(target[edges[:,0]]-target[edges[:,1]],axis=1)/np.maximum(length,1e-8)
        records.append(dict(asset_frame=i,source_frame=i+30,edge_ratio_max=float(ratio.max()),signed_min_m=float(signed.min()),below_minus2cm_fraction=float((signed<-.02).mean()),adjacent_vertex_displacement_max_m=None if i==0 else float(np.linalg.norm(target-clothes[i-1],axis=1).max())))
    clothes=np.array(clothes);np.testing.assert_array_equal(clothes[PROBE],np.load(out/'probe_clothing_vertices.npy'))
    np.save(out/'full_clothing_vertices.npy',clothes);np.save(out/'full_asset_frames.npy',np.arange(76))
    write(out/'full_validation.json',dict(status='numeric_pass_candidate_visual_review_pending',frames=76,fixed_topology=True,fixed_binding=True,body_parameters_unchanged=True,groom_unchanged=True,R138_reference=str(OUT/'accepted/validation.json'),records=records,elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('FULL CLOTHING',time.monotonic()-start)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','accepted','bedlam','correction','full']);ap.add_argument('--resume-metadata',action='store_true');ap.add_argument('--check-only',action='store_true');ap.add_argument('--validation-output',type=Path);args=ap.parse_args();globals()[args.stage]()
