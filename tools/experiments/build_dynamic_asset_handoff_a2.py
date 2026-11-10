"""Create explicit asset combinations, without changing any authoritative source."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/plans/dynamic_asset_handoff_a2_20261009'
FEMALE=ROOT/'outputs/experiments/cycling_female_asset_a2_20261009'
MALE=ROOT/'outputs/experiments/continuous_cycling_asset_a2_20261009/accepted'
BEDLAM=ROOT/'outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2'
HAIR=ROOT/'outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009'


def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def path(p):return str(Path(p).absolute())


def base_entry(asset_id):
    return dict(schema='densekp_explicit_asset_combination_a2_proposal_v1',asset_id=asset_id,
        integration_only=True,A1_schema_effective=False,public_consumer_tested=False,
        units='metres',coordinates='SMPL-X model Y-up',regressor=path(ROOT/'assets/densekp_regressor_R.npy'),
        densekp_label_ready=True,densekp_status='compatible',scene_observations_produced=False,
        independent_validation=False,split_policy='group shared motion, adjacent segments, resampling, alternate appearance and seeds together; retain original family restrictions',
        source_actor_identity='unknown',persistent_dependencies_only=True)


def cycling(female):
    old=read(MALE/'manifest.json');folder=FEMALE if female else MALE
    character=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930/characters'/('cycling_F02_v3' if female else 'cycling_M01_v2')/'manifest.json'
    char=read(character);g=folder/'full' if female else folder/'geometry'
    clothes=Path(char['reference']['clothing_geometry']);hair=Path(char['reference']['groom_geometry'])
    params=folder/'parameters.npz' if female else Path(old['parameters'])
    item=base_entry(('cycling_F02_v3' if female else 'cycling_M01_v2')+'_APD6526_30_106_simple_v1')
    item.update(character_family='cycling_F02' if female else 'cycling_M01',character_manifest=path(character),
        authority_manifest=path(FEMALE/'frozen_recipe.json') if female else path(MALE/'manifest.json'),parameters=path(params),
        model=dict(file=char['model_file'],gender=char['gender'],betas=char['betas'],num_betas=30,
                   template='standard SMPL-X same-gender local model; topology unmodified',
                   semantics_evidence=path(FEMALE/'model_semantics.json') if female else old['model_reference']),
        source_group='APD_5724486_v1_identity_session_unknown',motion_source_id=old['motion_source_id'],
        source_parameters=old['parameters'],frame_count=76,timeline=path(MALE/'timeline.json'),time_base=[1,1000000],
        time_unit_status=old['time_unit_status'],time_mapping=old['time_mapping'],
        geometry=dict(vertices=path(g/'vertices.npy'),faces=path(FEMALE/'faces.npy') if female else path(g/'faces.npy'),densekp=path(g/'densekp_3d.npy')),
        garment=dict(representation='simple_fixed_barycentric_normal_offset',vertices=path(g/'clothing_vertices.npy'),
            faces=path(clothes/'clothing_faces.npy'),binding_indices=path(clothes/'body_barycentric_indices.npy'),
            binding_weights=path(clothes/'body_barycentric_weights.npy'),offset_m=.004,
            material_kind='GarmentRest_continuous_shell',materials=path(clothes/'materials.json'),
            material_reference=path(clothes/'garment_material.json'),rest_coordinates=path(clothes/'clothing_rest_vertices.npy'),uv=None),
        groom=dict(representation='fixed_reference_curves_rigid_head',reference=path(hair),points=path(hair/'points.npy'),
            radius=path(hair/'radius.npy'),curve_offsets=path(hair/'curve_offsets.npy'),active_mask=path(hair/'active_curve_mask.npy'),
            head_transforms=path(g/'head_transforms.npy'),reference_space='target female shaped rest pelvis-centered model' if female else 'target male shaped rest pelvis-centered model',
            root_in_head=True,physics=False),
        appearance=dict(kind='existing_character_skin_and_native_eye_shader',character_manifest=path(character)),
        root_mode='fixed_pelvis_scene_owns_progression',root_baked='constant transl=-target rest pelvis and source global_orient already in body/head',
        support_policy='fixed_scene_placement_height_no_pedestrian_foot_contact',
        scene_contract='body/cloth/R: scene_from_model[t]; groom: scene_from_model[t] @ head[asset_frame]; axis conversion included once in scene',
        technical_status='all76_frames_pass',technical_evidence=path(FEMALE/'validation.json') if female else old['numeric_validation'],
        human_acceptance='pending_female_dynamic_review' if female else 'accepted_body_cloth_groom_by_user',
        preview_html=path(FEMALE/'preview/index.html') if female else old['preview_html'],
        preview_mp4=path(FEMALE/'preview/comparison.mp4') if female else old['preview_mp4'],
        known_limits=['same accepted APD rapid left arm and unobserved hand/face constraints','minor local clothing folds/intersections permitted','groom follows head rigidly, no physics','not an independent motion source'],
        provenance_refs=old['provenance_refs']+[path(character)],licenses=['APD CC BY4.0 DOI10.5281/zenodo.5724486','SMPL-X research model license','BEDLAM2 groom license','Meshcapade CC BY-NC4.0 texture version20220207'])
    return item


def bedlam(with_hair):
    source=read((HAIR if with_hair else BEDLAM)/'manifest.json');g=BEDLAM/'geometry'
    item=base_entry('bedlam2_us_2495_L_2200_native_0_91'+('_eyes_bob_v1' if with_hair else '_base'))
    item.update(character_family='BEDLAM2_body_us_2495_L',authority_manifest=path((HAIR if with_hair else BEDLAM)/'manifest.json'),
        parameters=source['parameters'],model=dict(file=source['model'],gender='neutral',betas=source['shape_values'],num_betas=16,
            template='locked-head/no-head-bun neutral 20230302; no shoe/toe edits',semantics_evidence=path(ROOT/'outputs/audits/bedlam_r138_compatibility_a2_20261009/compatibility.json')),
        source_group='KIT_425_walking_medium05',motion_source_id=source['motion_source_id'],frame_count=91,
        timeline=path(OUT/'bedlam_timeline.json'),time_base=[1,30],time_mapping=path(BEDLAM/'time_mapping.csv'),
        time_unit_status='published processed30Hz clock; original100Hz correspondence not claimed',
        geometry=dict(vertices=path(g/'vertices.npy'),faces=path(g/'faces.npy'),densekp=path(g/'densekp_3d_pending.npy')),
        garment=dict(representation='native_simulation_cache',vertices=path(g/'clothing_vertices.npy'),faces=path(g/'clothing_faces.npy'),
            original_npz=source['garment']['original_npz'],uv=path(g/'clothing_vertices_uv.npy'),face_corner_uv=path(g/'clothing_faces_uv.npy'),
            material_kind='independent_face_corner_uv_diffuse_normal',diffuse=source['garment']['diffuse'],normal=source['garment']['normal'],
            uv_flip_v=False,roughness=.58,normal_strength=1,material_scalar_origin='frozen preview recipe'),
        groom=None,appearance=source['body_appearance'],root_mode='native_root_baked_static_scene_placement',
        root_baked='original global_orient/trans already in body and cloth; do not add them again',
        support_policy='scene native walking support evaluation with fixed placement, no per-frame lift',
        scene_contract='one common scene_from_model on body/cloth/R; if groom present scene @ head; source movement already retained',
        technical_status='all91_frames_pass',technical_evidence=path((HAIR/'delivery_validation.json') if with_hair else BEDLAM/'validation.json'),
        human_acceptance='base_body_motion_cloth_accepted_new_eyes_hair_pending' if with_hair else 'accepted_body_motion_cloth_by_user',
        preview_html=path((HAIR if with_hair else BEDLAM)/'preview.html'),preview_mp4=path((HAIR if with_hair else BEDLAM)/'native_walk_two_views.mp4'),
        known_limits=['no shoes or displacement added','preview skin is neutral matte, not original actor texture','R compatibility limited to audited local model/cache'],
        licenses=source['licenses'],official_source_split='training_mapping',historical_project_split='KIT425_validation_source_unchanged')
    if with_hair:
        folder=HAIR/'groom'
        item['groom']=dict(representation='fixed_reference_curves_rigid_head',reference=path(folder),points=path(folder/'points.npy'),
            radius=path(folder/'radius.npy'),curve_offsets=path(folder/'curve_offsets.npy'),active_mask=None,
            head_transforms=path(HAIR/'head_transforms.npy'),reference_space='accepted body model coordinates at asset frame0; includes one static affine fit',
            root_in_head=True,physics=False,original_actor_pairing=False)
        item['appearance']=dict(kind='neutral_matte_plus_native_eye_attribute_shader',appearance_manifest=path(HAIR/'appearance.json'))
    return item


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'bedlam_timeline.json',[dict(asset_frame=i,source_frame=i,source_time_tick=i,episode_time_tick=i,time_base=[1,30],cloth_simulation_frame=i+101) for i in range(91)])
    entries=[cycling(False),cycling(True),bedlam(False),bedlam(True)];records=[]
    for name,item in zip(['cycling_m01','cycling_f02','bedlam_base','bedlam_eyes_hair'],entries):
        target=OUT/(name+'.json');write(target,item)
        records.append(dict(asset_id=item['asset_id'],manifest=path(target),human_acceptance=item['human_acceptance'],densekp_status=item['densekp_status']))
    write(OUT/'index.json',dict(schema='densekp_explicit_asset_catalog_a2_proposal_v1',integration_only=True,
        combination_count=4,character_family_count=3,independent_motion_group_count=2,
        arbitrary_cartesian_product_allowed=False,A1_schema_effective=False,public_consumer_tested=False,assets=records,
        notes=['M01 and F02 share APD motion group; do not put them into independent train/val by character alone',
               'BEDLAM base and eyes/hair version are the same body/motion family, not independent people']))
    write(OUT/'contract_sample.json',dict(status='proposal_only_not_observed_or_packed',
        sample=dict(asset_manifest=records[1]['manifest'],timeline_entry=read(MALE/'timeline.json')[6],
                    scene_from_model='provided by A1 per actual episode time; no road matrix invented here',
                    support_policy=entries[1]['support_policy']),
        geometry_rule='body/cloth/R share one asset frame and scene matrix; groom uses scene @ head for that frame'))
    write(FEMALE/'delivery_manifest.json',entries[1])
    print(path(OUT/'index.json'))


if __name__=='__main__':main()
