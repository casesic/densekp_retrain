"""Freeze conservative identity/source groups before producing new combinations."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LIB=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930'


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--exclude-family',choices=['F04'],required=True,
                   help='Explicit scope change; never silently discard a family')
    p.add_argument('--freeze',action='store_true',help='Only after resolving the scope decision')
    args=p.parse_args();root=args.root.resolve()
    filename='split_v2.json' if args.freeze else 'split_proposal_v2.json'
    if (root/filename).exists():raise FileExistsError('Do not overwrite split version')
    # Conservative possible face families, including superseded textures. These
    # are source-risk groups, not assertions about ethnicity or scan identities.
    audit=json.loads((root/'identity_source_audit/report.json').read_text())
    if audit['status']!='preliminary_split_rejected_before_generation':raise ValueError('Missing updated source audit')
    groups={'face_A_f':['F01','F03','F05'], 'face_B_f':['F02'],
            'face_A_m':['M01','M04','M05'], 'face_B_m':['M02','M03']}
    assignment={short:('train' if '_A_' in group else 'val')
                for group,members in groups.items() for short in members}
    report=json.loads((LIB/'report.json').read_text());characters=[];excluded=[]
    for c in report['characters']:
        path=Path(c['manifest']);m=json.loads(path.read_text());short=m['character_id'].split('_')[1]
        if short==args.exclude_family:
            excluded.append(dict(**c,identity_family_id=m['identity_family_id'],all_versions_excluded=True,
                reason='Historical textures bridge female possible-source groups; retained in original integration library'))
            continue
        chain=[];textures=[];seen=set();current=path
        while current:
            if str(current) in seen:raise ValueError('Cyclic ancestry')
            seen.add(str(current));v=json.loads(current.read_text())
            texture=Path(v['appearance']['body_texture']).name
            textures.append(texture)
            chain.append(dict(asset_id=v['character_id'],manifest=str(current),
                texture=texture,shape_source=v.get('shape_source',v.get('shape_fit')),
                betas=v['betas']))
            current=Path(v['parent_manifest']) if v.get('parent_manifest') else None
        characters.append(dict(**c,identity_family_id=m['identity_family_id'],gender=m['gender'],
            partition=assignment[short],face_source_group=next(g for g,cs in groups.items() if short in cs),
            ancestry=chain,all_texture_members=sorted(set(textures)),
            shape_source_group=m['identity_family_id'] if short!='M01' else 'BEDLAM_shape_rp_aaron_posed_009',
            shape_note='synthetic preset/whole-model fit; shared official zero/template is not a scanned-person source',
            current_betas=m['betas']))
    old=ROOT/'outputs/experiments/mixed_riders_pedestrians_20260930'
    acq=json.loads((root/'sources/acquisition.json').read_text())
    if json.loads((root/'sources/rig_compatibility.json').read_text())['status']!='rotation_conventions_match':
        raise ValueError('New source rotation convention not checked')
    z=np.load(acq['local_file'])
    if z['poses'].shape!=(127,165):raise ValueError('Unexpected source')
    source_groups={
      'train':dict(cycling='coco_292226_person_1330527',pedestrian='AMASS_KIT_314_bend_left01',
                   pedestrian_person='KIT/314',source_file=str(ROOT/'third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz'),
                   standing_frame=0,walking_frames=[30,40,50,60]),
      'val':dict(cycling='coco_306800_person_225160',pedestrian='AMASS_KIT_425_walking_medium05',
                 pedestrian_person='KIT/425',source_file=acq['local_file'],standing_frame=120,walking_frames=[20,40])}
    pose_review=dict(status='accepted_for_target_geometry_generation',source=acq['source'],
        source_group=source_groups['val']['pedestrian'],source_file=acq['local_file'],
        body_model='source neutral locked-head SMPL-X 16 betas; target standard gendered SMPL-X 30 betas unchanged',
        pose_convention='55 axis-angle joints, SMPL-X Y-up, flat hands; official addon same joint order/rotation path',
        compatibility=str(root/'sources/rig_compatibility.json'),
        shape_transfer='none; no source betas copied or zero-padded',
        selected=[dict(frame=20,action='walking',evidence='long stride, one supporting leg, opposite arm swing'),
                  dict(frame=40,action='walking',evidence='opposite leg forward after intervening swing frames'),
                  dict(frame=120,action='standing',evidence='end-of-walk upright rest, arms down, legs nearly aligned')],
        review_images=[str(root/'pose_audit'/f'{s}_source_review.jpg') for s in ['F02','M02']],
        temporal_context='entire 127-frame sequence retained; frames 0..126 reviewed every 10 plus last',
        local_fps=30,original_fps=100,original_range=[0,381],
        sampling_mapping=dict(local_frames=list(range(127)),estimated_original_frames=list(range(0,381,3)),
            status='inferred from official interval/count; original per-frame map not provided; not exact timestamp proof'),
        translation='source retained unchanged in NPZ; each target pelvis re-centered and grounded independently',
        source_independence='KIT/425 differs from KIT/314; BEDLAM body us_2495_L is a retarget identity, not original actor',
        limitation='one new clip; no independent test; target cloth/ground/appearance still require checks')
    if not (root/'pose_selection.json').exists():write(root/'pose_selection.json',pose_review)
    checks={}
    for key in ['identity_family_id','face_source_group','shape_source_group']:
        a={c[key] for c in characters if c['partition']=='train'};b={c[key] for c in characters if c['partition']=='val'}
        checks[key]=sorted(a&b)
    for key in ['all_texture_members','ancestry']:
        sets=[]
        for split in ['train','val']:
            sets.append({(v['asset_id'] if isinstance(v,dict) else v) for c in characters if c['partition']==split for v in c[key]})
        checks[key]=sorted(sets[0]&sets[1])
    checks['pose_source_groups']=sorted(set(source_groups['train'][k] for k in ['cycling','pedestrian']) & set(source_groups['val'][k] for k in ['cycling','pedestrian']))
    if any(checks.values()):raise ValueError(checks)
    regression=[]
    for p in [LIB/'candidate_index.json',old/'candidate_index.json',old/'original_baseline/report.json']:
        st=p.stat();regression.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    for a in characters:
        for b in characters:
            if a['partition']!=b['partition'] and a['gender']==b['gender']:
                if np.array_equal(a['current_betas'],b['current_betas']):raise ValueError('Same actual shape across partitions')
    result=dict(status='frozen_before_new_combinations' if args.freeze else 'proposal_awaiting_scope_decision',created_utc=datetime.now(timezone.utc).isoformat(),
        characters=characters,source_groups=source_groups,source_intersections=checks,
        excluded_families=excluded,source_audit=str(root/'identity_source_audit/report.json'),
        conservative_face_groups=groups,
        grouping_evidence=[str(LIB/'identity_head_front.jpg'),str(LIB/'female_face_texture_diagnostic.jpg'),str(LIB/'identity_review.json')],
        grouping_rule='Group plausible shared underlying facial detail across tone/hair variants and all parent revisions. A/B are conservative visual source-risk groups, not proven scan IDs.',
        limitation='Texture pack gives no photographed-person identifiers. Synthetic identity/source-risk isolation is auditable; absolute original face-subject independence is not established.',
        shared_assets_allowed=['official SMPL-X model','generic groom','clothing template','forest/street scenes'],
        regression_only=regression,regression_rows=51,
        scene_policy='All targets and occluders, both views and all derivatives belong to one partition',
        split_use='train/val candidates; not independent test; no training executed')
    write(root/filename,result)
    print({s:[c['character_id'] for c in characters if c['partition']==s] for s in ['train','val']})

if __name__=='__main__':main()
