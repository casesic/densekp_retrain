"""Fail closed on identity, source, image, occluder or paired-box leakage."""
import argparse
import json
from pathlib import Path


def validate_split(split, indices=None, allow_proposal=False):
    allowed={'frozen_before_new_combinations'}
    if allow_proposal:allowed.add('proposal_awaiting_scope_decision')
    if split['status'] not in allowed:raise ValueError('Split is not approved/frozen')
    audit_path=split.get('source_audit')
    if not audit_path:raise ValueError('Updated facial-source audit is required; old split is rejected')
    audit=json.loads(Path(audit_path).read_text())
    characters={c['character_id']:c for c in split['characters']}
    families={c['identity_family_id']:c['partition'] for c in characters.values()}
    exclusions={c['identity_family_id'] for c in split['excluded_families']}
    if exclusions&families.keys():raise ValueError('Excluded family is still included')
    for members in audit['conservative_possible_source_groups'].values():
        sides={families['cycling_'+s] for s in members if 'cycling_'+s in families}
        if len(sides)>1:raise ValueError('Possible facial source crosses partitions')
    memberships={}
    for key in ['identity_family_id','shape_source_group','face_source_group']:
        seen={}
        for c in characters.values():
            old=seen.setdefault(c[key],c['partition'])
            if old!=c['partition']:raise ValueError(f'{key} crosses partitions')
        memberships[key]={s:sorted(v for v,side in seen.items() if side==s) for s in ['train','val']}
    for key in ['all_texture_members','ancestry']:
        seen={}
        for c in characters.values():
            for value in c.get(key,[]):
                identity=value['asset_id'] if isinstance(value,dict) else value
                if seen.setdefault(identity,c['partition'])!=c['partition']:
                    raise ValueError(f'{key} crosses partitions')
        memberships[key]={s:sorted(v for v,side in seen.items() if side==s) for s in ['train','val']}
    for key in ['cycling','pedestrian','pedestrian_person']:
        if split['source_groups']['train'][key]==split['source_groups']['val'][key]:
            raise ValueError(f'{key} not held out')
        memberships[key]={s:[split['source_groups'][s][key]] for s in ['train','val']}
    count={s:sum(c['partition']==s for c in characters.values()) for s in ['train','val']}
    for side in count:
        if {c['gender'] for c in characters.values() if c['partition']==side}!={'male','female'}:
            raise ValueError('Each partition must include both genders')
    images={};scenes={};paired={};rows_count={s:0 for s in count}
    for side,index in (indices or {}).items():
        if side not in count:raise ValueError('Unknown partition')
        if index['split']!=side or not index.get('frozen_utc'):raise ValueError('Unfrozen candidate index')
        for row in index['rows']:
            if row['quality_state']!='candidate_accepted' or row['split']!=side:raise ValueError('Unaccepted row')
            image=str(Path(row['image']).resolve());scene=str(Path(row['scene']).resolve())
            if images.setdefault(image,side)!=side or scenes.setdefault(scene,side)!=side:
                raise ValueError('Same image/dual-view layout crosses partitions')
            # Audit everyone rendered, including people with no target box.
            people=json.loads((Path(scene)/'people.json').read_text())['people']
            for person in people:
                c=characters.get(person['character_id'])
                if c is None or c['partition']!=side:raise ValueError('Target/occluder identity leakage')
                sources=split['source_groups'][side]
                expected=sources['cycling'] if person['action']=='cycling' else sources['pedestrian']
                if person['source_group']!=expected:raise ValueError('Target/occluder pose-source leakage')
            if row.get('occlusion_pair_id'):
                key=(row['occlusion_pair_id'],row['camera'])
                signature=(side,row['character_id'],row['pose_source_id'],row['camera'],
                    tuple(row['given_bbox_xyxy']),tuple(row['affine_flat']),
                    tuple(row['target_object_to_world_flat']),tuple(row['camera_to_world_flat']))
                if paired.setdefault(key,signature)!=signature:raise ValueError('Paired target/camera/box/affine changed')
            rows_count[side]+=1
    memberships['image']={s:sorted(v for v,side in images.items() if side==s) for s in count}
    memberships['layout']={s:sorted(v for v,side in scenes.items() if side==s) for s in count}
    intersections={key:sorted(set(values['train'])&set(values['val'])) for key,values in memberships.items()}
    if any(intersections.values()):raise ValueError('Unexpected train/val intersection')
    return dict(status='passed_proposal_only' if allow_proposal else 'passed',characters=count,
        candidate_rows=rows_count,images=len(images),layouts=len(scenes),
        memberships=memberships,train_val_intersections=intersections,
        meaning='Declared source-risk groups and every rendered target/occluder checked; not proof of unknown photographed face identities')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True);p.add_argument('--train-index',type=Path)
    p.add_argument('--val-index',type=Path);p.add_argument('--allow-proposal',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    indices={s:json.loads(v.read_text()) for s,v in [('train',a.train_index),('val',a.val_index)] if v}
    result=validate_split(json.loads(a.split.read_text()),indices,a.allow_proposal)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
