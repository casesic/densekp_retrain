"""Record reviewed candidates separately from diagnostic rows and superseded versions."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.prepare_cycling_characters import write_json,POSES

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);p.add_argument('--visual-review',type=Path,required=True);a=p.parse_args()
    root=a.assets.resolve();report=json.loads((root/'report.json').read_text());visual=json.loads(a.visual_review.read_text())
    ids={c['character_id'] for c in report['characters']}
    if visual['status']!='passed_for_candidate_use' or set(visual['reviewed_assets'])!=ids:
        raise ValueError('Explicit completed visual review must cover these exact asset versions')
    if json.loads((root/'body_clothing_readback.json').read_text())['status']!='passed':raise ValueError('Body readback missing')
    if json.loads((root/'groom_binding_readback.json').read_text())['status']!='passed':raise ValueError('Groom readback missing')
    rows=[];excluded=[];validations=[];groom_checks=[];scenes=[]
    for folder in sorted(root.glob('ds_*')):
        if not (folder/'validation.json').exists():continue
        v=json.loads((folder/'validation.json').read_text());g=json.loads((folder/'groom_validation.json').read_text())
        if v['status']!='passed' or g['status']!='world_transform_and_instance_assignment_passed':raise ValueError('DS numeric gate failed')
        validations.append(v);groom_checks.extend(g['cases']);scene=json.loads((folder/'report.json').read_text());scenes.append(folder)
        people={p['id']:p for p in scene['people']};z=np.load(folder/'diagnostic_loader.npz')
        for i,(pid,camera) in enumerate(zip(z['person_id'],z['camera_id'])):
            person=people[int(pid)];target=z['proj_verts'][i,:,2]>0;visible=z['visible'][i]
            row=dict(scene=str(folder),camera=str(camera),person_id=int(pid),character_id=person['character_id'],
                pose_source_id=person['pose_source_id'],loader_row=i,loader_input=str(folder/'diagnostic_loader.npz'),
                image=str(folder/str(camera)/'normal_erp.png'),square=str(folder/'actual_inputs'/f'{camera}_person_{pid}_square.png'),
                network_center=str(folder/'actual_inputs'/f'{camera}_person_{pid}_center.png'),
                supervised_points=int(target.sum()),supervised_occluded_points=int((target&~visible).sum()),
                quality_state='candidate_accepted' if person['character_id'] in ids else 'excluded_superseded_appearance',split='integration_only')
            (rows if person['character_id'] in ids else excluded).append(row)
    if {r['character_id'] for r in rows}!=ids:raise ValueError('Every adult needs an accepted given-box sample')
    characters=[];combos=[];not_attempted=[]
    for item in sorted(report['characters'],key=lambda x:x['character_id']):
        mp=Path(item['manifest']);m=json.loads(mp.read_text());cid=m['character_id'];jobs=[d for d in report['derivatives'] if d['character_id']==cid]
        for d in jobs:
            samples=[x for x in rows if x['character_id']==cid and x['pose_source_id']==d['pose_source_id']]
            combos.append(dict(character_id=cid,pose_source_id=d['pose_source_id'],status='passed_for_candidate_use',
                geometry=d['geometry'],clothing=d['clothing_geometry'],groom=d['groom_geometry'],lineage=d['lineage'],
                ds_boxes=len(samples),qualification='reviewed finite views only; not zero-intersection or arbitrary-clothing guarantee'))
        present={d['pose_source_id'] for d in jobs}
        not_attempted.extend(dict(character_id=cid,pose_source_id=pose,status='not_attempted',reason='bounded scope; one passing pose sufficient') for pose in POSES if pose not in present)
        sample_rows=[x for x in rows if x['character_id']==cid]
        info=json.loads((Path(m['reference']['groom_geometry'])/'groom_manifest.json').read_text())
        parents=[];parent=m.get('parent_manifest')
        while parent:
            pm=json.loads(Path(parent).read_text());parents.append(dict(asset_id=pm['character_id'],manifest=parent));parent=pm.get('parent_manifest')
        m.update(status='frozen_candidate_for_densekp_integration',age_group='adult',appearance_acceptance=str(a.visual_review.resolve()),
                 training_eligibility='integration_only; formal source grouping and split not assigned')
        write_json(mp,m)
        characters.append(dict(asset_id=cid,identity_family_id=m['identity_family_id'],manifest=str(mp),parent_asset_id=m['parent_asset_id'],
            parent_chain=parents,gender=m['gender'],height_m=m['proportions']['height_m'],model_file=m['model_file'],
            shape_source=m['shape_fit'],betas=m['betas'],skin=m['texture_source'],appearance=m['appearance'],groom_source=info['source_abc'],
            groom_geometry=m['reference']['groom_geometry'],outfit=m['outfit'],model_license=m['model_license'],texture_license=m['texture_license'],
            hair_license='https://bedlam2.is.tuebingen.mpg.de/license.html',status=m['status'],age_group='adult',split='integration_only',
            ds_boxes=len(sample_rows),observed_cameras=sorted({x['camera'] for x in sample_rows}),poses=sorted(present)))
    counts=dict(adults=len(ids),female=sum(c['gender']=='female' for c in characters),male=sum(c['gender']=='male' for c in characters),
        independent_pose_sources=len({x['pose_source_id'] for x in rows}),current_combinations=len(combos),not_attempted_combinations=len(not_attempted),
        ds_layouts=len(scenes),ds_images=2*len(scenes),diagnostic_boxes=len(rows)+len(excluded),candidate_boxes=len(rows),excluded_superseded_boxes=len(excluded),
        supervised_points=sum(x['supervised_points'] for x in rows),supervised_occluded_points=sum(x['supervised_occluded_points'] for x in rows),
        diagnostic_supervised_points=sum(v['supervised_points'] for v in validations),diagnostic_supervised_occluded_points=sum(v['supervised_occluded_points'] for v in validations))
    write_json(root/'character_manifest.json',dict(characters=characters,counts=counts,split='integration_only',
        identity_scope='synthetic families, not ten verified independent scanned face sources',shoe_socks='optional, absent',children='deferred'))
    write_json(root/'combination_status.json',dict(accepted=combos,not_attempted=not_attempted,excluded_ds_rows=excluded,
        failed_appearance_versions=['cycling_F04_v1','cycling_F04_v2','cycling_F05_v1'],
        failed_appearance_reasons={'cycling_F04_v1':'same-condition face detail too close to F02',
            'cycling_F04_v2':'first skin replacement still too close to F02',
            'cycling_F05_v1':'source face detail too close to F01; retained geometry, revised brows/lip texture'},
        not_applicable=[dict(component='old complex commuter outfits',reason='not selected for the simplified route; original failures retained',
            reports=['docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md','docs/appearance/COMMUTER_CLOTHING_REPAIR_20260930.md'])],
        old_commuter_failures='preserved in their original experiment directories; not reclassified as passed',
        internal_crossings='diagnostic, not zero; see contact_diagnostics.json',split='integration_only'))
    write_json(root/'candidate_index.json',dict(rows=rows,excluded_rows=excluded,split='integration_only',not_a_train_val_split=True,
        selection='pre-training source/appearance quality; no model predictions used'))
    maxima={key:max(v[key] for v in validations) for key in ['crop_roundtrip_max_px','operator_max_error_m','rendered_mesh_operator_max_error_m','rendered_clothing_transform_max_error_m']}
    maxima['groom_world_max_m']=max(x['world_max_m'] for x in groom_checks)
    write_json(root/'acceptance.json',dict(status='passed_for_bounded_densekp_candidate_experiments',counts=counts,maxima=maxima,
        visual_review=str(a.visual_review.resolve()),identity_review=str(root/'identity_review.json'),split='integration_only',
        exclusions='old face variants excluded from current candidate rows, not deleted',
        remaining=['no independent split','eight P2 combinations untested','F05/M05 given boxes currently cam2 only','no zero-intersection claim','low-sample render noise'],
        source_gates_preserved=True,old_failed_reports_preserved=True))
    report.update(status='bounded_candidates_accepted_integration_only',counts=counts,acceptance=str(root/'acceptance.json'),finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'report.json',report);print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
