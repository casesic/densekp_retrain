"""Physically remove a bounded number of invalid strands, consistently across poses."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.clothing.binding import closest_surface
from densekp_retrain.io.json import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('assets',type=Path)
    args=parser.parse_args();root=args.assets.resolve();start=time.monotonic()
    report=json.loads((root/'report.json').read_text());records=[]
    for character in report['characters']:
        mp=Path(character['manifest']);manifest=json.loads(mp.read_text())
        jobs=[manifest['reference']]+[d for d in report['derivatives'] if d['character_id']==character['character_id']]
        scans=[];removed=set();reference_offsets=None;reference_source=None
        for job in jobs:
            source=Path(job['geometry']).parent/'groom_v4';info=json.loads((source/'groom_manifest.json').read_text())
            points=np.load(source/'points.npy');offsets=np.load(source/'curve_offsets.npy');correction=np.load(source/'collision_correction.npy')
            if reference_offsets is not None:
                np.testing.assert_array_equal(offsets,reference_offsets)
                if info['source_abc']!=reference_source:raise ValueError('An identity changed groom across poses')
            reference_offsets=offsets;reference_source=info['source_abc']
            point_curve=np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
            sample=np.unique(np.r_[np.arange(0,len(points),20),offsets[:-1]])
            geometry=Path(job['geometry']);body=np.load(geometry/'vertices.npy')[0];faces=np.load(geometry/'faces.npy');tree=cKDTree(body[faces].mean(1))
            signed=[]
            for i in range(0,len(sample),8192):signed.append(closest_surface(points[sample[i:i+8192]],body,faces,tree)[2])
            signed=np.concatenate(signed);magnitude=np.linalg.norm(correction,axis=1)
            bad=np.unique(np.r_[point_curve[sample[signed<-.001]],point_curve[magnitude>.03]])
            removed.update(bad.tolist());scans.append((job,source,info,sample,signed,magnitude,bad))
        active=np.ones(len(offsets)-1,bool);active[list(removed)]=False
        if (~active).mean()>.01:raise ValueError('More than 1% of groom would be removed; choose another asset instead')
        for job,source,info,sample,signed,magnitude,bad in scans:
            out=source.with_name('groom_v5');out.mkdir()
            active_points=active[point_curve];retained_samples=active_points[sample]
            # Keep provenance arrays in their original indexing. The renderer actually
            # removes inactive curves, rather than merely ignoring them in the audit.
            for name in ['points','radius','curve_offsets','target_roots','collision_correction']:
                np.save(out/f'{name}.npy',np.load(source/f'{name}.npy'))
            np.save(out/'active_curve_mask.npy',active)
            collision=dict(info['collision']);collision.update(min_sample_signed_m=float(signed[retained_samples].min()),
                below_minus_1mm=int((signed[retained_samples]<-.001).sum()),
                max_correction_m=float(magnitude[active_points].max()),budget_exhausted=False,checked_points=int(retained_samples.sum()))
            info.update(derived_from=str(source),collision=collision,status='numeric_candidate_visual_pending',
                curve_pruning=dict(removed_count=len(removed),fraction=float((~active).mean()),max_allowed_fraction=.01,
                    removed_curve_ids=sorted(removed),shared_across_reference_and_both_poses=True,
                    reasons='union of strands exceeding 30mm body correction or retaining sampled >1mm body penetration',
                    rendered_curves=int(active.sum()),rendered_control_points=int(active_points.sum())))
            write_json(out/'groom_manifest.json',info);job['groom_geometry']=str(out);records.append(info)
        manifest['hair_status']='body_collision_numeric_passed_visual_and_clothing_contact_pending';write_json(mp,manifest)
    write_json(root/'report.json',report)
    write_json(root/'groom_pruning_v5.json',dict(seconds=time.monotonic()-start,cases=records,split='integration_only'))


if __name__=='__main__':main()
