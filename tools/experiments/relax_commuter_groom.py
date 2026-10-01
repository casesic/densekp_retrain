"""Bounded control-point/body collision correction; originals and bindings remain intact."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.build_commuter_clothing import closest_surface
from tools.experiments.prepare_cycling_clothing import vertex_normals
from tools.experiments.prepare_cycling_characters import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('assets',type=Path)
    parser.add_argument('--source-version',default='v1');parser.add_argument('--version',default='v4')
    args=parser.parse_args();root=args.assets.resolve();start=time.monotonic()
    report=json.loads((root/'report.json').read_text());records=[]
    for character in report['characters']:
        mp=Path(character['manifest']);manifest=json.loads(mp.read_text())
        jobs=[manifest['reference']]+[d for d in report['derivatives'] if d['character_id']==character['character_id']]
        for job in jobs:
            source=Path(job['geometry']).parent/f'groom_{args.source_version}'
            info=json.loads((source/'groom_manifest.json').read_text())
            geometry=Path(job['geometry']);body=np.load(geometry/'vertices.npy')[0];faces=np.load(geometry/'faces.npy')
            original=np.load(source/'points.npy');points=original.copy();radii=np.load(source/'radius.npy');offsets=np.load(source/'curve_offsets.npy')
            roots=np.zeros(len(points),bool);roots[offsets[:-1]]=True
            vertex_tree=cKDTree(body);face_tree=cKDTree(body[faces].mean(1));normal=vertex_normals(body,faces)
            max_corrected=0;exhausted=False
            for iteration in range(5):
                _,nearest=vertex_tree.query(points)
                approximate=((points-body[nearest])*normal[nearest]).sum(1)
                candidates=np.flatnonzero((approximate<.006)&~roots)
                count=0
                for i in range(0,len(candidates),8192):
                    ids=candidates[i:i+8192]
                    _,n,d,_=closest_surface(points[ids],body,faces,face_tree)
                    target=np.maximum(.0003,radii[ids])
                    bad=d<target
                    if bad.any():
                        points[ids[bad]]+=(target[bad]-d[bad])[:,None]*n[bad];count+=int(bad.sum())
                magnitude=np.linalg.norm(points-original,axis=1);max_corrected=max(max_corrected,float(magnitude.max()))
                if magnitude.max()>.03:exhausted=True;break
                if count==0:break
            sample=np.unique(np.r_[np.arange(0,len(points),20),offsets[:-1]])
            signed=[]
            for i in range(0,len(sample),8192):signed.append(closest_surface(points[sample[i:i+8192]],body,faces,face_tree)[2])
            signed=np.concatenate(signed)
            root_error=float(np.linalg.norm(points[offsets[:-1]]-np.load(source/'target_roots.npy'),axis=1).max())
            out=source.with_name(f'groom_{args.version}');out.mkdir()
            for name,value in [('points',points),('radius',radii),('curve_offsets',offsets),
                               ('target_roots',np.load(source/'target_roots.npy')),('collision_correction',points-original)]:
                np.save(out/f'{name}.npy',value)
            info.update(derived_from=str(source),collision=dict(method='all-point nearest-vertex candidate filter, exact closest point among 24 nearby triangle centroids; final every-20th control point plus all roots',
                checked_points=len(sample),min_sample_signed_m=float(signed.min()),below_minus_1mm=int((signed<-.001).sum()),
                max_correction_m=max_corrected,root_drift_max_m=root_error,iterations=iteration+1,budget_exhausted=exhausted,
                limitation='control-point sampling is not an exhaustive NURBS-curve or strand self-intersection proof'),
                status='numeric_candidate_visual_pending' if not exhausted and not (signed<-.001).any() and root_error<1e-6 else 'failed_collision',
                thresholds=dict(max_correction_m=.03,min_sample_signed_m=-.001,max_root_drift_m=1e-6))
            write_json(out/'groom_manifest.json',info);job['groom_geometry']=str(out);records.append(info)
            print(json.dumps(dict(geometry=str(geometry),collision=info['collision'],status=info['status'])),flush=True)
        manifest['hair_status']='candidate_review_required';write_json(mp,manifest)
    write_json(root/'report.json',report)
    write_json(root/f'groom_relaxation_{args.version}.json',dict(seconds=time.monotonic()-start,cases=records,split='integration_only'))


if __name__=='__main__':main()
