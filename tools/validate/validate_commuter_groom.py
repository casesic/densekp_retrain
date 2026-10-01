"""Independent saved binding/collision readback; does not replace curve or clothing visual review."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.bind_commuter_groom import CONV,triangle_frame
from tools.experiments.prepare_cycling_characters import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('assets',type=Path)
    args=parser.parse_args();root=args.assets.resolve();report=json.loads((root/'report.json').read_text());records=[]
    for character in report['characters']:
        manifest=json.loads(Path(character['manifest']).read_text())
        rest=np.load(Path(manifest['reference']['geometry'])/'vertices.npy')[0]@CONV.T
        jobs=[manifest['reference']]+[d for d in report['derivatives'] if d['character_id']==character['character_id']]
        previous_active=None
        for job in jobs:
            folder=Path(job['groom_geometry']);info=json.loads((folder/'groom_manifest.json').read_text())
            binding=np.load(Path(info['binding'])/'binding.npz');faces=binding['body_faces'];weights=binding['weights']
            body=np.load(Path(job['geometry'])/'vertices.npy')[0]@CONV.T
            np.testing.assert_allclose(weights.sum(1),1,atol=1e-6,rtol=0)
            if (weights<0).any():raise ValueError('Invalid root barycentrics')
            anchor=(body[faces]*weights[:,:,None]).sum(1)
            rotation=np.einsum('nij,nkj->nik',triangle_frame(body,faces),triangle_frame(rest,faces))
            frames=np.einsum('nij,njk->nik',rotation,binding['target_rest_frames'])
            target=anchor+frames[:,:,2]*.0004
            offsets=binding['curve_offsets'];curve=np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
            original=(target[curve]+np.einsum('nij,nj->ni',frames[curve],binding['local_points']))@CONV
            expected=original+np.load(folder/'collision_correction.npy')
            actual=np.load(folder/'points.npy');active=np.load(folder/'active_curve_mask.npy')
            np.testing.assert_array_equal(offsets,np.load(folder/'curve_offsets.npy'))
            if previous_active is not None:np.testing.assert_array_equal(active,previous_active)
            previous_active=active
            error=float(np.abs(expected[active[curve]]-actual[active[curve]]).max())
            if error>2e-6:raise ValueError('Hair no longer reconstructs from its source binding')
            root_error=float(np.linalg.norm(actual[offsets[:-1]][active]-(target@CONV)[active],axis=1).max())
            if root_error>2e-6:raise ValueError('Retained roots drifted from target scalp')
            radius=np.load(folder/'radius.npy')
            if not np.isfinite(actual).all() or not np.isfinite(radius).all() or (radius<0).any():raise ValueError('Invalid groom arrays')
            records.append(dict(character_id=character['character_id'],geometry=job['geometry'],groom=str(folder),
                reconstruction_max_m=error,root_binding_max_m=root_error,retained_curves=int(active.sum()),
                removed_curves=int((~active).sum()),retained_points=int(active[curve].sum()),
                max_radius_m=float(radius[active[curve]].max()),collision=info['collision']))
    write_json(root/'groom_independent_readback.json',dict(status='binding_readback_passed',split='integration_only',
               cases=records,threshold_m=2e-6,
               limitations=['not an exhaustive evaluated NURBS/body intersection check','clothing contact and DS render remain separate gates']))


if __name__=='__main__':main()
