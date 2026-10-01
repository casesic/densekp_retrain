"""Read stored groom bindings independently; labels remain defined on the body."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.bind_commuter_groom import CONV,triangle_frame
from tools.experiments.prepare_cycling_characters import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);a=p.parse_args();root=a.assets.resolve()
    report=json.loads((root/'report.json').read_text());cases=[]
    for row in report['characters']:
        m=json.loads(Path(row['manifest']).read_text());jobs=[m['reference']]+[d for d in report['derivatives'] if d['character_id']==m['character_id']]
        rest=np.load(Path(m['reference']['geometry'])/'vertices.npy')[0]@CONV.T
        for job in jobs:
            folder=Path(job['groom_geometry']);info=json.loads((folder/'groom_manifest.json').read_text())
            z=np.load(Path(info['binding'])/'binding.npz');tri=z['body_faces'];weights=z['weights'];offsets=z['curve_offsets']
            body=np.load(Path(job['geometry'])/'vertices.npy')[0]@CONV.T
            tf=z['target_rest_frames'];rotation=np.einsum('nij,nkj->nik',triangle_frame(body,tri),triangle_frame(rest,tri))
            target=np.einsum('nij,njk->nik',rotation,tf);roots=(body[tri]*weights[:,:,None]).sum(1)+target[:,:,2]*.0004
            pc=np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
            expected=(roots[pc]+np.einsum('nij,nj->ni',target[pc],z['local_points']))@CONV
            if (folder/'collision_correction.npy').exists():expected+=np.load(folder/'collision_correction.npy')
            actual=np.load(folder/'points.npy');error=float(np.abs(expected-actual).max())
            root_error=float(np.abs(roots@CONV-np.load(folder/'target_roots.npy')).max())
            if error>2e-6 or root_error>1e-6:raise ValueError(f'{m["character_id"]}: groom readback {error}, root {root_error}')
            cases.append(dict(character_id=m['character_id'],pose=job['pose_source_id'],readback_max_m=error,root_max_m=root_error,
                              body_provenance=info['body_geometry'],source_abc=info['source_abc']))
    write_json(root/'groom_binding_readback.json',dict(status='passed',cases=cases,readback_gate_m=2e-6,root_gate_m=1e-6,
        limitation='binding correctness only; sampled contact and appearance separate'))
    print('passed',len(cases))

if __name__=='__main__':main()
