"""Read-only geometry audit of preserved commuter prototypes (run with Blender)."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.validate.commuter_intersections import crossings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    old = json.loads((args.source / 'clothing_build_v4.json').read_text())
    report = {'split': 'integration_only', 'method': 'BVH broad phase, segment-triangle narrow phase; no nearest-centroid truncation', 'cases': []}
    for case in old['cases']:
        geom, cloth = Path(case['geometry']), Path(case['clothing'])
        pose = 'reference' if 'reference' in geom.parts else geom.parent.name
        out = args.output / case['character_id'] / pose
        out.mkdir(parents=True)
        body, bf = np.load(geom/'vertices.npy')[0], np.load(geom/'faces.npy')
        cf = np.load(cloth/'clothing_faces.npy')
        rv = np.load(cloth/'clothing_rest_vertices.npy')
        bound = np.load(cloth/'reference_body_vertices.npy')[0] + np.einsum(
            'nab,nb->na', np.load(cloth/'binding_rotations.npy'), np.load(cloth/'rest_offsets.npy'))
        final = np.load(cloth/'clothing_vertices.npy')[0]
        record = {'character_id': case['character_id'], 'pose': pose, 'geometry': str(geom), 'clothing': str(cloth), 'checks': {}}
        arrays = {'body': body, 'body_faces': bf, 'rest': rv, 'bound': bound, 'final': final, 'cloth_faces': cf}
        tests = [('body_self', body, bf, None, None)]
        for name, vertices in [('bound', bound), ('final', final)]:
            tests += [(name+'_body', vertices, cf, body, bf), (name+'_self', vertices, cf, None, None)]
        if pose == 'reference':
            tests += [('rest_body', rv, cf, body, bf), ('rest_self', rv, cf, None, None)]
        for name, va, fa, vb, fb in tests:
            pairs, locations, coplanar = crossings(va, fa, vb, fb)
            arrays[name+'_pairs'], arrays[name+'_locations'] = pairs, locations
            record['checks'][name] = {'proper_crossing_pairs': len(pairs), 'unresolved_coplanar_candidates': coplanar}
            if len(locations):
                record['checks'][name]['bounds'] = [locations.min(0).tolist(), locations.max(0).tolist()]
            if name == 'body_self':
                restbody = np.load(args.source/'characters'/case['character_id']/'reference/geometry/vertices.npy')[0]
                mids = restbody[bf].mean(1)
                arm = (np.abs(mids[:, 0]) > .12) & (mids[:, 1] > .29) & (mids[:, 1] < .49)
                mask = arm[pairs].any(1) if len(pairs) else np.zeros(0, bool)
                record['checks'][name]['armpit_region_pairs'] = int(mask.sum())
                arrays['armpit_body_pairs'] = pairs[mask]
                arrays['armpit_body_locations'] = locations[mask]
        edges = np.unique(np.sort(np.concatenate([cf[:, [0,1]], cf[:, [1,2]], cf[:, [2,0]]]), axis=1), axis=0)
        restlen = np.linalg.norm(rv[edges[:, 1]]-rv[edges[:, 0]], axis=1)
        for name, vertices in [('bound', bound), ('final', final)]:
            ratio = np.linalg.norm(vertices[edges[:, 1]]-vertices[edges[:, 0]], axis=1)/np.maximum(restlen, 1e-9)
            record[name+'_edge_ratio'] = dict(zip(['min','p01','median','p99','max'], np.quantile(ratio,[0,.01,.5,.99,1]).tolist()))
        np.savez_compressed(out/'evidence.npz', **arrays)
        (out/'audit.json').write_text(json.dumps(record, indent=2)+'\n')
        report['cases'].append(record)
        print(json.dumps(record), flush=True)
    # Audit the actual stored numerical initialization, without evaluating a new body.
    path = args.source/'derivatives/cycling_F02_v2/coco_306800_person_225160/cloth_simulation_v6/input.npz'
    with np.load(path) as z:
        history = []
        for index, body in enumerate(z['body_path']):
            pairs, locations, coplanar = crossings(body, z['body_faces'])
            mids = z['body_path'][0][z['body_faces']].mean(1)
            arm = (np.abs(mids[:, 0])>.12)&(mids[:, 1]>.29)&(mids[:, 1]<.49)
            history.append({'sample': index, 'proper_crossing_pairs': len(pairs),
                            'armpit_region_pairs': int(arm[pairs].any(1).sum()), 'coplanar_unresolved': coplanar})
        report['stored_initialization_body_self_crossings'] = history
    report['seconds'] = time.monotonic()-start
    (args.output/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
