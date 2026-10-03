"""Independent BVH clothing crossings and sampled hair/cloth segments; diagnostic only."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[2];sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.rendering.blender.intersections import crossings

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    root=a.assets.resolve();report=json.loads((root/'report.json').read_text());rows=[];start=time.monotonic()
    for job in report['derivatives']:
        g=Path(job['geometry']);c=Path(job['clothing_geometry']);h=Path(job['groom_geometry'])
        v=np.load(g/'vertices.npy')[0];f=np.load(g/'faces.npy');cv=np.load(c/'clothing_vertices.npy')[0];cf=np.load(c/'clothing_faces.npy')
        pairs,locations,_=crossings(cv,cf,v,f);self_pairs,self_locations,_=crossings(cv,cf)
        points=np.load(h/'points.npy');offsets=np.load(h/'curve_offsets.npy')
        active=np.load(h/'active_curve_mask.npy') if (h/'active_curve_mask.npy').exists() else np.ones(len(offsets)-1,bool)
        tree=BVHTree.FromPolygons(cv.tolist(),cf.tolist(),all_triangles=True);hits=[];segments=0
        for curve in np.flatnonzero(active)[::50]:
            poly=points[offsets[curve]:offsets[curve+1]]
            for x,y in zip(poly[:-1],poly[1:]):
                delta=y-x;length=float(np.linalg.norm(delta))
                if length<1e-9:continue
                hit=tree.ray_cast(Vector(x),Vector(delta/length),length);segments+=1
                if hit[0] is not None:hits.append(list(hit[0]))
        out=root/'contact_diagnostics'/job['character_id']/job['pose_source_id'];out.mkdir(parents=True)
        np.savez(out/'crossings.npz',body_pairs=pairs,body_locations=locations,self_pairs=self_pairs,self_locations=self_locations,hair_cloth_hits=np.asarray(hits))
        rows.append(dict(character_id=job['character_id'],pose_source_id=job['pose_source_id'],
            clothing_body_crossing_pairs=len(pairs),clothing_self_crossing_pairs=len(self_pairs),
            hair_clothing_sampled_segments=segments,hair_clothing_hit_segments=len(hits),
            interpretation='diagnostic only; image/depth/instance defects decide acceptance',
            limitation='proper noncoplanar triangle crossings; every 50th active control polygon, not exhaustive evaluated NURBS or containment proof'))
        print(job['character_id'],rows[-1],flush=True)
    (root/'contact_diagnostics.json').write_text(json.dumps(dict(cases=rows,seconds=time.monotonic()-start,split='integration_only'),indent=2)+'\n')

if __name__=='__main__':main()
