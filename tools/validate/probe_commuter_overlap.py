"""Multi-point solid-angle audit of saved F02/P2 geometry, no pose changes."""
import argparse
import json
from pathlib import Path
import numpy as np


def winding(points,vertices,faces):
    triangles=vertices[faces].astype(np.float64);values=[]
    for point in points:
        a,b,c=(triangles-point).transpose(1,0,2)
        la,lb,lc=[np.linalg.norm(x,axis=1) for x in (a,b,c)]
        numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        values.append(float(np.arctan2(numerator,denominator).sum()/(2*np.pi)))
    return np.asarray(values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--localized',action='store_true');args=p.parse_args()
    diag=args.root/'diagnosis/cycling_F02_v2/coco_306800_person_225160/evidence.npz';z=np.load(diag)
    pairs=z['armpit_body_pairs'];bf=z['body_faces'];body=z['body']
    if args.localized:
        old=Path(json.loads((diag.parent/'audit.json').read_text())['geometry']).parents[3]
        rest=np.load(old/'characters/cycling_F02_v2/reference/geometry/vertices.npy')[0]
        centers=rest[bf].mean(1)
        mask=(np.abs(centers[:,0])>.12)&(np.abs(centers[:,0])<.30)&(centers[:,1]>.28)&(centers[:,1]<.47)
        pairs=z['body_self_pairs'][mask[z['body_self_pairs']].any(1)]
    ids=np.unique(pairs)
    tri=body[bf[ids]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(normal,axis=1)*.5
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    points=tri.mean(1)-normal*.001;values=winding(points,body,bf)
    candidates=[]
    for name in ['F02_top_union','F02_top_dqs_union']:
        folder=args.root/name/'P2';b=np.load(folder/'binding.npz');i=int(b['reverse_surface_distance'].argmax())
        parent=Path(json.loads((folder/'volume.json').read_text())['parent']);s=np.load(parent/'evidence.npz')
        q=s['bound'][i]
        # A 3x3x3 neighborhood checks the original single-sample inference.
        offsets=np.stack(np.meshgrid(*([[-.003,0,.003]]*3),indexing='ij'),-1).reshape(-1,3)
        local=winding(q+offsets,body,bf)
        candidates.append(dict(garment=name,vertex=i,rest_position=s['rest'][i].tolist(),posed_position=q.tolist(),
            reverse_surface_distance_m=float(b['reverse_surface_distance'][i]),neighborhood_offsets_m=offsets.tolist(),
            winding_values=local.tolist(),inside_two_layers_count=int((local>1.5).sum())))
    report=dict(split='integration_only',body_geometry=str(diag),armpit_intersecting_face_pairs=len(pairs),
        region='reference face centers .12 < abs(x) < .30 and .28 < y < .47 m' if args.localized else 'broad upper-limb band; includes forearm, not strictly axilla',
        unique_intersecting_faces=len(ids),summed_face_area_m2=float(area.sum()),
        face_centroid_inward_offset_m=.001,face_sample_winding=values.tolist(),face_samples_above_1p5=int((values>1.5).sum()),
        neighborhoods=candidates,body_changed=False,
        interpretation='multiple crossing triangles and solid-angle neighborhoods support local body overlap; not proof that every garment method or original source pose is unusable',
        limitation='summed triangle area is affected-face area, not intersection volume; nearest envelope distance is not a certified minimum body escape distance')
    name='body_overlap_armpit_probe' if args.localized else 'body_overlap_multi_probe'
    np.savez_compressed(args.root/f'{name}_samples.npz',face_ids=ids,points=points,winding=values)
    (args.root/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['face_sample_winding','neighborhoods']}))
    print([(c['garment'],c['inside_two_layers_count']) for c in candidates])


if __name__=='__main__':main()
