"""Open cap surfaces, bind to the parent garment, test the untouched body."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import bmesh
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parents[2];sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.rendering.blender.intersections import crossings
from tools.experiments.repair_commuter_top import edges_of,average


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--smooth',action='store_true')
    p.add_argument('--planar-y-openings',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);start=time.monotonic()
    info=json.loads((args.folder/'volume.json').read_text());parent=Path(info['parent']);source=np.load(parent/'evidence.npz')
    out=args.output or args.folder
    if args.output:
        out.mkdir(parents=True,exist_ok=False)
        (out/'volume.json').write_text(json.dumps(dict(info,source_volume=str(args.folder)),indent=2)+'\n')
    vol=np.load(info.get('volume_path',args.folder/'volume.npz'));mesh=np.load(args.folder/'closed_envelope.npz');v=mesh['vertices'];f=mesh['faces']
    tree=lambda v,f:BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
    capset=set(vol['cap_faces'].tolist());capped=tree(vol['garment_vertices'],vol['garment_faces'])
    if args.planar_y_openings:
        parentmeta=json.loads((parent/'repair.json').read_text())
        if parentmeta['pose']!='reference' or parentmeta['component']!='trousers':
            raise ValueError('Exact horizontal cuts apply only to this reference trouser pattern')
        pe,pc=edges_of(source['cloth_faces']);boundary=np.unique(pe[pc==1])
        y=source['bound'][boundary,1]
        if np.ptp(y[y>0])>1e-5 or np.ptp(y[y<-.5])>1e-5:
            raise ValueError('Offset waist/cuff boundaries are not planar; bounding-box cuts would retain cap fragments')
        import bpy
        data=bpy.data.meshes.new('OpenEnvelope');data.from_pydata(v.tolist(),[],f.tolist())
        bm=bmesh.new();bm.from_mesh(data)
        low,high=float(source['bound'][:,1].min()),float(source['bound'][:,1].max())
        for height,normal in [(low,(0,1,0)),(high,(0,-1,0))]:
            bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
                dist=1e-7,plane_co=(0,height,0),plane_no=normal,clear_inner=True)
        bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.verts.index_update()
        v=np.asarray([v.co[:] for v in bm.verts],np.float32)
        f=np.asarray([[v.index for v in face.verts] for face in bm.faces],np.int32)
        bm.free();bpy.data.meshes.remove(data)
    else:
        remove=[]
        for point in v[f].mean(1):
            near,normal,faceid,_=capped.find_nearest(point.tolist());remove.append(faceid in capset)
        f=f[~np.asarray(remove)]
    used,inverse=np.unique(f,return_inverse=True);v=v[used];f=inverse.reshape(-1,3)
    smoothing_max=0.
    if args.smooth:
        initial=v.copy();edges,counts=edges_of(f);boundary=np.unique(edges[counts==1])
        for _ in range(6):
            for weight in [.5,-.53]:
                delta=weight*(average(v,edges)-v);delta[boundary]=0;v+=delta
        smoothing_max=float(np.linalg.norm(v-initial,axis=1).max())
    parenttree=tree(source['bound'],source['cloth_faces']);parentindex=[];near=[]
    for point in v:
        q,_,i,_=parenttree.find_nearest(point.tolist());near.append(q[:]);parentindex.append(i)
    idx=source['cloth_faces'][parentindex];tri=source['bound'][idx].astype(float);near=np.asarray(near)
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0];d=near-tri[:,0]
    aa=(e1*e1).sum(1);ab=(e1*e2).sum(1);bb=(e2*e2).sum(1);ad=(e1*d).sum(1);bd=(e2*d).sum(1)
    det=aa*bb-ab*ab;u=(bb*ad-ab*bd)/det;w=(aa*bd-ab*ad)/det
    weights=np.clip(np.column_stack((1-u-w,u,w)),0,1);weights/=weights.sum(1,keepdims=True)
    anchor=(source['bound'][idx]*weights[:,:,None]).sum(1);correction=v-anchor
    if not all(np.isfinite(value).all() for value in [v,weights,anchor,correction]):
        raise ValueError('Nonfinite remesh binding; cannot export a candidate')
    finaltree=tree(v,f)
    reverse=np.array([finaltree.find_nearest(point.tolist())[3] for point in source['bound']])
    pairs,points,_=crossings(v,f,source['body'],source['body_faces']);selfpairs,selfpoints,_=crossings(v,f)
    np.savez_compressed(out/'binding.npz',parent_triangle_vertices=idx,parent_weights=weights,
                        correction=correction.astype(np.float32),reverse_surface_distance=reverse)
    np.savez_compressed(out/'evidence.npz',body=source['body'],body_faces=source['body_faces'],cloth_faces=f,
                        final=v,bound=v-correction,final_body_pairs=pairs,final_body_locations=points,
                        final_self_pairs=selfpairs,final_self_locations=selfpoints)
    np.save(out/'clothing_vertices.npy',v[None]);np.save(out/'clothing_faces.npy',f)
    if json.loads((parent/'repair.json').read_text())['pose']=='reference':
        np.save(out/'clothing_rest_vertices.npy',v)
    forward=np.linalg.norm(correction,axis=1);error=float(np.abs(anchor+correction.astype(np.float32)-v).max())
    report=dict(split='integration_only',parent=str(parent),body_changed=False,contract='parent_garment_barycentric_plus_recorded_remesh_correction',
                vertices=len(v),faces=len(f),body_crossings=len(pairs),self_crossings=len(selfpairs),
                smoothing=dict(method='six Taubin lambda .5 mu -.53 pairs, open boundaries fixed' if args.smooth else 'none',max_displacement_m=smoothing_max),
                openings='exact original reference waist/cuff planes' if args.planar_y_openings else 'closest original cap classification',
                forward_distance_quantiles_m=np.quantile(forward,[0,.5,.95,1]).tolist(),
                reverse_distance_quantiles_m=np.quantile(reverse,[0,.5,.95,1]).tolist(),
                binding_readback_max_m=error,correction_failure_m=.06,seconds=time.monotonic()-start,
                status='numeric_candidate_visual_and_containment_pending',
                limitations=['posed envelope can merge cloth layers at contact; it is not a physical simulation',
                             'coplanar overlap and complete containment not certified by triangle crossing count'])
    if max(forward.max(),reverse.max())>=.06 or len(pairs) or len(selfpairs) or error>2e-6:report['status']='failed_geometry'
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
