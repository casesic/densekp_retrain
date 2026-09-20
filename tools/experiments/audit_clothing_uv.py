#!/usr/bin/env python3
"""全量核对 BEDLAM clothing NPZ 与参考 OBJ 的几何及 corner UV 索引。"""
import argparse, json
from pathlib import Path
import numpy as np

def read_obj(path):
    vertices=[]; uv=[]; faces=[]; faces_uv=[]; objects=[]; groups=[]; materials=[]
    for line in path.read_text().splitlines():
        fields=line.split()
        if not fields: continue
        if fields[0]=='v': vertices.append([float(x) for x in fields[1:4]])
        elif fields[0]=='vt': uv.append([float(x) for x in fields[1:3]])
        elif fields[0]=='f':
            corners=[x.split('/') for x in fields[1:]]
            if len(corners)!=3: raise RuntimeError('仅接受三角面 OBJ')
            faces.append([int(x[0])-1 for x in corners])
            if all(len(x)>=2 and x[1] for x in corners): faces_uv.append([int(x[1])-1 for x in corners])
        elif fields[0]=='o': objects.append(' '.join(fields[1:]))
        elif fields[0]=='g': groups.append(' '.join(fields[1:]))
        elif fields[0] in ('usemtl','mtllib'): materials.append(line)
    return map(np.asarray,(vertices,uv,faces,faces_uv)),objects,groups,materials

def main():
    p=argparse.ArgumentParser(); p.add_argument('--npz',type=Path,required=True); p.add_argument('--obj',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    (ov,ouv,of,ofu),objects,groups,materials=read_obj(a.obj)
    with np.load(a.npz,allow_pickle=True) as z: nv=np.asarray(z['vertices_seq'][0]); nuv=np.asarray(z['vertices_uv']); nf=np.asarray(z['faces']); nfu=np.asarray(z['faces_uv'])
    has_obj_uv=len(ouv)>0 and len(ofu)==len(of); direct=float(np.max(np.abs(ouv-nuv))) if has_obj_uv and ouv.shape==nuv.shape else None; flipped=float(np.max(np.abs(ouv-np.column_stack((nuv[:,0],1-nuv[:,1]))))) if direct is not None else None
    uv_indices_valid=bool(nfu.shape==nf.shape and nfu.min()>=0 and nfu.max()<len(nuv) and np.isfinite(nuv).all())
    report={'obj_vertices_shape':list(ov.shape),'obj_uv_shape':list(ouv.shape),'obj_faces_shape':list(of.shape),'obj_has_corner_uv':has_obj_uv,'npz_vertices_shape':list(nv.shape),'npz_uv_shape':list(nuv.shape),'npz_faces_shape':list(nf.shape),'faces_and_faces_uv_shape_match':bool(nf.shape==nfu.shape),'geometry_indices_valid':bool(nf.min()>=0 and nf.max()<len(nv)),'uv_index_valid':uv_indices_valid,'uv_range_u':[float(nuv[:,0].min()),float(nuv[:,0].max())],'uv_range_v':[float(nuv[:,1].min()),float(nuv[:,1].max())],'polygon_loop_count_match':bool(np.all(np.full(len(nf),3)==np.full(len(nfu),3))),'geometry_vertex_max_error':float(np.max(np.abs(ov-nv))),'geometry_faces_match':bool(np.array_equal(of,nf)),'uv_faces_match':bool(np.array_equal(ofu,nfu)) if has_obj_uv else None,'uv_face_match_rate':float(np.mean(np.all(ofu==nfu,axis=1))) if has_obj_uv else None,'uv_max_error_direct':direct,'uv_max_error_v_flipped':flipped,'v_flip_required':bool(flipped<direct) if direct is not None else None,'obj_objects':objects,'obj_groups':groups,'obj_material_statements':materials,'single_material_region':not objects and not groups and not materials}
    report['verified']=bool(report['faces_and_faces_uv_shape_match'] and report['geometry_indices_valid'] and report['uv_index_valid'] and report['polygon_loop_count_match'])
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
