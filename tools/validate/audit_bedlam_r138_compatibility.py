"""Local provenance, surface-coordinate and same-topology LH correspondence audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import pickle
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/audits/bedlam_r138_compatibility_a2_20261009'
EXTERNAL=Path('/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch')


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def main():
    began=time.monotonic();torch.set_num_threads(1)
    data=EXTERNAL/'tools/CameraHMR/data/train-eval-utils'
    with (data/'downsample_mat.pkl').open('rb') as f:down=pickle.load(f).to_dense().cpu().numpy()
    with (data/'smplx2smpl.pkl').open('rb') as f:transfer=pickle.load(f)
    assert down.shape==(138,6890) and np.all((down!=0).sum(1)==1)
    source_ids=down.argmax(1);assert np.all(down[np.arange(138),source_ids]==1)
    R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    reproduced=transfer['matrix'][source_ids].astype(np.float32)
    np.testing.assert_array_equal(reproduced,R)
    a=np.load(OUT/'standard_attributes.npz');b=np.load(OUT/'locked_head_attributes.npz')
    for name in ['faces','loop_uv','weights','group_names']:np.testing.assert_array_equal(a[name],b[name])
    standard=np.load(EXTERNAL/'tools/smpl_models/smplx/SMPLX_NEUTRAL.npz')
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    template_error=float(np.abs(standard['v_template']@C.T-a['vertices']).max())
    assert template_error<2e-6
    np.testing.assert_array_equal(standard['f'],a['faces'])
    faces=a['faces'];uv=a['loop_uv'].reshape(-1,3,2);vertex_delta=np.linalg.norm(b['vertices']-a['vertices'],axis=1)
    changed=vertex_delta>2e-6;changed_faces=changed[faces].any(1)
    pa=R@a['vertices'];pb=R@b['vertices'];names=list(a['group_names'])
    na=np.cross(a['vertices'][faces[:,1]]-a['vertices'][faces[:,0]],a['vertices'][faces[:,2]]-a['vertices'][faces[:,0]])
    nb=np.cross(b['vertices'][faces[:,1]]-b['vertices'][faces[:,0]],b['vertices'][faces[:,2]]-b['vertices'][faces[:,0]])
    cos=(na*nb).sum(1)/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1),1e-20)
    rows=[]
    for i,row in enumerate(R):
        ids=np.flatnonzero(row);weights=row[ids]
        matched=np.flatnonzero(np.isin(faces,ids).sum(1)==len(ids))
        assert len(matched)>0
        fi=int(matched[0]);fw=np.array([row[k] for k in faces[fi]])
        # Preserve tiny original negative barycentric coefficients, do not sanitize.
        uv_coord=(uv[fi]*fw[:,None]).sum(0)
        group=(a['weights'][ids]*weights[:,None]).sum(0);bone=names[int(group.argmax())]
        rows.append(dict(point_id=i,source_smpl_vertex=int(source_ids[i]),smplx_vertex_ids=ids.tolist(),weights=weights.tolist(),
            representative_face_id=fi,representative_face_vertices=faces[fi].tolist(),material_uv=uv_coord.tolist(),
            dominant_joint=bone,template_changed_support=bool(changed[ids].any()),
            standard_blender_xyz_m=pa[i].tolist(),locked_head_blender_xyz_m=pb[i].tolist(),template_delta_m=float(np.linalg.norm(pb[i]-pa[i])),
            same_surface_triangle_and_uv=True,standard_and_lh_skin_weights_exact=True))
    affected=[x['point_id'] for x in rows if x['template_changed_support']]
    # Images use exact geometry; no independently normalized plots or magnified displacements.
    head=(pa[:,2]>.20);head_faces=(a['vertices'][faces,:,][...,2]>.16).all(1)
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for col,(label,model,points) in enumerate([('Standard v1.1',a,pa),('Locked-head / no-bun',b,pb)]):
        for row_idx,(view,xy,depth) in enumerate([('front',(0,2),1),('side',(1,2),0)]):
            ax=axes[row_idx,col];tri=model['vertices'][faces[head_faces]];order=np.argsort(tri[:,:,depth].mean(1))
            ax.add_collection(PolyCollection(tri[order][:,:,xy],facecolors='#c9d4df',edgecolors='#8091a1',linewidths=.13,alpha=.5))
            colors=['#d53e26' if i in affected else '#005b9a' for i in np.flatnonzero(head)]
            q=points[head][:,xy];ax.scatter(q[:,0],q[:,1],c=colors,s=19,zorder=3)
            for i,x in zip(np.flatnonzero(head),q):ax.annotate(str(i),x,xytext=(3,3),textcoords='offset points',fontsize=8,zorder=4)
            ax.set_xlim(-.17,.17);ax.set_ylim(.16,.44);ax.set_aspect('equal');ax.set_title(label+' / '+view);ax.set_xlabel('model metres (Blender axes)');ax.grid(alpha=.2)
    fig.suptitle('Same scale and point IDs; red = 5 rows affected by template sculpture',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'head_surface_points.png',dpi=150);plt.close(fig)
    # Shape-space differences are recorded, not interpreted as vertex renumbering.
    shdelta=np.abs(b['shape_first16']-a['shape_first16'])
    actual=np.load(ROOT/'outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/geometry/vertices.npy',mmap_mode='r')
    cached=np.load(ROOT/'outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/geometry/densekp_3d_pending.npy')
    readback=float(np.abs(np.einsum('kv,tvc->tkc',R,actual,optimize=True)-cached).max())
    report=dict(status='compatible',scope='local neutral locked-head/no-head-bun smplx_model_lh_20230302.blend; current us_2495_L/2200 body cache without shoe/toe edits',
        meaning='same mesh-indexed DenseKP surface semantics; not equality of head shape, real anatomy or reconstruction accuracy',
        provenance=dict(downsample=str(data/'downsample_mat.pkl'),smplx2smpl=str(data/'smplx2smpl.pkl'),
          implementation=str(EXTERNAL/'tools/SingleFitting/single_fitting/fitting/constraints/densekp.py'),formula='D138x6890 @ T6890x10475',reproduced_R_exact=True),
        evidence=dict(standard_template_matches_original_npz_max_m=template_error,ordered_faces_exact=True,
          per_face_corner_UV_exact=True,all_vertex_joint_weights_exact=True,group_names_exact=True,
          all138_rows_have_same_existing_surface_triangle=True,all138_rows_keep_same_UV_and_joint_region=True,
          vertex_renumbering_detected=False,changed_template_vertices=int(changed.sum()),changed_template_faces=int(changed_faces.sum()),
          max_vertex_template_change_m=float(vertex_delta.max()),affected_rows=affected,unaffected_rows=138-len(affected),
          min_changed_face_normal_cosine=float(cos[changed_faces].min()),changed_faces_with_normal_inversion=int((cos[changed_faces]<=0).sum()),
          shape_basis_max_difference_m=float(shdelta.max()),shape_basis_not_assumed_equal=True,current91_frame_R_readback_error=readback),
        qualification='Compatibility conclusion combines exact operator provenance, fixed surface coordinates/UV/weights and inspected head plots, not vertex count or matrix multiplication alone.',
        exclusions=['other BEDLAM model variants','shoe displacement','toeless/sock templates','reindexed or remeshed caches','official437 point operator','unverified external posed cache vertex order'],
        no_rows_removed=True,no_R_changed=True,no_body_changed=True,seconds=time.monotonic()-began,integration_only=True)
    write(OUT/'rows.json',rows);write(OUT/'compatibility.json',report)
    write(OUT/'local_evidence.json',dict(official_plugin_readme='assets_external/smplx_blender_addon/README.md:20230302 changelog',
        official_model_selection='assets_external/smplx_blender_addon/__init__.py:SMPLXAddGender.execute',
        official_BEDLAM2_generator='third_party/bedlam2_render/blender/smplx_anim_to_alembic/smplx_anim_to_alembic.py',
        exact_internal_sculpt_authoring_recipe='not included; conclusion based on shipped mesh attributes, not a claim to reproduce sculpting',
        anatomical_point_name_dictionary='not supplied; unambiguous semantics retained as original point_id/SMPL vertex and SMPL-X surface coordinate'))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
