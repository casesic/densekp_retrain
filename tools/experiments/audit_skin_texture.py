#!/usr/bin/env python3
"""Record official texture provenance and component/eye UV compatibility evidence."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from audit_body_uv import read_obj
root=Path(__file__).resolve().parents[2]; out=root/'outputs/body_texture_audit'
obj=Path('/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/PEAR/assets/SMPLX/smplx_uv.obj')
v,u,f,t,*_=read_obj(obj)
geometry=root/'outputs/p0_dressed_dataset/geometry'
assert np.array_equal(f,np.load(geometry/'faces.npy'))
assert np.isfinite(u).all() and t.min()>=0 and t.max()<len(u)
n,labels=connected_components(coo_matrix((np.ones(f.size),(f.ravel(),np.roll(f,1,axis=1).ravel())),shape=(len(v),len(v))),directed=False)
model=np.load(Path(json.loads((geometry/'geometry_manifest.json').read_text())['model_dir'])/'SMPLX_MALE.npz',allow_pickle=True)
assert np.array_equal(model['f'],f)
components=[]
for c in range(n):
    ids=np.where(labels==c)[0]; fi=np.where(labels[f[:,0]]==c)[0]; uv=u[t[fi]]; w=model['weights'][ids].mean(0); j=int(w.argmax())
    components.append(dict(component=c,vertices=len(ids),faces=len(fi),vertex_index_range=[int(ids.min()),int(ids.max())],dominant_joint=j,dominant_joint_mean_weight=float(w[j]),region={23:'left_eye_smplhf',24:'right_eye_smplhf'}.get(j,'body'),uv_min=uv.min((0,1)).tolist(),uv_max=uv.max((0,1)).tolist()))
np.savez(out/'body_uv.npz',uv=u,faces=f,faces_uv=t,component_labels=labels)
base=Path('/home/yy/datasets/BEDLAM/body_textures'); dest=base/'tiny/rp_aaron_posed_009'
z=zipfile.ZipFile(base/'bedlam_body_textures_meshcapade.zip'); assets=[]
for suffix in ['smpl/MC_texture_skintones/male/skin/skin_m_white_01_ALB.png','eye/SMPLX_eye.png']:
    member='bedlam_body_textures_meshcapade/'+suffix; data=z.read(member); p=dest/Path(member).name
    assert p.read_bytes()==data
    assets.append(dict(path=str(p),member=member,sha256=hashlib.sha256(data).hexdigest(),size=Image.open(p).size))
eye=np.asarray(Image.open(dest/'SMPLX_eye.png').convert('RGB')); yy,xx=np.where(eye.min(2)<100)
report=dict(body_texture_id='skin_m_white_01_ALB',selection='deterministic official male baseline requested by user',uv_obj=str(obj),topology_exact_match=True,components=components,assets=assets,eye_dark_pixels_uv_bounds=[float(xx.min()/eye.shape[1]),float(1-yy.max()/eye.shape[0]),float(xx.max()/eye.shape[1]),float(1-yy.min()/eye.shape[0])],eye_texture_integrated=False,eye_status='BLOCKED',eye_reason='Verified eye components use top-of-atlas UV islands; official eye iris is outside both islands. No verified official eye-specific UV assignment found locally. Preserve original eye material and UV; no guessed remapping.',body_material=dict(colorspace='sRGB',metallic=0,roughness=.58,alpha=1,normal_map=None,displacement=None))
(out/'skin_texture_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
