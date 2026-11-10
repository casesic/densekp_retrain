"""Read same-distribution standard/LH model attributes for correspondence audit."""
import json
from pathlib import Path
import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/audits/bedlam_r138_compatibility_a2_20261009'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    r=np.load(ROOT/'assets/densekp_regressor_R.npy');support=np.flatnonzero(np.any(r!=0,axis=0));records=[]
    for label,filename in [('standard','smplx_model_20230302.blend'),('locked_head','smplx_model_lh_20230302.blend')]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        source=ROOT/'assets_external/smplx_blender_addon/data'/filename
        bpy.ops.wm.append(filename='SMPLX-mesh-neutral',directory=str(source)+'/Object/')
        ob=bpy.data.objects['SMPLX-mesh-neutral'];mesh=ob.data
        vertices=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',vertices.ravel())
        faces=np.array([list(p.vertices) for p in mesh.polygons],np.int32)
        uv=np.empty((len(mesh.uv_layers.active.data),2),np.float32);mesh.uv_layers.active.data.foreach_get('uv',uv.ravel())
        groups=np.array([g.name for g in ob.vertex_groups]);weights=np.zeros((len(vertices),len(groups)),np.float32)
        for v in mesh.vertices:
            for g in v.groups:weights[v.index,g.group]=g.weight
        shapes=[]
        for i in range(16):
            x=np.empty_like(vertices);mesh.shape_keys.key_blocks[f'Shape{i:03d}'].data.foreach_get('co',x.ravel());shapes.append(x-vertices)
        np.savez(OUT/(label+'_attributes.npz'),vertices=vertices,faces=faces,loop_uv=uv,weights=weights,group_names=groups,shape_first16=np.array(shapes),support=support)
        records.append(dict(label=label,source=str(source),object=ob.name,properties={k:ob[k] for k in ob.keys()},matrix_world=np.asarray(ob.matrix_world).tolist(),nonzero_shapes={k.name:k.value for k in mesh.shape_keys.key_blocks if k.value}))
    (OUT/'model_export.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
