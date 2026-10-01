"""Read the official locked-head neutral mesh as a groom reference, never a label body."""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--addon',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.addon/'data/smplx_model_lh_20230302.blend'
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.append(filename='SMPLX-mesh-neutral',directory=str(source.resolve())+'/Object/')
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    vertices=np.empty((len(obj.data.vertices),3),np.float32)
    obj.data.vertices.foreach_get('co',vertices.ravel())
    faces=np.array([list(p.vertices) for p in obj.data.polygons],np.int32)
    weights=np.zeros((len(vertices),len(obj.vertex_groups)),np.float32)
    for vertex in obj.data.vertices:
        for group in vertex.groups:weights[vertex.index,group.group]=group.weight
    np.savez(args.output/'reference.npz',vertices=vertices,faces=faces,weights=weights,
             group_names=np.array([g.name for g in obj.vertex_groups]),matrix_world=np.asarray(obj.matrix_world))
    info=dict(source=str(source.resolve()),actual_bytes=source.stat().st_size,
              object=obj.name,properties=dict(obj.items()),vertices=len(vertices),faces=len(faces),
              bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
              nonzero_shape_keys={k.name:k.value for k in obj.data.shape_keys.key_blocks if k.value},
              matrix_world=np.asarray(obj.matrix_world).tolist(),
              purpose='groom authoring reference only; DenseKP label body remains standard SMPL-X')
    (args.output/'report.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':main()
