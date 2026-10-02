"""Run inside Blender: observe actual assembly immediately before rendering.

--mode assembly stops at the first render call; --mode render also executes the
unmodified renderer. All observations go into a fresh acceptance directory.
Neither mode saves a blend file or alters camera/device/seed/material settings.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import bpy
import numpy as np


def scalars(owner):
    result={}
    for prop in owner.bl_rna.properties:
        if prop.identifier in ('rna_type',) or prop.type not in ('BOOLEAN','INT','FLOAT','STRING','ENUM'): continue
        try:
            value=getattr(owner,prop.identifier)
            if isinstance(value,set):value=sorted(value)
            elif getattr(prop,'is_array',False):value=list(value)
            json.dumps(value,allow_nan=True);result[prop.identifier]=value
        except (AttributeError,TypeError,ValueError):pass
    return result


def tree_record(tree):
    if tree is None:return None
    nodes=[]
    for node in tree.nodes:
        row=dict(type=node.bl_idname,name=node.name,settings=scalars(node),inputs=[])
        for socket in node.inputs:
            value=getattr(socket,'default_value',None)
            try:
                if value is not None and not isinstance(value,(str,int,float,bool)):value=list(value)
                json.dumps(value)
            except TypeError:value=None
            row['inputs'].append(dict(name=socket.name,identifier=socket.identifier,value=value))
        if getattr(node,'image',None):
            image=node.image;row['image']=dict(path=image.filepath,name=image.name,colorspace=image.colorspace_settings.name)
        nodes.append(row)
    return dict(nodes=nodes,links=[(l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in tree.links])


def snapshot(output, number):
    scene=bpy.context.scene
    depsgraph=bpy.context.evaluated_depsgraph_get()
    arrays={};objects=[];materials={}
    for obj in scene.objects:
        row=dict(name=obj.name,type=obj.type,matrix_world=np.asarray(obj.matrix_world).tolist(),
                 pass_index=obj.pass_index,hide_render=obj.hide_render,hide_viewport=obj.hide_viewport,
                 materials=[slot.material.name if slot.material else None for slot in obj.material_slots])
        if obj.type in ('CAMERA','LIGHT'):row['data']=scalars(obj.data)
        selected=obj.name.startswith(('SMPLX','Clothing','Person'))
        if selected and obj.type=='MESH':
            evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
            coordinates=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',coordinates.ravel())
            matrix=np.asarray(evaluated.matrix_world,dtype=np.float64)
            arrays[obj.name+'/world']=coordinates@matrix[:3,:3].T+matrix[:3,3]
            arrays[obj.name+'/vertices']=coordinates
            arrays[obj.name+'/loop_vertices']=np.asarray([loop.vertex_index for loop in mesh.loops],np.int32)
            arrays[obj.name+'/polygon_vertices']=np.asarray([(p.loop_start,p.loop_total,p.material_index,p.use_smooth) for p in mesh.polygons],np.int32)
            for layer in mesh.uv_layers:arrays[obj.name+'/uv/'+layer.name]=np.asarray([v.uv[:] for v in layer.data],np.float32)
            for attr in mesh.attributes:
                if attr.name=='GarmentRest':arrays[obj.name+'/GarmentRest']=np.asarray([v.vector[:] for v in attr.data],np.float32)
            evaluated.to_mesh_clear()
        if selected and obj.type=='CURVES':
            evaluated=obj.evaluated_get(depsgraph);data=evaluated.data
            coordinates=np.empty((len(data.points),3),np.float32);data.position_data.foreach_get('vector',coordinates.ravel())
            matrix=np.asarray(evaluated.matrix_world,dtype=np.float64)
            arrays[obj.name+'/world']=coordinates@matrix[:3,:3].T+matrix[:3,3]
            arrays[obj.name+'/radius']=np.asarray([v.value for v in data.attributes['radius'].data],np.float32)
            arrays[obj.name+'/curves']=np.asarray([(c.first_point_index,c.points_length) for c in data.curves],np.int32)
        if selected:
            for slot in obj.material_slots:
                if slot.material:materials[slot.material.name]=dict(settings=scalars(slot.material),tree=tree_record(slot.material.node_tree))
        objects.append(row)
    compositor=scene.compositing_node_group if hasattr(scene,'compositing_node_group') else scene.node_tree
    record=dict(blender_version=bpy.app.version_string,build_hash=bpy.app.build_hash.decode(),camera=scene.camera.name,
                render=scalars(scene.render),cycles=scalars(scene.cycles),view_settings=scalars(scene.view_settings),
                image_settings=scalars(scene.render.image_settings),frame=scene.frame_current,
                view_layers=[scalars(layer) for layer in scene.view_layers],
                world=tree_record(scene.world.node_tree),compositor=tree_record(compositor),objects=objects,materials=materials)
    np.savez(output/f'assembly_{number}.npz',**arrays)
    (output/f'assembly_{number}.json').write_text(json.dumps(record,indent=2,allow_nan=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['assembly','render'],required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    case=json.loads(args.case.read_text());script=Path(case['script']);renderer_args=[v.replace('<OUTPUT>',str(args.output/'source')) for v in case['args']]
    count=0;observed=[]
    class AssemblyComplete(Exception):pass
    def trace(frame,event,arg):
        nonlocal count
        if event=='line':
            # Original and extracted implementations both retain the bpy render call.
            import linecache
            line=linecache.getline(frame.f_code.co_filename,frame.f_lineno)
            if 'bpy.ops.render.render(' in line:
                sys.settrace(None)
                snapshot(args.output,count);observed.append(dict(file=frame.f_code.co_filename,line=frame.f_lineno));count+=1
                if args.mode=='assembly':raise AssemblyComplete()
                sys.settrace(trace)
        return trace
    old_argv=sys.argv;sys.argv=[str(script),'--',*renderer_args]
    try:
        sys.settrace(trace);runpy.run_path(str(script),run_name='__main__')
    except AssemblyComplete:pass
    finally:sys.settrace(None);sys.argv=old_argv
    if not count:raise RuntimeError('No renderer boundary observed')
    (args.output/'observation.json').write_text(json.dumps(dict(mode=args.mode,render_calls=count if args.mode=='render' else 0,snapshots=count,observed=observed),indent=2)+'\n')

if __name__=='__main__':main()
