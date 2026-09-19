"""Run in Blender with the downloaded scene open; writes a read-only inventory."""
import bpy
import json
from pathlib import Path

out=Path(__file__).resolve().parents[2]/'outputs/photoreal_hidden_alley'
out.mkdir(parents=True,exist_ok=True)
s=bpy.context.scene
report={
    'file':bpy.data.filepath,
    'scenes':[x.name for x in bpy.data.scenes],
    'active_scene':s.name,
    'objects':len(s.objects),
    'materials':len(bpy.data.materials),
    'mesh_vertices':sum(len(m.vertices) for m in bpy.data.meshes),
    'camera':s.camera.name if s.camera else None,
    'cameras':[{'name':o.name,'location':list(o.location),'rotation':list(o.rotation_euler),
                'lens':o.data.lens} for o in s.objects if o.type=='CAMERA'],
    'collections':[c.name for c in bpy.data.collections],
    'world':s.world.name if s.world else None,
    'engine':s.render.engine,
    'render':[s.render.resolution_x,s.render.resolution_y,s.render.resolution_percentage],
    'images':[{'name':i.name,'path':i.filepath,'packed':bool(i.packed_file),
               'exists':Path(bpy.path.abspath(i.filepath)).exists()} for i in bpy.data.images if i.source=='FILE'],
    'libraries':[{'path':l.filepath,'exists':Path(bpy.path.abspath(l.filepath)).exists()} for l in bpy.data.libraries],
}
(out/(Path(bpy.data.filepath).stem+'_inventory.json')).write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['images','collections']},indent=2))
print('MISSING_IMAGES',sum(not i['packed'] and not i['exists'] for i in report['images']))
