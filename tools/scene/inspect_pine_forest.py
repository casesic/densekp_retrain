"""Inspect the source forest without changing it."""
import bpy
import json
from pathlib import Path
from mathutils import Vector
out=Path(__file__).resolve().parents[2]/'outputs/cycling_forest'
out.mkdir(parents=True,exist_ok=True)
scene=bpy.context.scene
report={'scenes':[s.name for s in bpy.data.scenes],'camera':scene.camera.name if scene.camera else None,
        'objects':[],'images':[],'materials':[m.name for m in bpy.data.materials],
        'world':scene.world.name if scene.world else None,
        'collections':[c.name for c in bpy.data.collections]}
for o in scene.objects:
    entry={'name':o.name,'type':o.type,'location':list(o.location),'dimensions':list(o.dimensions),
           'hidden':o.hide_render,'collections':[c.name for c in o.users_collection],
           'modifiers':[(m.name,m.type) for m in o.modifiers]}
    if o.type=='MESH':
        points=[o.matrix_world@Vector(v) for v in o.bound_box]
        entry['bounds']=[[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]]
        entry['vertices']=len(o.data.vertices)
        entry['materials']=[m.name for m in o.data.materials if m]
        entry['particle_systems']=[{'name':p.name,'count':p.settings.count,'instance':p.settings.instance_object.name if p.settings.instance_object else None,
              'collection':p.settings.instance_collection.name if p.settings.instance_collection else None,
              'density_group':p.vertex_group_density} for p in o.particle_systems]
    report['objects'].append(entry)
for i in bpy.data.images:
    if i.source=='FILE': report['images'].append({'name':i.name,'path':i.filepath,'packed':bool(i.packed_file)})
(out/'source_inventory.json').write_text(json.dumps(report,indent=2))
print('FOREST_INVENTORY',len(report['objects']),len(report['images']))
