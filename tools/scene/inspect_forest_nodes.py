import bpy,json
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'outputs/cycling_forest'
def value(v):
    if hasattr(v,'name'):return v.name
    if hasattr(v,'__len__') and not isinstance(v,str):return list(v)
    return v
data={}
for name in ['path','river_bed','terrain_main']:
    o=bpy.data.objects[name]
    d={'location':list(o.location),'matrix':[list(r) for r in o.matrix_world]}
    if o.type=='CURVE':
        d['splines']=[{'type':s.type,'points':[list(p.co) for p in s.bezier_points],
                       'poly_points':[list(p.co) for p in s.points]} for s in o.data.splines]
    d['modifiers']=[{'name':m.name,'node_group':m.node_group.name if m.type=='NODES' else None,
                      'settings':{k:value(m[k]) for k in m.keys()} if m.type=='NODES' else {}} for m in o.modifiers]
    data[name]=d
data['groups']={}
for g in bpy.data.node_groups:
    data['groups'][g.name]={'nodes':[{'name':n.name,'type':n.bl_idname,'group':getattr(getattr(n,'node_tree',None),'name',None),
      'inputs':{i.name:value(i.default_value) for i in n.inputs if hasattr(i,'default_value') and not i.is_linked}} for n in g.nodes],
      'links':[(l.from_node.name,l.from_socket.name,l.to_node.name,l.to_socket.name) for l in g.links]}
(out/'source_nodes.json').write_text(json.dumps(data,indent=2,default=str))
print('NODE_INSPECTION_COMPLETE')
