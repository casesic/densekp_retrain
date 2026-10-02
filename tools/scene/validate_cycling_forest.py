"""Reopen audit: dependencies, road geometry and sampled 2.5m riding clearance.

Run against the saved adapted scene. Scatter is checked with viewport evaluation;
the report states this limit explicitly (render density may differ).
"""
import bpy
import json
import math
from pathlib import Path
from mathutils import Vector



def main():
    ROOT=Path(__file__).resolve().parents[2]
    OUT=ROOT/'outputs/scenes/cycling_forest'
    s=bpy.context.scene
    missing=[]
    for image in bpy.data.images:
        if image.source=='FILE' and not image.packed_file:
            p=Path(bpy.path.abspath(image.filepath))
            if not p.is_file():missing.append(str(p))
    road=bpy.data.objects['Asphalt_5m_continuous']
    verts=[road.matrix_world@v.co for v in road.data.vertices]
    widths=[(verts[i+1]-verts[i]).length for i in range(0,len(verts),2)]
    report={'file':bpy.data.filepath,'missing_textures':missing,
            'road_z_range':[min(v.z for v in verts),max(v.z for v in verts)],
            'road_width_range':[min(widths),max(widths)],
            'centerline_csv':str(OUT/'route_centerline.csv'),
            'clearance_test':'viewport dependency graph; 5 lateral positions every 2m, ray from Z=2.5m down',
            'ray_samples':0,'intrusions':[]}
    assert not missing,missing
    assert max(abs(v.z) for v in verts)<1e-5
    assert max(abs(w-5) for w in widths)<1e-4
    dg=bpy.context.evaluated_depsgraph_get()
    for y in range(-90,91,2):
        x=15+10*math.sin(y/32)
        n=Vector((1,-10/32*math.cos(y/32),0)).normalized()
        for lateral in [-2,-1,0,1,2]:
            origin=Vector((x,y,2.5))+lateral*n
            hit,loc,normal,index,obj,matrix=s.ray_cast(dg,origin,Vector((0,0,-1)),distance=2.6)
            report['ray_samples']+=1
            if not hit or loc.z>.04 or loc.z<-.02:
                report['intrusions'].append({'y':y,'lateral':lateral,'hit':bool(hit),
                  'object':obj.name if obj else None,'z':loc.z if hit else None})
    report['passed']=not missing and not report['intrusions']
    (OUT/'reopen_geometry_check.json').write_text(json.dumps(report,indent=2))
    print('CYCLING_SCENE_CHECK',json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
