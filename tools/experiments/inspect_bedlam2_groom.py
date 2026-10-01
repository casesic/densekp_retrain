"""Blender Alembic import audit; preserves raw curves without claiming a head binding."""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();records=[]
    for source in args.input:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.alembic_import(filepath=str(source.resolve()),as_background_job=False,set_frame_range=False)
        bpy.context.scene.frame_set(1)
        record=dict(source=str(source.resolve()),actual_bytes=source.stat().st_size,objects=[])
        points=[];radius=[];offsets=[0]
        for obj in bpy.context.scene.objects:
            info=dict(name=obj.name,type=obj.type,matrix_world=np.asarray(obj.matrix_world).tolist())
            record['objects'].append(info)
            matrix=np.asarray(obj.matrix_world,dtype=np.float32)
            if obj.type=='CURVES':
                data=obj.data
                p=np.empty((len(data.points),3),np.float32)
                data.position_data.foreach_get('vector',p.ravel())
                local_offsets=np.empty(len(data.curve_offset_data),np.int32)
                data.curve_offset_data.foreach_get('value',local_offsets)
                r=np.empty(len(data.points),np.float32)
                if 'radius' in data.attributes:data.attributes['radius'].data.foreach_get('value',r)
                else:r.fill(np.nan)
                info.update(curve_count=len(data.curves),point_count=len(data.points),
                            curve_start=len(offsets)-1,point_start=offsets[-1],
                            attributes=[dict(name=a.name,domain=a.domain,data_type=a.data_type) for a in data.attributes])
                points.append(p@matrix[:3,:3].T+matrix[:3,3]);radius.append(r)
                offsets.extend((local_offsets[1:]+offsets[-1]).tolist())
                continue
            if obj.type!='CURVE':
                info['status']='unsupported_by_this_curve_exporter';continue
            info['curve_count']=len(obj.data.splines)
            info['spline_types']=sorted({s.type for s in obj.data.splines})
            for spline in obj.data.splines:
                if spline.type=='BEZIER':raise ValueError('Bezier handle semantics need a separate exporter')
                p=np.empty((len(spline.points),4),np.float32)
                r=np.empty(len(spline.points),np.float32)
                spline.points.foreach_get('co',p.ravel())
                spline.points.foreach_get('radius',r)
                points.append(p[:,:3]@matrix[:3,:3].T+matrix[:3,3])
                radius.append(r);offsets.append(offsets[-1]+len(p))
        if points:
            points=np.concatenate(points);radius=np.concatenate(radius);offsets=np.asarray(offsets,np.int64)
            destination=args.output/(source.stem+'.npz')
            np.savez(destination,points=points,offsets=offsets,imported_radius=radius)
            record.update(export=str(destination.resolve()),curves=len(offsets)-1,points=len(points),
                          bounds=[points.min(0).tolist(),points.max(0).tolist()],
                          radius_range=[float(radius.min()),float(radius.max())],
                          status='imported_unbound_units_root_direction_and_width_semantics_unverified')
        else:record['status']='no_supported_curves'
        records.append(record)
    (args.output/'report.json').write_text(json.dumps(dict(blender=bpy.app.version_string,
        seconds=time.monotonic()-start,cases=records,split='integration_only',
        binding_status='pending_authoring_reference'),indent=2)+'\n')


if __name__=='__main__':main()
