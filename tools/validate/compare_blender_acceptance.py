#!/usr/bin/env python3
"""Compare actual Blender assembly, EXR pass pixels and downstream artifacts exactly."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
import numpy as np
from PIL import Image


def normalize(value,root,parts=()):
    if isinstance(value,dict):
        return {k:normalize(v,root,parts+(k,)) for k,v in value.items()
                # Original single-line render writes filepath after our line hook;
                # extracted code has already written it. Destination only, no setting change.
                if parts+(k,) != ('render','filepath')}
    if isinstance(value,list):return [normalize(v,root,parts) for v in value]
    if isinstance(value,str):return value.replace(str(root.resolve()),'<OUTPUT>')
    return value


def differences(a,b,path=''):
    if type(a)!=type(b):return [dict(path=path,before=str(a)[:180],after=str(b)[:180])]
    if isinstance(a,dict):
        if list(a)!=list(b):return [dict(path=path,before_keys=list(a),after_keys=list(b))]
        return [d for k in a for d in differences(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,before_length=len(a),after_length=len(b))]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+'/'+str(i))]
    if a==b or isinstance(a,float) and np.isnan(a) and np.isnan(b):return []
    return [dict(path=path,before=a,after=b)]


def compare(before,after):
    files=lambda root:{p.relative_to(root) for p in root.rglob('*') if p.is_file() and p.name!='observation.json'}
    left,right=files(before),files(after)
    failures=[];counts=dict(arrays=0,json=0,images=0,exr_channels=0);details=[]
    if left!=right:failures.append(dict(files_before_only=sorted(map(str,left-right)),files_after_only=sorted(map(str,right-left))))
    def array(a,b,key,kind='arrays'):
        equal=a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
        row=dict(key=key,shape=list(a.shape),dtype=str(a.dtype),exact=equal)
        counts[kind]+=1
        if not equal:
            if a.shape==b.shape:
                finite=np.isfinite(a)&np.isfinite(b)
                row['max_abs_finite']=float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float)))) if finite.any() else None
                row['different_elements']=int((~((a==b)|(np.isnan(a)&np.isnan(b)))).sum())
            failures.append(row)
        details.append(row)
    for relative in sorted(left&right):
        p,q=before/relative,after/relative
        if p.suffix=='.npz':
            with np.load(p) as a,np.load(q) as b:
                if a.files!=b.files:failures.append(dict(file=str(relative),keys_before=a.files,keys_after=b.files));continue
                for key in a.files:array(a[key],b[key],f'{relative}:{key}')
        elif p.suffix=='.npy':array(np.load(p),np.load(q),str(relative))
        elif p.suffix=='.json':
            diff=differences(normalize(json.loads(p.read_text()),before),normalize(json.loads(q.read_text()),after))
            if diff:failures.append(dict(file=str(relative),differences=diff))
            counts['json']+=1
        elif p.suffix in ('.png','.jpg'):
            array(np.asarray(Image.open(p)),np.asarray(Image.open(q)),str(relative),'images')
        elif p.suffix=='.exr':
            import OpenEXR
            a,b=OpenEXR.File(str(p)),OpenEXR.File(str(q))
            if len(a.parts)!=len(b.parts):failures.append(dict(file=str(relative),parts_differ=True));continue
            for i,(pa,pb) in enumerate(zip(a.parts,b.parts)):
                if list(pa.channels)!=list(pb.channels):failures.append(dict(file=str(relative),channels_differ=True));continue
                for key in pa.channels:array(np.asarray(pa.channels[key].pixels),np.asarray(pb.channels[key].pixels),f'{relative}:part{i}:{key}','exr_channels')
        else:
            if p.read_bytes()!=q.read_bytes():failures.append(dict(file=str(relative),bytes_differ=True))
    return dict(exact=not failures,counts=counts,failures=failures,arrays=details,
                ignored=['acceptance output root','assembly render.filepath (line-hook timing only)','observation.json code locations'],
                exr_comparison='All channel pixels/dtypes/shapes/part and channel order; binary container metadata not compared')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path);parser.add_argument('after',type=Path);parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args();result=compare(args.before.resolve(),args.after.resolve())
    args.report.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n');print(json.dumps(dict(exact=result['exact'],counts=result['counts'],failures=len(result['failures']))))
    raise SystemExit(not result['exact'])

if __name__=='__main__':main()
