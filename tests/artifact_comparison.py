"""Exact comparison of bounded test output directories, including partial outputs."""

import json

import numpy as np
from PIL import Image


def normalize(value, root):
    if isinstance(value,dict):
        return {k:normalize(v,root) for k,v in value.items() if k!='elapsed_seconds'}
    if isinstance(value,list): return [normalize(v,root) for v in value]
    if isinstance(value,str): return value.replace(str(root.resolve()),'<OUTPUT>')
    return value

def compare(before, after):
    def paths(root):
        return {p.relative_to(root) for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
    assert paths(before)==paths(after), (paths(before)-paths(after),paths(after)-paths(before))
    counts=dict(arrays=0,images=0,json=0,other=0); files=[]
    def array(a,b,key):
        assert a.dtype==b.dtype and a.shape==b.shape,key
        if a.dtype.kind in 'USO':
            assert normalize(a.tolist(),before)==normalize(b.tolist(),after),key
        else: np.testing.assert_array_equal(a,b,err_msg=key)
        counts['arrays']+=1
    for relative in sorted(paths(before)):
        a,b=before/relative,after/relative
        if a.suffix=='.npy': array(np.load(a),np.load(b),str(relative))
        elif a.suffix=='.npz':
            with np.load(a) as aa,np.load(b) as bb:
                assert aa.files==bb.files,str(relative)
                for key in aa.files: array(aa[key],bb[key],f'{relative}:{key}')
        elif a.suffix in ('.png','.jpg'):
            aa,bb=np.asarray(Image.open(a)),np.asarray(Image.open(b))
            assert aa.dtype==bb.dtype and aa.shape==bb.shape,str(relative)
            np.testing.assert_array_equal(aa,bb,err_msg=str(relative));counts['images']+=1
            assert a.read_bytes()==b.read_bytes(),str(relative)
        elif a.suffix=='.json':
            aa=json.dumps(normalize(json.loads(a.read_text()),before),sort_keys=False,allow_nan=True)
            bb=json.dumps(normalize(json.loads(b.read_text()),after),sort_keys=False,allow_nan=True)
            assert aa==bb,str(relative);counts['json']+=1
        else: assert a.read_bytes()==b.read_bytes(),str(relative);counts['other']+=1
        files.append(str(relative))
    return dict(exact=True,counts=counts,files=files,allowed_changes=['output root','elapsed_seconds'])
