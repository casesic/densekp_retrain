#!/usr/bin/env python3
"""CPU replay of existing EXR assets into a fresh directory, never a renderer.

Capture before migrating process_scene/warp_case. --before compares complete
artifacts and remap buffers; output roots and elapsed_seconds alone may differ.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'assets_external/python_deps')]
import cv2
import numpy as np
from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True)+'\n')


@contextmanager
def capture_remaps(directory):
    """Keep maps, interpolation/border choices and output *before* masking."""
    directory.mkdir()
    original = cv2.remap
    rows = []
    def capture(src, map1, map2, interpolation, **kwargs):
        result = original(src, map1, map2, interpolation, **kwargs)
        number = len(rows)
        for name, array in [('x',map1),('y',map2),('result',result)]:
            np.save(directory/f'{number:02}_{name}.npy', array)
        rows.append(dict(source_shape=list(src.shape), source_dtype=str(src.dtype),
                         interpolation=interpolation, kwargs=kwargs))
        return result
    with patch.object(cv2, 'remap', capture):
        yield
    write_json(directory/'calls.json', rows)


def link_source(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=True)


def run_main(module, args):
    with patch.object(sys, 'argv', [module.__file__, *map(str,args)]):
        module.main()


def replay(output):
    from tools.fisheye import run_ds_scene_pair_pilot as ds
    from tools.fisheye import run_fisheye_camera_p0 as omni
    from tools.fisheye import run_fisheye_multiperson_pilot as multi
    from tools.fisheye.build_omni_erp_pilot import build as to_erp
    from tools.annotation.build_omni_multiperson_tiny import build as pack
    from tools.validate.validate_ds_scene_pair_pilot import validate
    from densekp_retrain.camera import OmniRadtanIntrinsics
    from densekp_retrain.io import read_combined_depth_index
    ds_source=ROOT/'outputs/experiments/source_isolated_pilot_20260930/layouts/val_000'
    multi_source=ROOT/'outputs/experiments/fisheye_near_multiperson'
    single_source=ROOT/'outputs/baselines/fisheye_camera_p0'
    sources=[ds_source/'source/cam2.exr',ds_source/'source/cam4.exr',
             multi_source/'source/fisheye_center.exr',
             *[single_source/'equirectangular_source'/f'{name}.exr' for name in ('fisheye_center','cam_left','cam_right')]]
    raw=output/'raw_source'; raw.mkdir()
    for number,path in enumerate(sources):
        rgb,depth,instance,channels=read_combined_depth_index(path)
        np.savez(raw/f'{number}.npz',linear_rgb=rgb,source_erp_ray_range_m=depth,source_instance=instance)
        write_json(raw/f'{number}.json',dict(path=str(path),channels=channels))
    directory=output/'ds'; directory.mkdir()
    link_source(directory/'source',ds_source/'source')
    for name in ('rig.json','people.json'):
        shutil.copyfile(ds_source/name,directory/name)
    report=json.loads((ds_source/'report.json').read_text())
    poses={row['id']:np.asarray(row['camera_to_world']) for row in json.loads((directory/'rig.json').read_text())['cameras']}
    with np.load(ROOT/'assets_external/real_erp_valid_masks.npz') as z:
        masks={name:z[f'valid_masks_{name}'] for name in poses}
    lenses={name:np.asarray(Image.open(ds_source/name/'reference_lens_mask.png'))>0 for name in poses}
    metadata={row['camera']:{key:value for key,value in row['lens_mask'].items() if key!='removed_erp_valid_pixels'} for row in report['cameras']}
    with capture_remaps(output/'ds_remaps'):
        result=ds.process_scene(directory, report['people'],poses,masks,None,None,lens_masks=lenses,lens_metadata=metadata)
    report.update(result); write_json(directory/'report.json',report)
    validate(directory)
    # Full original CLI manifest and three fixed 3000-square cases, no render.
    directory=output/'omni_single'
    link_source(directory/'equirectangular_source',single_source/'equirectangular_source')
    with capture_remaps(output/'omni_single_remaps'):
        run_main(omni,['--skip-render','--output',directory])
    # Original multi-person source camera config recovered from its render manifest.
    directory=output/'omni_multi'
    link_source(directory/'source',multi_source/'source')
    source=json.loads((multi_source/'source/source_manifest.json').read_text())
    config=dict(camera_model='Omni-Radtan',intrinsics=OmniRadtanIntrinsics().as_dict(),
                cameras=[dict(id=row['id'],camera_to_world=row['camera_to_world']) for row in source['sources']])
    write_json(directory/'cameras.json',config)
    write_json(directory/'people.json',dict(people=source['people']))
    with capture_remaps(output/'omni_multi_remaps'):
        run_main(multi,['--skip-render','--output',directory,'--cameras',directory/'cameras.json',
                        '--people-config',directory/'people.json','--width','2048','--height','1024'])
    # Continue through unchanged fish->ERP and label packer to check supervision.
    with capture_remaps(output/'omni_erp_remaps'):
        to_erp(directory,output/'omni_erp','fisheye_center',1536,768)
    pack(output/'omni_erp',output/'omni_labels',val_person_id=3)
    write_json(output/'provenance.json',dict(ds=str(ds_source),omni_multi=str(multi_source),
        omni_single=str(single_source),depth_semantics={'ds':'raw source ERP ray range for visibility; no depth warp in process_scene',
        'omni':'nearest remapped source ray range, nonfinite/outside zeroed, fish 5x5 visibility'},
        render_executed=False))


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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--before',type=Path)
    args=parser.parse_args()
    if not args.before:
        from tools.fisheye.run_ds_scene_pair_pilot import process_scene
        from tools.fisheye.run_fisheye_camera_p0 import warp_case
        if not process_scene.__module__.startswith('tools.') or not warp_case.__module__.startswith('tools.'):
            raise RuntimeError('Independent before must be captured before migration')
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    replay(args.output)
    if args.before:
        result=compare(args.before.resolve(),args.output)
        write_json(args.output.parent/(args.output.name+'_comparison.json'),result)
        print(json.dumps(result['counts']))


if __name__=='__main__': main()
