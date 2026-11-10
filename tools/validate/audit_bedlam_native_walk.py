#!/usr/bin/env python3
"""Bounded native-pair audit. No body construction, retargeting or rendering.

Reads only declared candidates and extracts one indexed local NPZ. A missing
motion or model correspondence never falls back to another character.
"""
import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import resource
import tarfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/mnt/data1/yy/home/yy/datasets/BEDLAM')
AUDIT = ROOT / 'outputs/audits/cycling_character_feasibility_20260929'


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def index(path):
    return {name: int(size) for size, name in
            (line.split('\t', 1) for line in path.read_text().splitlines())}


def safe_member_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not p.parts:
        raise ValueError('Unsafe archive member name')
    return p


def extract_one(archive, member_name, declared_size, destination):
    safe_member_name(member_name)
    if not 0 < declared_size < 1024**3:
        raise ValueError('Extraction budget exceeded')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    for parent in [destination.parent, *destination.parent.parents]:
        if parent.is_symlink():
            raise ValueError('Symlink in destination path')
    start = time.monotonic()
    scanned = 0
    # r: on the known uncompressed TAR skips unrelated payloads by seeking.
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            scanned += 1
            if member.name != member_name:
                continue
            if not member.isfile() or member.issym() or member.islnk() or member.size != declared_size:
                raise ValueError('Unexpected type/link/size for selected member')
            count = 0
            with tar.extractfile(member) as src, destination.open('xb') as dst:
                while chunk := src.read(1024 * 1024):
                    count += len(chunk)
                    if count > declared_size:
                        raise ValueError('Payload exceeded indexed size')
                    dst.write(chunk)
            if count != declared_size:
                raise ValueError('Incomplete payload')
            return dict(archive=str(archive), member=member_name, bytes=count,
                        output=str(destination), header_entries_visited=scanned,
                        seconds=time.monotonic()-start, regular_file_only=True,
                        extractall_used=False, payloads_extracted=1)
    raise FileNotFoundError(member_name)


def cloth_check(path):
    # This is a trusted, already-local official archive. Only its small metadata
    # object is unpickled; no remote or user-supplied pickle is accepted here.
    with np.load(path, allow_pickle=False) as z:
        v, f, uv, fu = (z[k] for k in ('vertices_seq', 'faces', 'vertices_uv', 'faces_uv'))
        checks = dict(vertices_finite=bool(np.isfinite(v).all()), uv_finite=bool(np.isfinite(uv).all()),
                      faces_valid=bool(f.min() >= 0 and f.max() < v.shape[1]),
                      uv_faces_valid=bool(fu.min() >= 0 and fu.max() < len(uv)),
                      same_face_count=f.shape == fu.shape,
                      topology_single_shared_array=True)
        if not all(checks.values()):
            raise ValueError(checks)
        summary = dict(path=str(path), bytes=path.stat().st_size,
                       vertices_shape=list(v.shape), faces_shape=list(f.shape),
                       uv_shape=list(uv.shape), checks=checks,
                       all_frame_bounds_min=v.min(axis=(0,1)).tolist(),
                       all_frame_bounds_max=v.max(axis=(0,1)).tolist(),
                       coordinate_units='not inferred from bounds',
                       body_pair_validated=False, ordinary_walking_verified=False)
    with np.load(path, allow_pickle=True) as z:
        summary['metadata'] = z['metadata'].item()
    return summary


def motion_check(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(path=str(path), bytes=path.stat().st_size,
                    shapes={k:list(z[k].shape) for k in z.files},
                    finite={k:bool(np.isfinite(z[k]).all()) for k in ('poses','trans','betas')},
                    fps=float(z['mocap_frame_rate']), gender=str(z['gender']),
                    betas=z['betas'].tolist(),
                    motion_info=z['motion_info'].tolist() if 'motion_info' in z.files else None,
                    source_translation_min=z['trans'].min(0).tolist(),
                    source_translation_max=z['trans'].max(0).tolist(),
                    body_evaluated=False, dynamic_review='not_performed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'candidate_freeze.json').exists():
        raise FileExistsError('Use a new output directory')
    start=time.monotonic();cpu=time.process_time()
    members=index(AUDIT/'clothing_npz_6fps_members.txt')
    textures=index(AUDIT/'clothing_textures_members.txt')
    member='rp_aaron_posed_009/clothing_simulations/1033/1033.npz'
    freeze=dict(candidates=['BEDLAM1/rp_aaron_posed_009/1095',
                            'BEDLAM1/rp_aaron_posed_009/1033','BEDLAM2/us_2495_L/2200'],
                reasons=['Known complete local pair; verify action mismatch',
                         'First indexed simulation for same body with existing textures; motion/action may be absent',
                         'Existing ordinary-walk-named motion; verify original outfit and model availability'],
                extracted_members=[dict(name=member,bytes=members[member])],
                maximum_extracted_and_derived_bytes=1024**3,
                no_selected_preview_interval_until_pair_complete=True)
    write(out/'candidate_freeze.json',freeze)
    result=extract_one(DATA/'clothing_data/clothing_npz_6fps/clothing_npz_6fps.0.tar',
                       member,members[member],out/'extracted'/member)
    write(out/'extraction.json',result)
    clothing={}
    for mid,path in [('1095',DATA/'clothing_data/tiny/rp_aaron_posed_009/clothing_simulations/1095/1095.npz'),
                     ('1033',out/'extracted'/member)]:
        clothing[mid]=cloth_check(path)
    source=ROOT/'outputs/experiments/source_isolated_pilot_20260930/sources'
    motions={'1095':motion_check(DATA/'motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz'),
             'us_2495_L_2200':motion_check(source/'us_2495_L_2200.npz')}
    with (source/'b2_bedlam2amass_motion_training.csv').open() as handle:
        mapping=next(r for r in csv.DictReader(handle) if r['body_id']=='us_2495_L' and r['motion_id']=='2200')
    config=ROOT/'third_party/bedlam2_render/config'
    outfits=json.loads((config/'motion2outfit_training.json').read_text())
    outfit=outfits['motions']['us_2495_L']['2200']
    expected_paths=[
        DATA/'motion_data/tiny/rp_aaron_posed_009/1033/motion_seq.npz',
        DATA/'motion_data/gendered_ground_truth/rp_aaron_posed_009/1033/motion_seq.npz',
        DATA/'bedlam_data/smplx_gt/gendered_ground_truth/rp_aaron_posed_009/1033/motion_seq.npz',
        ROOT/'third_party/bedlam_clothing/support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz',
        ROOT/'assets_external/smplx_blender_addon/data/smplx_model_lh_20230302.blend',
        ROOT/'outputs/experiments/cycling_commuter_pilot_20260930/models/SMPLX_MALE.npz',
        Path('/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/PEAR/assets/models_smplx_v1_1.zip'),
    ]
    checks=[dict(path=str(p),exists=p.exists(),bytes=p.stat().st_size if p.is_file() else None) for p in expected_paths]
    texture_members=[dict(name=n,bytes=size) for n,size in textures.items()
                     if n.startswith('rp_aaron_posed_009/') and '/texture_01/' in n]
    body_dirs=set(n.split('/')[0] for n in members)
    write(out/'candidate_checks.json',dict(clothing=clothing,motions=motions,
        B2_source_mapping=mapping,B2_outfit=outfit,B2_texture_options=outfits['textures'][outfit],
        B2_body_in_B1_clothing_shard='us_2495_L' in body_dirs,
        texture_members=texture_members,explicit_path_checks=checks,
        limits=['No full-machine search','No motion/body/cache correspondence inferred from names',
                'Local locked-head .blend exists; compatibility not evaluated',
                'Model construction, R138 multiplication and rendering not performed']))
    write(out/'cpu_resources.json',dict(wall_seconds=time.monotonic()-start,
        cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        extraction_seconds=result['seconds'],extracted_bytes=result['bytes'],
        body_evaluation_seconds=0,render_seconds=0,gpu_used=False,failed_geometry_jobs=0))
    print(json.dumps(dict(extracted_bytes=result['bytes'],clothing_frames={k:v['vertices_shape'][0] for k,v in clothing.items()},
        B2_outfit=outfit,status='resource_checks_complete_not_native_pair_acceptance')))


if __name__=='__main__':main()
