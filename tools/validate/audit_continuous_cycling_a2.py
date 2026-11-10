"""Bounded A2 read-only inventory. Prints JSON; never imports project/model code.

Reads named manifests, NPY/NPZ headers and small motion arrays only. No media
decode, model construction, geometry generation, hashes or recursive scan.
Run from the repository root; optional positional root overrides that location.
"""

import json
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
EXT = Path('/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch')


def read(path):
    return json.loads(Path(path).read_text())


def header(stream):
    version = np.lib.format.read_magic(stream)
    shape, order, dtype = np.lib.format._read_array_header(stream, version)
    return dict(shape=list(shape), dtype=str(dtype), fortran_order=order)


def inspect(path):
    path = Path(path)
    result = dict(path=str(path), exists=path.is_file())
    if not result['exists']:
        return result
    result['bytes'] = path.stat().st_size
    if path.suffix == '.npy':
        with path.open('rb') as stream:
            result.update(header(stream))
    elif path.suffix == '.npz':
        with zipfile.ZipFile(path) as archive:
            result['arrays'] = {}
            for name in archive.namelist():
                if name.endswith('.npy'):
                    with archive.open(name) as stream:
                        result['arrays'][name[:-4]] = header(stream)
    return result


def motion(path, key):
    result = inspect(path)
    if result['exists']:
        with np.load(path, allow_pickle=False) as archive:
            value = archive[key]
            result.update(frames=len(value), finite=bool(np.isfinite(value).all()),
                          changing_adjacent_pairs=int(np.count_nonzero(
                              np.any(np.diff(value, axis=0) != 0,
                                     axis=tuple(range(1, value.ndim))))))
            for name in ('fps', 'mocap_frame_rate', 'frame_time', 'sequence'):
                if name in archive:
                    result[name] = archive[name].item()
            if 'source_frame_ids' in archive:
                ids = archive['source_frame_ids']
                result['source_frame_range_inclusive'] = [int(ids[0]), int(ids[-1])]
                result['source_ids_contiguous'] = bool(np.all(np.diff(ids) == 1))
    return result


def main():
    base = ROOT / 'outputs/experiments/cycling_ten_adults_candidates_20260930'
    characters = read(base / 'character_manifest.json')['characters']
    combinations = read(base / 'combination_status.json')
    result = dict(schema='continuous_cycling_a2_readonly_evidence_v1',
                  evidence_level='headers_small_arrays_manifests_not_dynamic_acceptance',
                  characters=[], static_photos=[], native_cycling=[], pedestrian=[])
    for c in characters:
        m = read(c['manifest'])
        ref = m['reference']
        groom = Path(ref['groom_geometry'])
        gm = read(groom / 'groom_manifest.json')
        files = [Path(ref['geometry']) / n for n in ('vertices.npy', 'faces.npy', 'densekp_3d.npy')]
        files += [Path(ref['clothing_geometry']) / n for n in
                  ('clothing_vertices.npy', 'clothing_faces.npy', 'body_barycentric_indices.npy',
                   'body_barycentric_weights.npy', 'reference_body_vertices.npy')]
        files += [groom / n for n in ('points.npy', 'curve_offsets.npy', 'radius.npy',
                                     'active_curve_mask.npy', 'collision_correction.npy')]
        files += [Path(gm['binding']) / 'binding.npz', Path(gm['source_abc']), Path(c['model_file'])]
        files += [Path(c['appearance'][n]) for n in ('body_texture', 'body_uv', 'eye_mapping')]
        result['characters'].append(dict(
            asset_id=c['asset_id'], identity_family_id=c['identity_family_id'],
            gender=c['gender'], betas_count=len(c['betas']),
            manifest=c['manifest'], reference={k:ref[k] for k in
                ('geometry', 'clothing_geometry', 'groom_geometry')},
            groom_source=gm['source_abc'], binding=gm['binding'],
            accepted_static_poses=[r['pose_source_id'] for r in combinations['accepted']
                                   if r['character_id'] == c['asset_id']],
            dynamic_acceptance=False, files=[inspect(p) for p in files]))
    result['combination_counts'] = {k:len(combinations[k]) for k in
        ('accepted', 'not_attempted', 'excluded_ds_rows', 'failed_appearance_versions')}
    for name in ('coco_292226_person_1330527', 'coco_306800_person_225160'):
        result['static_photos'].append(motion(
            ROOT / 'outputs/experiments/cycling_smplx_20260929' / name / 'canonical_pose.npz', 'body_pose'))
    for name in ('participant_1', 'participant_2'):
        p = ROOT / 'outputs/motion_inputs/cycling_fit_20260929' / name
        item = motion(p / 'clip_native.npz', 'channels_native')
        item['historical_kinematic_review_not_rerun'] = read(p / 'kinematic_review.json')
        result['native_cycling'].append(item)
    for p in (ROOT / 'outputs/experiments/source_isolated_pilot_20260930/sources/us_2495_L_2200.npz',
              Path('/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz'),
              ROOT / 'third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz'):
        result['pedestrian'].append(motion(p, 'poses'))
    p = EXT / 'data_process/tmp/production_regression_20260320_dataset3_multiperson/standalone_sequence/sequence_result.json'
    seq = read(p)
    tracks = {}
    keys = set()
    for frame in seq['frames']:
        for person in frame['persons']:
            tracks.setdefault(person['track_id'], []).append(frame['frame'])
            keys.update(person)
    rows = []
    for tid, frames in tracks.items():
        run = best = 1
        for a, b in zip(frames, frames[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        rows.append(dict(track_id=tid, observations=len(frames), first=frames[0],
                         last=frames[-1], gap_count=sum(b-a != 1 for a,b in zip(frames,frames[1:])),
                         longest_contiguous_frames=best))
    result['external_sequence'] = dict(path=str(p), frames=seq['num_frames'],
        observations=seq['matched_person_observations'], algorithm_tracks=len(tracks),
        saved_person_keys=sorted(keys), tracks=sorted(rows, key=lambda x:-x['observations']),
        pose_beta_root_export_present=False, real_gt=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
