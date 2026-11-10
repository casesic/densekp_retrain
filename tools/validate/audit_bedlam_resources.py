#!/usr/bin/env python3
"""Inventory known BEDLAM inputs; read array headers, never construct geometry.

Outputs are research evidence, not motion quality or license approval. No network,
archive extraction, model loading, hashing, or disk-wide searches.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import zipfile

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'local_inventory.json'
    if target.exists():
        raise FileExistsError(target)
    config = root / 'third_party/bedlam2_render/config'
    source = root / 'outputs/experiments/source_isolated_pilot_20260930/sources'

    def rows(path: Path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    motions = rows(source / 'b2_bedlam2amass_motion_training.csv')
    stats_rows = rows(config.parent / 'stats/motion_stats_training.csv')
    stats = {(r['body_id'], r['motion_id']): r for r in stats_rows}
    outfits = json.loads((config / 'motion2outfit_training.json').read_text())
    whitelist = json.loads((config / 'whitelist_animations_training.json').read_text())
    starter = json.loads((config / 'whitelist_animations_starterpack.json').read_text())
    genders = {r['Name']: r['Gender'] for r in rows(config / 'gender_training.csv')}
    selected = []
    for row in motions:
        if 'motorcycle' not in row['motion_rel_fname'].lower():
            continue
        b, m = row['body_id'], row['motion_id']
        outfit = outfits['motions'].get(b, {}).get(m)
        selected.append({
            **row, 'body_gender_metadata': genders.get(b),
            'outfit_id': outfit,
            'texture_choices': outfits['textures'].get(outfit, []),
            'training_animation_whitelist': m in whitelist.get(b, []),
            'starterpack_animation_whitelist': m in starter.get(b, []),
            'official_stats': stats.get((b, m)),
            'source_group': row['ds_name'] + '/' + row['motion_rel_fname'],
            'indexed_duration_seconds': (float(row['orig_end']) - float(row['orig_start'])) / float(row['orig_fps']),
            'actual_motion_array_read': False,
            'dynamic_preview_reviewed': False,
            'clothing_realism_score': None,
            'bbox_or_foot_motion_inferred_from_coverage': False,
            'local_expected_source_npz': {
                'path': str(source / f'{b}_{m}.npz'),
                'exists': (source / f'{b}_{m}.npz').exists(),
            },
        })
    (out / 'motorcycle_candidates.json').write_text(json.dumps(selected, indent=2) + '\n')
    fields = list(motions[0]) + ['outfit_id', 'body_gender_metadata', 'starterpack_animation_whitelist', 'source_group']
    with (out / 'motorcycle_candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selected)

    dataset_root = Path('/mnt/data1/yy/home/yy/datasets/BEDLAM')
    paths = [
        source / 'us_2495_L_2200.npz',
        dataset_root / 'motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz',
        dataset_root / 'clothing_data/tiny/rp_aaron_posed_009/clothing_simulations/1095/1095.npz',
        dataset_root / 'clothing_data/clothing_npz_6fps/clothing_npz_6fps.0.tar',
        dataset_root / 'clothing_data/clothing_textures/clothing_textures.0.tar',
        dataset_root / 'body_textures/bedlam_body_textures_meshcapade.zip',
        root / 'assets_external/bedlam2_pilot/b2_groom_abc.tar.gz',
        root / 'assets_external/bedlam2_pilot/vinefx_groom_01-1_curly-M.abc',
        root / 'assets_external/bedlam2_pilot/vinefx_groom_18-1_short-bob-F.abc',
        root / 'outputs/experiments/cycling_ten_adults_candidates_20260930/local_groom_sources/vinefx_groom_03-1_short-casual-MF.abc',
        root / 'third_party/bedlam_clothing/support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz',
        root / 'third_party/bedlam_clothing/support_data/v_templates/smplx_neutral-lh_vtemplate_toeless.obj',
        root / 'third_party/bedlam_clothing/support_data/shoes/texture_maps',
        Path('/mnt/data1/yy/home/yy/datasets/BEDLAM2'),
    ]
    files = []
    for path in paths:
        record = {'path': str(path), 'exists': path.exists()}
        if path.is_file():
            record.update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            if path.suffix == '.npz':
                headers = {}
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        with archive.open(name) as handle:
                            version = np.lib.format.read_magic(handle)
                            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
                            headers[name] = {'shape': list(shape), 'dtype': str(dtype), 'fortran': fortran}
                record['array_headers_only'] = headers
                with np.load(path, allow_pickle=False) as arrays:
                    record['small_metadata'] = {
                        k: arrays[k].tolist() for k in
                        ('gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 'num_betas', 'motion_info')
                        if k in arrays.files
                    }
        files.append(record)
    archive_indices = []
    for name in ('clothing_npz_6fps_members.txt', 'clothing_textures_members.txt'):
        path = root / 'outputs/audits/cycling_character_feasibility_20260929' / name
        lines = path.read_text().splitlines()
        members = [line.split('\t', 1)[-1] for line in lines]
        archive_indices.append({'path': str(path), 'lines': len(lines),
                                'npz_members': sum(s.endswith('.npz') for s in members),
                                'obj_members': sum(s.endswith('.obj') for s in members),
                                'top_level_names': sorted({s.split('/')[0] for s in members})})
    inventory = {
        'date': '2026-10-09', 'scope': 'known local indexes and NPZ headers only',
        'not_a_quality_acceptance': True,
        'training_mapping_rows': len(motions),
        'unique_source_files': len({(r['ds_name'], r['motion_rel_fname']) for r in motions}),
        'dataset_mapping_row_counts': dict(collections.Counter(r['ds_name'] for r in motions)),
        'stats_rows': len(stats_rows),
        'starterpack_bodies': len(starter),
        'starterpack_body_motion_pairs': sum(map(len, starter.values())),
        'filename_substring_counts_not_action_labels': {
            term: sum(term in r['motion_rel_fname'].lower() for r in motions)
            for term in ('walk', 'run', 'jog', 'jump', 'dance', 'sit', 'stair', 'motorcycle', 'bicycle', 'cycling', 'bike')},
        'motorcycle_body_motion_pairs': len(selected),
        'motorcycle_unique_source_files': len({r['source_group'] for r in selected}),
        'files': files, 'existing_archive_member_indexes': archive_indices,
        'absence_scope': 'An absent explicit path does not assert absence from the whole machine.',
        'groom_inventory': str(root / 'assets_external/bedlam2_pilot/inventory.json'),
    }
    target.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'motion_rows': len(motions),
                      'motorcycle_pairs': len(selected), 'checked_paths': len(files)}))


if __name__ == '__main__':
    main()
