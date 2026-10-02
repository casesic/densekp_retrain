#!/usr/bin/env python3
"""Three-level same-target occlusion experiment using the existing street pilot."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from tools.fisheye.run_ds_scene_pair_pilot import add_lens_mask_arguments, load_lens_masks
import numpy as np
from densekp_retrain.imaging.ds import process_scene
from densekp_retrain.body.transforms import world_points
from densekp_retrain.camera.double_sphere import DS_20260320
from densekp_retrain.runtime import load_appearance_baseline

LEVELS = ('clear', 'mild', 'heavy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/experiments/ds_scene_pair_20260928_final/urban_street_assets')
    parser.add_argument('--blender-bin', type=Path, default=Path('/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender'))
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--device', choices=['CPU', 'CUDA'], default='CUDA')
    parser.add_argument('--level', choices=LEVELS, nargs='+', default=list(LEVELS))
    parser.add_argument('--skip-render', action='store_true')
    add_lens_mask_arguments(parser)
    args = parser.parse_args()
    lens_masks, lens_metadata = load_lens_masks(args.fisheye_reference_root, args.reference_frame)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.source/'people.json').read_text())['people']
    rig = json.loads((args.source/'rig.json').read_text())
    if json.loads((args.source/'report.json').read_text())['scene'] != 'urban_street_assets':
        raise ValueError('This controlled layout requires the street pilot')
    if rig['intrinsics'] != {n:asdict(DS_20260320[n]) for n in ('cam2','cam4')}:
        raise ValueError('Source camera intrinsics differ from active DS parameters')
    poses = {c['id']:np.asarray(c['camera_to_world']) for c in rig['cameras']}
    geom = ROOT/'outputs/baselines/temporal_case_aaron_1095/geometry'
    clothing = geom.parent/'clothing_geometry'
    vertices = np.load(geom/'vertices.npy', mmap_mode='r')
    dense = np.load(geom/'densekp_3d.npy', mmap_mode='r')
    appearance, _ = load_appearance_baseline()
    with np.load(ROOT/'assets_external/real_erp_valid_masks.npz') as archive:
        masks = {n:archive[f'valid_masks_{n}'] for n in poses}
    scene = ROOT/'assets/scenes/photoreal/urban_street/urban_street_assets.blend'
    for level in args.level:
        people = copy.deepcopy(original)
        if level == 'clear':
            people = [p for p in people if p['id'] != 1]
        elif level == 'mild':
            next(p for p in people if p['id'] == 1)['location'][1] -= .30
        bounds = []
        for person in people:
            body = world_points(vertices[person['frame']], vertices[person['frame']], person)
            bounds.append((body.min(0), body.max(0)))
        for i, (low, high) in enumerate(bounds):
            for other_low, other_high in bounds[i+1:]:
                if not ((high+.05 < other_low) | (other_high+.05 < low)).any():
                    raise ValueError('Body AABBs lack 5cm separation')
        output = root/level
        output.mkdir(exist_ok=True)
        if args.skip_render:
            if json.loads((output/'people.json').read_text())['people'] != people:
                raise ValueError('Saved people differ')
            if json.loads((output/'rig.json').read_text()) != rig:
                raise ValueError('Saved rig differs')
        else:
            if (output/'source/source_manifest.json').exists():
                raise FileExistsError('Use a new output or --skip-render')
            (output/'people.json').write_text(json.dumps(dict(people=people), indent=2)+'\n')
            (output/'rig.json').write_text(json.dumps(rig, indent=2)+'\n')
            command = [str(args.blender_bin), '-b', '--disable-autoexec', '--python-exit-code', '1',
                       '--python', str(ROOT/'tools/fisheye/render_equirectangular_source.py'), '--',
                       '--geometry', str(geom), '--clothing-geometry', str(clothing),
                       '--cameras', str(output/'rig.json'), '--people-config', str(output/'people.json'),
                       '--scene-blend', str(scene), '--output-dir', str(output/'source'),
                       '--width', '1536', '--height', '768', '--samples', str(args.samples),
                       '--seed', '0', '--device', args.device,
                       '--body-texture', appearance['body_diffuse_texture_path'], '--body-uv', appearance['body_uv_path']]
            print(f'Rendering {level}, {args.samples} samples', flush=True)
            with (output/'render.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        source = json.loads((output/'source/source_manifest.json').read_text())
        if source['people'] != people or source['samples'] != args.samples or source['seed'] != 0 or source['animated_seed']:
            raise ValueError('Render settings/people do not match control')
        if source['scene_blend'] != str(scene.resolve()) or source['resolution'] != [1536,768]:
            raise ValueError('Rendered scene or resolution differs')
        for camera in source['sources']:
            np.testing.assert_allclose(camera['camera_to_world'], poses[camera['id']], atol=1e-12, rtol=0)
        result = process_scene(output, people, poses, masks, vertices, dense,
                               lens_masks=lens_masks, lens_metadata=lens_metadata)
        result.update(level=level, people=people, scene='urban_street_assets', samples=args.samples,
                      target_id=3, fixed_control_ids=[2,4], occluder_id=1,
                      intervention='remove occluder' if level == 'clear' else ('shift occluder world Y by -0.30m' if level == 'mild' else 'original occluder'),
                      baseline_m=rig['baseline_m'], baseline_status='synthetic, not measured',
                      source_pilot=str(args.source.resolve()), regressor_source='assets/densekp_regressor_R.npy')
        (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['cameras']), flush=True)


if __name__ == '__main__':
    main()
