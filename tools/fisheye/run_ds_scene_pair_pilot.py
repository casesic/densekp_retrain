#!/usr/bin/env python3
"""Small same-world, opposite-facing camera pilot in the two selected scenes.

20260320 intrinsics; same-height 180-degree opposing optical axes as specified
by the user. Baseline distance is an explicit synthetic pilot parameter, not a
claim about measured hardware. Produces diagnostic annotations, not train/val NPZ.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tools'),
                str(ROOT / 'assets_external/python_deps')]
import cv2
import numpy as np
from PIL import Image, ImageDraw
from densekp_retrain.camera.double_sphere import (
    DS_20260320, CAMERA_EXTRINSIC, project_double_sphere, unproject_double_sphere,
    build_erp_to_fisheye_lut, build_fisheye_to_source_erp_lut,
)
from densekp_retrain.io import read_combined_depth_index
from densekp_retrain.runtime import load_appearance_baseline
from tools.fisheye.build_ds_multiperson_pilot import remap_periodic_erp, lens_mask_from_reference
from tools.fisheye.run_fisheye_multiperson_pilot import world_points, project_erp, erp_visibility

SCENES = {
    'cycling_forest': ('configs/fisheye_scene_cycling_forest.json', [13.676, -4.25, 0.], -17.2),
    'urban_street_assets': ('configs/fisheye_scene_urban_street_assets.json', [0., 0., 0.], 0.),
}


def add_lens_mask_arguments(parser):
    parser.add_argument('--fisheye-reference-root', type=Path, required=True,
                        help='Real 20260320 fisheye directory containing cam2 and cam4; only the binary footprint is used')
    parser.add_argument('--reference-frame', default='02214', help='Reference JPG stem, shared by both cameras')


def load_lens_masks(reference_root, reference_frame):
    masks, metadata = {}, {}
    for name, intrinsics in DS_20260320.items():
        path = reference_root.resolve() / name / f'{reference_frame}.jpg'
        masks[name] = lens_mask_from_reference(path, (intrinsics.height, intrinsics.width))
        metadata[name] = dict(reference_image=str(path),
                              method='gray > 5; elliptical close 51x51; fill largest external contour',
                              erode_px=0, valid_pixels=int(masks[name].sum()),
                              valid_fraction=float(masks[name].mean()))
    return masks, metadata


def rig_poses(origin, yaw_deg, baseline_m):
    """Return ERP camera-to-world and physical DS camera-to-world matrices.

    Both normal ERPs are rendered in the same axis orientation, at distinct
    centers. cam4's DS axis flip belongs to fisheye projection/remapping only.
    """
    if not np.isfinite(baseline_m) or baseline_m < 0:
        raise ValueError('baseline_m must be nonnegative and finite')
    angle = math.radians(yaw_deg)
    yaw = np.array([[math.cos(angle), -math.sin(angle), 0.],
                    [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    cv_basis = yaw @ np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    poses, lenses = {}, {}
    for name, sign in [('cam2', 1), ('cam4', -1)]:
        matrix = np.eye(4)
        matrix[:3, :3] = cv_basis
        matrix[:3, 3] = np.array(origin) + yaw @ [0., sign * baseline_m / 2, 1.15]
        lens = matrix.copy()
        lens[:3, :3] = cv_basis @ CAMERA_EXTRINSIC[name]
        poses[name], lenses[name] = matrix, lens
    return poses, lenses, yaw


def point_in_view(points, camera, name, mask):
    h, w = mask.shape
    xy, distance = project_erp(points, camera, w, h)
    q = (points - camera[:3, 3]) @ camera[:3, :3]
    fish, domain = project_double_sphere(q @ CAMERA_EXTRINSIC[name].T, DS_20260320[name])
    size = DS_20260320[name]
    finite = np.isfinite(xy).all(1) & np.isfinite(fish).all(1)
    ix = np.floor(np.nan_to_num(xy[:, 0])).astype(int) % w
    iy = np.clip(np.floor(np.nan_to_num(xy[:, 1])).astype(int), 0, h - 1)
    in_view = finite & domain & (fish[:, 0] >= 0) & (fish[:, 0] < size.width) & (fish[:, 1] >= 0) & (fish[:, 1] < size.height) & mask[iy, ix]
    return xy, fish, distance, domain, in_view


def select_people(origin, yaw_deg, yaw, poses, vertices, dense, masks):
    people = []
    for person_id, sign, frame in [(1, 1, 0), (2, -1, 8)]:
        choices = []
        for distance in [.9, 1.2, 1.5, 1.8, 2.2, 2.6]:
            for angle in np.linspace(77, 103, 27) * sign:
                radians = math.radians(angle)
                local = np.array([distance * math.sin(radians), distance * math.cos(radians), 0.])
                person = dict(id=person_id, frame=frame,
                              location=(np.asarray(origin) + yaw @ local).tolist(),
                              yaw_deg=yaw_deg + 90, category='edge_partial')
                points = world_points(dense[frame], vertices[frame], person)
                view = [point_in_view(points, poses[n], n, masks[n])[-1] for n in ('cam2', 'cam4')]
                counts = [int(v.sum()) for v in view]
                common = int((view[0] & view[1]).sum())
                if min(counts) >= 20 and max(counts) <= 110 and common >= 8:
                    score = common - 1.5 * sum(abs(c - 90) for c in counts)
                    choices.append((score, person, counts, common))
        if not choices:
            raise RuntimeError('No paired edge-truncated layout found under supplied masks')
        _, person, counts, common = max(choices, key=lambda row: row[0])
        person.update(projected_in_view=counts, shared_in_view=common)
        people.append(person)
    # A farther person partially overlaps the right-side target in the image.
    farther = dict(people[0])
    farther.update(id=3, frame=16, category='overlap_behind_edge_person')
    relative = np.asarray(people[0]['location']) - origin
    farther['location'] = (np.asarray(origin) + relative * 2.0 + yaw @ [0., .30, 0.]).tolist()
    farther.pop('projected_in_view'); farther.pop('shared_in_view')
    people.append(farther)
    people.append(dict(id=4, frame=24, location=(np.asarray(origin) + yaw @ [0., 3.8, 0.]).tolist(),
                       yaw_deg=yaw_deg, category='complete_single_view_control'))
    # Conservative spacing check: disjoint body AABBs guarantee no body intersection.
    boxes = []
    for person in people:
        body = world_points(vertices[person['frame']], vertices[person['frame']], person)
        boxes.append((body.min(0), body.max(0)))
    for i, (low, high) in enumerate(boxes):
        for other_low, other_high in boxes[i+1:]:
            if not ((high + .05 < other_low) | (other_high + .05 < low)).any():
                raise ValueError('Pilot body bounding boxes overlap or lack 5 cm clearance')
    return people


def draw_overlay(rgb, xy, visible, in_view, ids):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    colors = [(255, 80, 70), (60, 255, 100), (80, 150, 255), (255, 220, 60)]
    for j, person_id in enumerate(ids):
        for point, seen, present in zip(xy[j], visible[j], in_view[j]):
            if not present: continue
            x, y = point
            radius = 2
            color = colors[j % len(colors)] if seen else (170, 170, 170)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        draw.text((8, 8 + 16*j), f'person {person_id}: visible {int(visible[j].sum())} / in-view {int(in_view[j].sum())}', fill=colors[j % len(colors)])
    return image


def process_scene(output, people, poses, masks, vertices, dense, *, lens_masks, lens_metadata):
    ids = [person['id'] for person in people]
    world = []
    for person in people:
        if 'geometry' in person:
            geometry = Path(person['geometry'])
            body = np.load(geometry / 'vertices.npy', mmap_mode='r')
            points = np.load(geometry / 'densekp_3d.npy', mmap_mode='r')
        else:
            body, points = vertices, dense
        frame = person['frame']
        world.append(world_points(points[frame], body[frame], person))
    world = np.stack(world)
    np.save(output / 'shared_world_densekp.npy', world)
    camera_reports, states = [], {}
    for name in ('cam2', 'cam4'):
        _, depth, instance, _ = read_combined_depth_index(output / 'source' / f'{name}.exr')
        rgb = np.array(Image.open(output / 'source' / f'{name}_preview.png').convert('RGB'))
        height, width = depth.shape
        mask = cv2.resize(masks[name].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        lens_mask = np.asarray(lens_masks[name], dtype=bool)
        intrinsics = DS_20260320[name]
        if lens_mask.shape != (intrinsics.height, intrinsics.width):
            raise ValueError(f'{name}: lens mask does not match DS image size')
        back_x, back_y = build_erp_to_fisheye_lut(intrinsics, name, width, height)
        lens_erp = cv2.remap(lens_mask.astype(np.uint8), back_x, back_y,
                             cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
        removed_erp_pixels = int((mask & ~lens_erp).sum())
        # Labels, instances and RGB share the intersection with the measured footprint.
        mask &= lens_erp
        flat = world.reshape(-1, 3)
        xy, fish_xy, ranges, domain, in_view = point_in_view(flat, poses[name], name, mask)
        _, source_visible = erp_visibility(xy, ranges, depth, instance, ids)
        visible = source_visible & in_view
        q = (flat - poses[name][:3, 3]) @ poses[name][:3, :3]
        rays = q @ CAMERA_EXTRINSIC[name].T
        recovered, inverse_valid = unproject_double_sphere(fish_xy, DS_20260320[name])
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        valid_roundtrip = domain & inverse_valid
        ray_error = float(np.max(np.linalg.norm(recovered[valid_roundtrip] - rays[valid_roundtrip], axis=1)))
        if ray_error > 1e-8: raise RuntimeError(f'{name}: DS projection roundtrip {ray_error}')
        # Same physical camera center on both remaps. No per-view person roll.
        fish_x, fish_y = build_fisheye_to_source_erp_lut(DS_20260320[name], name, width, height)
        fish_rgb = remap_periodic_erp(rgb, fish_x, fish_y, cv2.INTER_LINEAR)
        fish_rgb[~lens_mask] = 0
        restored = cv2.remap(fish_rgb, back_x, back_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fish_instance = remap_periodic_erp(np.rint(instance).astype(np.float32), fish_x, fish_y, cv2.INTER_NEAREST)
        fish_instance[~lens_mask] = 0
        final_instance = cv2.remap(fish_instance, back_x, back_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        final_instance[~mask] = 0
        restored[~mask] = 0
        camera_dir = output / name; camera_dir.mkdir(exist_ok=True)
        Image.fromarray(fish_rgb).save(camera_dir / 'fisheye.png')
        Image.fromarray(lens_mask.astype(np.uint8)*255).save(camera_dir / 'reference_lens_mask.png')
        Image.fromarray(restored).save(camera_dir / 'normal_erp.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(camera_dir / 'valid_mask.png')
        shape = (len(people), 138)
        data = dict(person_ids=np.array(ids), xy=xy.reshape(*shape, 2).astype(np.float32),
                    fisheye_xy=fish_xy.reshape(*shape, 2).astype(np.float32),
                    model_domain=domain.reshape(shape), in_view=in_view.reshape(shape),
                    surface_visible=source_visible.reshape(shape), visible=visible.reshape(shape),
                    out_of_view=(~in_view).reshape(shape),
                    occluded_in_view=(in_view & ~source_visible).reshape(shape),
                    camera_to_world_erp=poses[name], world_densekp=world)
        np.savez(camera_dir / 'annotations.npz', **data)
        draw_overlay(restored, data['xy'], data['visible'], data['in_view'], ids).save(camera_dir / 'overlay.png')
        boxes = []
        for index, person_id in enumerate(ids):
            selected = (np.rint(final_instance) == person_id) & mask
            Image.fromarray(selected.astype(np.uint8)*255).save(camera_dir / f'person_{person_id}_mask.png')
            yy, xx = np.where(selected)
            bbox = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1] if len(xx) else None
            boxes.append(dict(person_id=person_id, bbox_xyxy_exclusive=bbox,
                              source='oracle rendered instance mask; not detector output'))
        (camera_dir / 'boxes.json').write_text(json.dumps(boxes, indent=2)+'\n')
        camera_reports.append(dict(camera=name, in_view=data['in_view'].sum(1).tolist(),
                                   visible=data['visible'].sum(1).tolist(), ray_roundtrip_max=ray_error,
                                   lens_mask={**lens_metadata[name], 'removed_erp_valid_pixels': removed_erp_pixels}))
        states[name] = data
    paired = []
    for j, person_id in enumerate(ids):
        a, b = states['cam2']['visible'][j], states['cam4']['visible'][j]
        paired.append(dict(person_id=person_id, shared_visible=int((a & b).sum()),
                           cam2_only_visible=int((a & ~b).sum()), cam4_only_visible=int((b & ~a).sum()),
                           neither_visible=int((~a & ~b).sum())))
    np.savez(output / 'pair_annotations.npz', person_ids=np.array(ids), world_densekp=world,
             shared_visible=states['cam2']['visible'] & states['cam4']['visible'],
             cam2_xy=states['cam2']['xy'], cam4_xy=states['cam4']['xy'])
    panels = [Image.open(output / n / 'overlay.png').convert('RGB') for n in ('cam2','cam4')]
    sheet = Image.new('RGB', (panels[0].width, 2*panels[0].height))
    for i, panel in enumerate(panels): sheet.paste(panel, (0, i*panel.height))
    sheet.save(output / 'pair_review.jpg')
    return dict(cameras=camera_reports, pairs=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-m', type=float, default=.18, help='Synthetic trial distance, not measured calibration')
    parser.add_argument('--blender-bin', type=Path, default=Path('/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender'))
    parser.add_argument('--width', type=int, default=1536)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--device', choices=['CPU','CUDA'], default='CPU')
    parser.add_argument('--skip-render', action='store_true')
    parser.add_argument('--scene', choices=list(SCENES), nargs='+', default=list(SCENES))
    add_lens_mask_arguments(parser)
    args = parser.parse_args()
    lens_masks, lens_metadata = load_lens_masks(args.fisheye_reference_root, args.reference_frame)
    geometry = ROOT / 'outputs/baselines/temporal_case_aaron_1095/geometry'
    clothing = ROOT / 'outputs/baselines/temporal_case_aaron_1095/clothing_geometry'
    vertices = np.load(geometry / 'vertices.npy', mmap_mode='r')
    dense = np.load(geometry / 'densekp_3d.npy', mmap_mode='r')
    appearance, _ = load_appearance_baseline()
    with np.load(ROOT / 'assets_external/real_erp_valid_masks.npz') as z:
        masks = {n: z[f'valid_masks_{n}'] for n in ('cam2','cam4')}
    for name in args.scene:
        config_path, origin, yaw_deg = SCENES[name]
        scene_config = json.loads((ROOT / config_path).read_text())
        output = args.output.resolve() / name; output.mkdir(parents=True, exist_ok=True)
        poses, lenses, yaw = rig_poses(origin, yaw_deg, args.baseline_m)
        if args.skip_render:
            people = json.loads((output / 'people.json').read_text())['people']
            saved = json.loads((output / 'rig.json').read_text())
            if saved['baseline_m'] != args.baseline_m: raise ValueError('Saved/rendered baseline differs')
            if saved['intrinsics'] != {n:asdict(DS_20260320[n]) for n in poses}:
                raise ValueError('Saved/rendered intrinsics differ from active projection parameters')
            poses = {camera['id']: np.asarray(camera['camera_to_world']) for camera in saved['cameras']}
        else:
            if (output / 'source/source_manifest.json').exists(): raise FileExistsError('Existing rendered pilot; use --skip-render or a fresh output directory')
            people = select_people(origin, yaw_deg, yaw, poses, vertices, dense, masks)
            rig = dict(camera_model='Double Sphere', calibration_group='20260320',
                       pose_status='user-specified same-height 180-degree orientation; synthetic baseline distance',
                       baseline_m=args.baseline_m, intrinsics={n:asdict(DS_20260320[n]) for n in poses},
                       cameras=[dict(id=n, camera_to_world=poses[n].tolist(),
                                     physical_ds_camera_to_world=lenses[n].tolist()) for n in poses])
            (output / 'rig.json').write_text(json.dumps(rig, indent=2)+'\n')
            (output / 'people.json').write_text(json.dumps(dict(people=people), indent=2)+'\n')
            command = [str(args.blender_bin), '-b', '--disable-autoexec', '--python-exit-code', '1',
                       '--python', str(ROOT/'tools/fisheye/render_equirectangular_source.py'), '--',
                       '--geometry',str(geometry), '--clothing-geometry',str(clothing),
                       '--cameras',str(output/'rig.json'), '--people-config',str(output/'people.json'),
                       '--scene-blend',str(ROOT/scene_config['blend_path']), '--output-dir',str(output/'source'),
                       '--width',str(args.width),'--height',str(args.width//2),'--samples',str(args.samples),
                       '--device',args.device,'--body-texture',appearance['body_diffuse_texture_path'],
                       '--body-uv',appearance['body_uv_path']]
            print('Rendering '+name, flush=True)
            with (output/'render.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        rendered = json.loads((output / 'source/source_manifest.json').read_text())
        if rendered['people'] != people:
            raise ValueError('Rendered people do not match annotation layout')
        for camera in rendered['sources']:
            if not np.allclose(camera['camera_to_world'], poses[camera['id']], atol=1e-12, rtol=0):
                raise ValueError('Rendered camera differs from annotation camera')
        report = process_scene(output, people, poses, masks, vertices, dense,
                               lens_masks=lens_masks, lens_metadata=lens_metadata)
        report.update(scene=name, baseline_m=args.baseline_m, baseline_status='synthetic distance; not measured',
                      orientation='same-height opposite optical axes', people=people,
                      limitation='one repeated motion/identity, oracle boxes, low-sample rendering; not a train/val corpus',
                      visibility_rule='matching instance and ray-range within 4 cm in a 5x5 source-ERP neighborhood',
                      valid_mask_source='assets_external/real_erp_valid_masks.npz',
                      regressor_source='assets/densekp_regressor_R.npy')
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
