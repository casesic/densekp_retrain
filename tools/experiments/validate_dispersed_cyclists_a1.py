"""Full-slot validation for a 3–5-person city scene with paired rig lists."""
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'assets_external/python_deps')]
from densekp_retrain.data.dynamic_scene import geometry_paths
from densekp_retrain.data.sequence_index import validate_sequence_index
from densekp_retrain.data.occlusion_events import three_stage_event


def read(path):
    return json.loads(Path(path).read_text())


def longest(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def validate(root, *, expected_rigs=None):
    root = Path(root).resolve()
    scene = read(root / 'recipe.json')
    count = len(scene['people'])
    if count not in (3, 4, 5) or [p['person_id'] for p in scene['people']] != list(range(1, count + 1)):
        raise ValueError('incorrect_person_axis')
    components = [p['component_ids'] for p in scene['people']]
    if len({x['body_id'] for x in components}) != count or len({x['motion_id'] for x in components}) < 3:
        raise ValueError('identity_or_motion_diversity_not_met')
    if len({p['source_group'] for p in scene['people']}) != 3:
        raise ValueError('motion_source_group_count_conflict')
    motion = []
    for p in scene['people']:
        asset = read(p['asset_manifest'])
        vertices = np.load(geometry_paths(asset)['body_vertices'], mmap_mode='r')
        frames = p['asset_frames']
        if len(frames) != 31 or any(frames[k] >= frames[k + 1] for k in range(30)):
            raise ValueError('person_original_frame_not_strictly_increasing')
        differences = [float(np.max(np.abs(vertices[frames[a]] - vertices[frames[b]])))
                       for a, b in ((0, 15), (15, 30))]
        if min(differences) <= 1e-5:
            raise ValueError('body_not_articulating')
        motion.append(dict(person_id=p['person_id'], motion_id=p['component_ids']['motion_id'],
                           asset_frames=[frames[k] for k in (0, 15, 30)],
                           max_vertex_changes_m=differences))
    world = np.load(root / 'world_densekp.npy')
    if world.shape != (31, count, 138, 3) or not np.isfinite(world).all():
        raise ValueError('shared_world_shape_or_finite_conflict')
    spec = read(root / 'multirig.json')
    scene_id = spec['scene_episode_id']
    rigs = [r['rig_id'] for r in spec['rigs']]
    if rigs != (expected_rigs or ['rig_edge', 'rig_front_oblique']):
        raise ValueError('rig_order_conflict')
    logs = [json.loads(line) for line in (root / 'render_calls.jsonl').read_text().splitlines()]
    ordered = [(k, rig, camera) for k in range(31) for rig in rigs for camera in ('cam2', 'cam4')]
    for event in ('started', 'completed'):
        actual = [(x['frame_index'], x['rig_id'], x['camera_id']) for x in logs if x['event'] == event]
        if actual != ordered:
            raise ValueError('camera_call_order_or_completion_conflict:' + event)
    lifecycle = read(root / 'multirig_render_result.json')
    if lifecycle['scene_load_count'] != 1 or lifecycle['normal_path_geometry_updates'] != 31 or lifecycle['consumer_process_count'] != 1:
        raise ValueError('shared_scene_lifecycle_conflict')
    relative = []
    timing = []
    results = {}
    for rig_id in rigs:
        folder = root / 'rigs' / rig_id
        recipe = read(folder / 'recipe.json')
        if recipe['scene_episode_id'] != scene_id or recipe['rig_id'] != rig_id or recipe['people'] != scene['people']:
            raise ValueError('rig_person_plan_changed')
        if not np.array_equal(np.load(folder / 'world_densekp.npy'), world):
            raise ValueError('world_changed_between_rigs')
        poses = {x['id']: np.asarray(x['camera_to_world']) for x in read(folder / 'rig.json')['cameras']}
        relative.append(np.linalg.inv(poses['cam2']) @ poses['cam4'])
        pack = folder / 'pack'
        episode = read(pack / 'episode.json')
        index = read(pack / 'sequence_index.json')
        validation = validate_sequence_index(episode, index, base_dir=pack, check_files=True, check_packed=True)
        expected_slots = 31 * 2 * count
        if episode['scene_episode_id'] != scene_id or episode['rig_id'] != rig_id or len(index['observations']) != expected_slots:
            raise ValueError('sequence_slot_count_or_identity_conflict')
        if any(x['rig_id'] != rig_id for x in index['observations'] + index['boxes']):
            raise ValueError('cross_rig_index')
        plan = read(pack / 'readback_plan.json')
        frozen = read(root.parent / 'readback_plan_frozen.json')['rigs'][rig_id]
        reads = read(pack / 'readback.json')
        if set(plan['selected_observation_ids']) != set(frozen) or reads['actual_Dataset_reads'] > 12:
            raise ValueError('readback_selection_or_limit_conflict')
        boxes = {x['observation_id'] for x in index['boxes']}
        visible = np.zeros((31, 2, count), dtype=np.int16)
        physical = np.zeros((31, 2, count, 138), dtype=bool)
        neighbor_code = np.zeros((31, 2, 138), dtype=np.uint8)
        neighbor_evidence_available = True
        area = []
        selected_times = []
        for k in range(31):
            frame = folder / f'frame_{k:03d}'
            selected_times.append(read(frame / 'report.json')['selected_time_record'])
            if not np.array_equal(np.load(frame / 'shared_world_densekp.npy'), world[k]):
                raise ValueError('frame_world_changed')
            for ci, camera in enumerate(('cam2', 'cam4')):
                camera_dir = frame / camera
                meta = read(camera_dir / 'raster_storage.json')
                x0, y0, x1, y1 = meta['roi_xyxy']
                shape = (x1 - x0, y1 - y0)
                area.append(shape[0] * shape[1])
                if (meta['offset_xy'] != [x0, y0] or meta['full_image_size_wh'] != [5760, 2880] or
                    area[-1] > 7_000_000):
                    raise ValueError('ROI_offset_full_size_or_area_conflict')
                for name in ('roi.png', 'valid_mask.png'):
                    with Image.open(camera_dir / name) as im:
                        if im.size != shape:
                            raise ValueError('ROI_image_or_mask_size_conflict')
                with Image.open(camera_dir / 'valid_mask.png') as im:
                    valid_pixels = np.asarray(im) > 0
                native = read(camera_dir / 'boxes.json')
                if [x['person_id'] for x in native] != list(range(1, count + 1)):
                    raise ValueError('box_person_axis_conflict')
                with np.load(camera_dir / 'annotations.npz', allow_pickle=False) as z:
                    if (str(z['scene_episode_id']) != scene_id or str(z['rig_id']) != rig_id or
                        not np.array_equal(z['person_ids'], np.arange(1, count + 1)) or
                        z['xy'].shape != (count, 138, 2) or z['in_view'].shape != (count, 138) or
                        not np.isfinite(z['xy']).all() or not np.array_equal(z['world_densekp'], world[k])):
                        raise ValueError('annotation_identity_axis_or_world_conflict')
                    expected = np.load(folder / 'expected_in_view.npy', mmap_mode='r')[k, :, ci]
                    if not np.array_equal(z['in_view'], expected):
                        raise ValueError('physical_projection_conflict')
                    physical[k, ci] = z['in_view']
                    visible[k, ci] = z['visible'].sum(axis=1)
                    if 'occlusion_evidence_code' in z:
                        neighbor_code[k, ci] = z['occlusion_evidence_code'][0]
                    else:
                        neighbor_evidence_available = False
                occupied_pixels = np.zeros(valid_pixels.shape, dtype=bool)
                for person in range(1, count + 1):
                    with Image.open(camera_dir / f'person_{person}_mask.png') as im:
                        if im.size != shape:
                            raise ValueError('person_mask_size_conflict')
                        mask_pixels = np.asarray(im) > 0
                    if (mask_pixels & ~valid_pixels).any() or (mask_pixels & occupied_pixels).any():
                        raise ValueError('person_mask_outside_domain_or_ID_overlap')
                    occupied_pixels |= mask_pixels
                    oid = f'f{k}_{camera}_p{person}'
                    if oid in boxes and native[person - 1]['bbox_xyxy_exclusive'] is None:
                        raise ValueError('packed_row_without_native_box')
        timing.append(selected_times)
        runs = {str(person): max(longest([(f'f{k}_{camera}_p{person}' in boxes and visible[k, ci, person - 1] >= 8)
                    for k in range(31)]) for ci, camera in enumerate(('cam2', 'cam4')))
                for person in range(1, count + 1)}
        target = physical[:, :, 0]
        common = np.logical_and(target[:, 0], target[:, 1]).sum(axis=1)
        edge = ((target[:, 0].sum(axis=1) >= 8) & ((~target[:, 0]).sum(axis=1) >= 5) &
                (target[:, 1].sum(axis=1) >= 8) & ((~target[:, 1]).sum(axis=1) >= 5) & (common >= 8))
        shared = target[:, 0] & target[:, 1]
        neighbor_fraction = {}
        neighbor_event = {}
        for ci, camera in enumerate(('cam2', 'cam4')):
            if neighbor_evidence_available:
                values = [float(((neighbor_code[k, ci] == 2) & shared[k]).sum() / common[k])
                          if common[k] else None for k in range(31)]
                neighbor_fraction[camera] = values
                neighbor_event[camera] = three_stage_event(values, delta=.15, consecutive=3)
            else:
                neighbor_fraction[camera] = None
                neighbor_event[camera] = dict(status='not_evaluable',
                    reason='no_persistent_neighbor_ID_and_range_evidence_in_this_episode')
        results[rig_id] = dict(planned_slots=expected_slots, packed_rows=index['packed']['row_count'],
            unavailable_slots=expected_slots - index['packed']['row_count'],
            actual_Dataset_reads=reads['actual_Dataset_reads'],
            each_person_max_legal_visible_run=runs, all_people_content_passed=all(x >= 6 for x in runs.values()),
            target_dual_edge_frames=int(edge.sum()), target_common_physical_frames=int((common >= 8).sum()),
            target_endpoint_outside_each_camera={camera: [int((~target[0, ci]).sum()),
                         int((~target[-1, ci]).sum())] for ci, camera in enumerate(('cam2', 'cam4'))},
            target_neighbor_occlusion_fraction_on_common_physical=neighbor_fraction,
            target_neighbor_three_stage_event=neighbor_event,
            target_neighbor_evidence_available=neighbor_evidence_available,
            max_roi_pixels=max(area), mean_roi_pixels=float(np.mean(area)),
            sequence_validation=validation, video=str(pack / 'dual_roi.mp4'))
    for matrix in relative[1:]:
        np.testing.assert_allclose(relative[0], matrix, atol=1e-9, rtol=0)
    if any(x != timing[0] for x in timing[1:]):
        raise ValueError('source_timeline_differs_between_rigs')
    content = {str(person): max(results[rig]['each_person_max_legal_visible_run'][str(person)] for rig in rigs)
               for person in range(1, count + 1)}
    outcome = dict(status='technical_passed', scene_episode_id=scene_id, person_count=count,
        distinct_body_families=count, distinct_motion_ids=len({x['motion_id'] for x in components}),
        distinct_source_groups=3, planned_camera_calls=31 * 2 * len(rigs), actual_completed_camera_calls=31 * 2 * len(rigs),
        planned_observation_slots=31 * 2 * len(rigs) * count,
        packed_rows=sum(results[x]['packed_rows'] for x in rigs),
        actual_Dataset_reads=sum(results[x]['actual_Dataset_reads'] for x in rigs),
        each_person_longest_legal_visible_run=content,
        all_people_content_passed=all(x >= 6 for x in content.values()),
        rig_pair_relative_transform_max_abs=float(max((np.max(np.abs(relative[0] - x)) for x in relative[1:]), default=0.0)),
        actual_articulation=motion, rigs=results,
        peak_validation_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (root / 'acceptance.json').write_text(json.dumps(outcome, indent=2, allow_nan=False) + '\n')
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.root)['status'])
