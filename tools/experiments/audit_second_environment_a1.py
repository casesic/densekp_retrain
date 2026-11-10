"""Read-only numerical audit and compact review drawings for A1's environment probe."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.data.dynamic_scene import frame_record
from densekp_retrain.data.frame_sources import frame_directory


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def svgs(base, projection, recipe):
    rows = projection['frames']
    colors = {'cam2': '#1976d2', 'cam4': '#d84315'}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="330" viewBox="0 0 860 330">',
           '<rect width="860" height="330" fill="white"/><text x="30" y="28" font-size="18">Physical-domain points: riverfront boundary pass</text>']
    for threshold, label in [(20, '20 outside'), (5, '5 outside')]:
        y = 280 - threshold * 2
        svg += [f'<line x1="45" y1="{y}" x2="820" y2="{y}" stroke="#bbb" stroke-dasharray="4 4"/>',
                f'<text x="46" y="{y-3}" font-size="11">{label}</text>']
    for camera, color in colors.items():
        points = ' '.join(f'{45+i*25},{280-row["views"][camera]["outside_domain"]*2}' for i,row in enumerate(rows))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    common = ' '.join(f'{45+i*25},{280-row["common_physical_points"]*1.6}' for i,row in enumerate(rows))
    svg.append(f'<polyline points="{common}" fill="none" stroke="#2e7d32" stroke-width="2"/>')
    svg += ['<text x="45" y="313" font-size="12">0 s</text><text x="770" y="313" font-size="12">3 s</text>',
            '<text x="520" y="28" font-size="12" fill="#1976d2">cam2 outside</text>',
            '<text x="620" y="28" font-size="12" fill="#d84315">cam4 outside</text>',
            '<text x="730" y="28" font-size="12" fill="#2e7d32">common inside</text></svg>']
    (base / 'physical_boundary_curve.svg').write_text(''.join(svg))
    matrices = np.asarray(recipe['people'][0]['scene_from_model'])
    rig = json.loads((base / 'prepared_riverfront/rig.json').read_text())['cameras']
    origin = np.asarray(matrices[0])[:3, 3]
    def point(x, y):
        return f'{250+x*55:.1f},{210-y*55:.1f}'
    path = ' '.join(point(*matrix[:2, 3]) for matrix in matrices)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="500" height="420" viewBox="0 0 500 420">',
           '<rect width="500" height="420" fill="#f3f5f2"/>',
           '<rect x="112" width="276" height="420" fill="#d6d2ca"/>',
           '<text x="20" y="26" font-size="17">Riverfront road and frozen rig (plan view)</text>',
           f'<polyline points="{path}" fill="none" stroke="#1976d2" stroke-width="4"/>']
    for camera in rig:
        x,y = np.asarray(camera['camera_to_world'])[:2,3]
        px,py=point(x,y).split(',')
        svg += [f'<circle cx="{px}" cy="{py}" r="7" fill="#a30000"/>',
                f'<text x="{float(px)+9}" y="{py}" font-size="12">{camera["id"]}</text>']
    for frame in (0,15,30):
        x,y=matrices[frame,:2,3];px,py=point(x,y).split(',')
        svg += [f'<circle cx="{px}" cy="{py}" r="5" fill="#1976d2"/>',
                f'<text x="{float(px)+8}" y="{float(py)-8}" font-size="12">t={frame/10:g}s</text>']
    svg.append('<text x="20" y="396" font-size="12">Road width 5 m; physical view boundary is checked by calibrated masks, not drawn as a world seam.</text></svg>')
    (base / 'trajectory_plan.svg').write_text(''.join(svg))


def main():
    base = ROOT / 'outputs/experiments/second_environment_boundary_pass_a1_20261010'
    prepared = base / 'prepared_riverfront'
    run = base / 'riverfront_complete'
    probe = base / 'probe_riverfront'
    recipe = json.loads((run / 'recipe.json').read_text())
    projection = json.loads((prepared / 'selected_projection.json').read_text())
    plan = json.loads((prepared / 'planned_regions.json').read_text())
    expected = np.load(prepared / 'expected_in_view.npy')
    world = np.load(prepared / 'world_densekp.npy')
    index = json.loads((run / 'pack_corrected/sequence_index.json').read_text())
    readback = json.loads((run / 'pack_corrected/readback.json').read_text())
    exact_roi = exact_in_view = exact_world = exact_time = True
    roi_areas = []
    for frame in range(31):
        folder = frame_directory(run, frame)
        report = json.loads((folder / 'report.json').read_text())
        exact_time &= report['selected_time_record'] == frame_record(recipe, frame)
        actual_world = np.load(folder / 'shared_world_densekp.npy')
        exact_world &= bool(np.allclose(actual_world, world[frame], rtol=0, atol=2e-5))
        for ci, camera in enumerate(('cam2', 'cam4')):
            camera_dir = folder / camera
            storage = json.loads((camera_dir / 'raster_storage.json').read_text())
            roi = storage['roi_xyxy']
            exact_roi &= roi == plan[frame][camera]['roi_xyxy']
            roi_areas.append((roi[2]-roi[0])*(roi[3]-roi[1]))
            with Image.open(camera_dir / 'roi.png') as image:
                if list(image.size) != storage['stored_size_wh']:
                    raise ValueError('stored_roi_size_conflict')
            with np.load(camera_dir / 'annotations.npz') as labels:
                exact_in_view &= bool(np.array_equal(labels['in_view'][0], expected[frame,0,ci]))
                if not np.isfinite(labels['xy']).all():
                    raise ValueError('nonfinite_projected_gt')
    if not all((exact_roi, exact_in_view, exact_world, exact_time)):
        raise ValueError('actual_episode_differs_from_frozen_geometry_or_time')
    if list(run.rglob('*.exr')) or list(probe.rglob('*.exr')):
        raise ValueError('successful_data_depends_on_temporary_exr')
    def calls(path):
        return [json.loads(line) for line in (path/'render_calls.jsonl').read_text().splitlines() if json.loads(line)['event']=='completed']
    forest = base / 'probe_forest'
    forest_calls = calls(forest)
    probe_calls = calls(probe)
    suffix_calls = calls(run)
    svgs(base, projection, recipe)
    result = dict(schema='second_environment_boundary_pass_a1_v1',status='riverfront_complete_forest_probe_budget_rejected',
        first_environment='urban_street_assets_unchanged',selected_second_environment='riverfront',
        gpu_uuid='GPU-d481acc7-106f-29e5-1298-8c0d398261b2',
        bounded_compute_accounted_seconds=(json.loads((forest/'failure.json').read_text())['seconds']+
            json.loads((probe/'task_wall.json').read_text())['total_generation_wall_seconds']+
            json.loads((run/'task_wall.json').read_text())['total_generation_wall_seconds']+
            json.loads((base/'finish_correction.json').read_text())['corrected_cpu_finish_wall_seconds_approx']),
        bounded_compute_budget_seconds=2700,
        retained_output_bytes=sum(path.stat().st_size for path in base.rglob('*') if path.is_file()),
        forest=dict(probe_wall_seconds=json.loads((forest/'failure.json').read_text())['seconds'],
                    completed_camera_calls=len(forest_calls),completed_times=int((forest/'frame_000/report.json').exists()),
                    first_camera_seconds=forest_calls[0]['seconds'],second_camera_seconds=forest_calls[1]['seconds'],
                    reason='8_minute_probe_budget_exhausted_before_three_times',
                    sampled_peak_GPU_MiB=json.loads((forest/'blender_process.json').read_text())['sampled_peak_GPU_MiB']),
        riverfront=dict(probe_wall_seconds=json.loads((probe/'task_wall.json').read_text())['total_generation_wall_seconds'],
                        probe_camera_calls=len(probe_calls),suffix_camera_calls=len(suffix_calls),
                        probe_camera_seconds=[x['seconds'] for x in probe_calls],suffix_camera_seconds=[x['seconds'] for x in suffix_calls],
                        probe_frame_indices=[0,1,2],resumed_frame_indices=list(range(3,31)),
                        blender_start_count=2,selected_environment_native_road_hits=sorted({o for x in json.loads((probe/'job_ground.json').read_text()) for o in x['hit_objects']}),
                        roi_area_range_pixels=[min(roi_areas),max(roi_areas)],
                        sampled_peak_GPU_MiB=max(json.loads((probe/'blender_process.json').read_text())['sampled_peak_GPU_MiB'],json.loads((run/'blender_process.json').read_text())['sampled_peak_GPU_MiB'])),
        dataset=dict(planned_times=31,camera_images=62,observation_slots=len(index['observations']),valid_rows=index['packed']['row_count'],
                     actual_Dataset_reads=readback['actual_Dataset_reads'],actual_readback_frame_indices=sorted({x['frame_index'] for x in readback['checks'] if x.get('Dataset_read')}),
                     all_actual_ROI_equal_frozen=exact_roi,all_actual_in_view_equal_frozen=exact_in_view,
                     all_world_R138_equal_frozen=exact_world,all_person_time_records_equal_frozen=exact_time,
                     sequence_validation=json.loads((run/'pack_corrected/validation.json').read_text())['status'],
                     video_frames=len(readback['video_pts_seconds']),video_pts_first_last_seconds=[readback['video_pts_seconds'][0],readback['video_pts_seconds'][-1]]),
        physical_boundary=dict(crossing=projection['boundary_crossing'],dual_edge_count=projection['dual_edge']['count'],
                               dual_edge_windows=projection['dual_edge']['windows'],
                               common_physical_first_mid_last=[projection['frames'][i]['common_physical_points'] for i in (0,15,30)]),
        correction=dict(initial_finish='failed_after_pack_before_readback',corrected_pack='riverfront_complete/pack_corrected',
                        rerendered_successful_frames=0),
        material_overlay='none; original riverfront blend and procedural materials read-only',
        scene_blend_runtime_stat_frozen=False,
        scene_blend_runtime_stat_note='path and scene ID frozen for this run; size/mtime enforcement was added after this run for future plans',
        result_status='integration_only_shared_source_diagnostic; human_environment_acceptance_pending')
    write(base/'acceptance.json',result)
    (base/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Second environment boundary pass</title>
<h1>Riverfront: single rider crossing the dual physical view boundary</h1>
<p>31 synthetic 10 Hz moments, cam2/cam4. Grey video layout is unrendered. Black lens area is outside the physical valid domain.</p>
<video controls width="1024" src="riverfront_complete/pack_corrected/dual_roi.mp4"></video>
<p><a href="physical_boundary_curve.svg">Physical boundary curve</a> · <a href="trajectory_plan.svg">Road trajectory</a> · <a href="acceptance.json">Numerical audit</a></p>
<p>Forest probe reached its 8-minute limit after one complete dual-camera moment; retained <a href="probe_forest/frame_000/cam2/roi.png">cam2 ROI</a> and <a href="probe_forest/frame_000/cam4/roi.png">cam4 ROI</a>.</p>
''')
    print('SECOND_ENVIRONMENT_AUDIT_PASSED',base)


if __name__ == '__main__':
    main()
