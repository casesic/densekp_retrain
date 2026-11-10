"""Resume a frozen episode without rewriting or cleaning parent products."""
import json
from pathlib import Path
import shutil
from PIL import Image
from .dynamic_scene import frame_record, person_matrices
from .observation_policy import RECORD


def inspect_parent(parent):
    parent = Path(parent).resolve()
    recipe = json.loads((parent / 'recipe.json').read_text())
    reused, reprocessed, new = [], [], []
    for k in range(recipe['frame_count']):
        frame = parent / f'frame_{k:03d}'
        expected = frame_record(recipe, k)
        if (frame / 'report.json').exists():
            report = json.loads((frame / 'report.json').read_text())
            if report['selected_time_record'] != expected:
                raise ValueError('parent_report_time_conflict')
            reused.append(k)
        elif (frame / 'rejected_frame.json').exists():
            source = frame / '_temporary_ds_source/source'
            manifest = json.loads((source / 'source_manifest.json').read_text())
            if manifest['selected_time_record'] != expected:
                raise ValueError('reprocess_source_time_conflict')
            for c in ('cam2', 'cam4'):
                for name in (f'{c}.exr', f'{c}_preview.png'):
                    path = source / 'roi' / name
                    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                        raise ValueError('missing_or_unsafe_reprocess_buffer')
            reprocessed.append(k)
        else:
            source = frame / '_temporary_ds_source/source'
            if any((frame / c / 'roi.png').exists() for c in ('cam2', 'cam4')) or (source.exists() and any(p.is_file() for p in source.rglob('*'))):
                raise ValueError('unclassified_partial_frame')
            new.append(k)
            continue
        for c in ('cam2', 'cam4'):
            cd = frame / c
            meta = json.loads((cd / 'raster_storage.json').read_text())
            for name in ['roi.png', 'valid_mask.png'] + [f'person_{p["person_id"]}_mask.png' for p in recipe['people']]:
                with Image.open(cd / name) as image:
                    if list(image.size) != meta['stored_size_wh']:
                        raise ValueError('parent_stored_size_conflict')
            for name in ('annotations.npz', 'boxes.json', 'render_coverage.json'):
                if not (cd / name).is_file():
                    raise ValueError('missing_parent_final_file')
    if reused + reprocessed + new != list(range(recipe['frame_count'])) or not new:
        raise ValueError('resume_requires_completed_prefix_then_reprocess_then_new_suffix')
    for person in recipe['people']:
        person_matrices(person, recipe['frame_count'])
    return dict(schema='frozen_episode_resume_v1', parent_run=str(parent),
        observation_policy=RECORD, planned=list(range(recipe['frame_count'])),
        reused=reused, reprocessed=reprocessed, new=new,
        parent_outputs_readonly=True, parent_failure_preserved=True,
        no_asset_time_trajectory_or_ROI_change=True,
        readback_policy='old first, first reprocessed, first target cam2 box reappearance (or last); <=12 rows')


def prepare_reprocessing(parent, out, resume):
    from densekp_retrain.imaging.output_modes import prepare_source_workspace
    parent, out = Path(parent).resolve(), Path(out).resolve()
    mapping = {}
    for k in resume['planned']:
        reused = k in resume['reused']
        mapping[str(k)] = dict(directory=str((parent if reused else out) / f'frame_{k:03d}'),
                               ownership='readonly_parent' if reused else 'this_run')
    for k in resume['reprocessed']:
        original = parent / f'frame_{k:03d}/_temporary_ds_source/source'
        dest = prepare_source_workspace(out / f'frame_{k:03d}', 'compact')
        # Copy buffers, never ownership tokens or symlinks into the parent.
        (dest / 'roi').mkdir()
        for rel in ['source_manifest.json', 'roi_plan.json'] + [f'roi/{c}{ext}' for c in ('cam2', 'cam4') for ext in ('.exr', '_preview.png')]:
            if (original / rel).is_symlink():
                raise ValueError('symlink_reprocess_buffer')
            shutil.copy2(original / rel, dest / rel)
    (out / 'frame_sources.json').write_text(json.dumps(dict(schema='readonly_frame_sources_v1', frames=mapping), indent=2)+'\n')
    (out / 'resume_manifest.json').write_text(json.dumps(resume, indent=2)+'\n')
