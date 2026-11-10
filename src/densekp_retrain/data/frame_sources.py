"""Explicit read-only final-frame references for a continued single episode.

Consumers writing/cleaning temporary sources must never use this resolver.
"""
import json
from pathlib import Path


def frame_directory(root, index):
    root = Path(root).resolve()
    manifest = root / 'frame_sources.json'
    if not manifest.exists():
        return root / f'frame_{index:03d}'
    sources = json.loads(manifest.read_text())
    if sources.get('schema') != 'readonly_frame_sources_v1':
        raise ValueError('unknown_frame_sources_schema')
    entry = sources['frames'][str(index)]
    path = Path(entry['directory']).resolve()
    if '_temporary_ds_source' in path.parts or path.name != f'frame_{index:03d}':
        raise ValueError('invalid_final_frame_reference')
    if entry['ownership'] not in ('readonly_parent', 'this_run'):
        raise ValueError('invalid_frame_ownership')
    if entry['ownership'] == 'this_run' and path.parent != root:
        raise ValueError('frame_ownership_conflict')
    return path
