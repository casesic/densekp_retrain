"""Transfer one original BEDLAM2 simulated garment onto a fixed target SMPL-X body.

The source cache retains its per-frame wrinkles and topology. A fixed source
frame-0 garment-to-body surface correspondence transfers the difference between
target and source body vertices each frame. This is an approximation, not a new
cloth simulation or a claim that the garment was originally made for M03.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from densekp_retrain.clothing.binding import bind_new_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-clothing', type=Path, required=True)
    ap.add_argument('--native-body', type=Path, required=True)
    ap.add_argument('--target-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    began = time.monotonic()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = json.loads(a.target_manifest.read_text())
    native_body = np.load(a.native_body / 'vertices.npy', mmap_mode='r')
    native_faces = np.load(a.native_body / 'faces.npy')
    target_body = np.load(target['body_vertices'], mmap_mode='r')
    target_faces = np.load(target['body_faces'])
    if native_body.shape != target_body.shape or not np.array_equal(native_faces, target_faces):
        raise ValueError('source_target_body_vertex_correspondence_unverified')
    n = len(target_body)
    with np.load(a.native_clothing, allow_pickle=False) as garment:
        source_seq = np.asarray(garment['vertices_seq'][:n], np.float32)
        faces = np.asarray(garment['faces'], np.int32)
        uv = np.asarray(garment['vertices_uv'], np.float32)
        uv_faces = np.asarray(garment['faces_uv'], np.int32)
        source_total = len(garment['vertices_seq'])
    if source_seq.shape[0] != n or source_total < n or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('garment_frame_or_topology_mismatch')
    if faces.min() < 0 or faces.max() >= source_seq.shape[1] or uv_faces.shape != faces.shape or uv_faces.min() < 0 or uv_faces.max() >= len(uv):
        raise ValueError('garment_face_or_uv_index_out_of_range')
    if not all(np.isfinite(x).all() for x in (source_seq, uv)):
        raise ValueError('nonfinite_original_garment')
    # Bind once in the original garment's first posed frame. Switching the
    # nearest body face every frame could introduce visible garment flicker.
    idx, weights = bind_new_points(source_seq[0], np.asarray(native_body[0]), native_faces)
    if idx.shape != (source_seq.shape[1], 3) or weights.shape != idx.shape:
        raise ValueError('invalid_fixed_transfer_binding')
    np.save(out / 'source_body_triangle_vertex_ids.npy', idx)
    np.save(out / 'source_body_triangle_weights.npy', weights)
    np.save(out / 'clothing_faces.npy', faces)
    np.save(out / 'clothing_vertices_uv.npy', uv)
    np.save(out / 'clothing_faces_uv.npy', uv_faces)
    output = np.lib.format.open_memmap(out / 'clothing_vertices.npy', mode='w+', dtype=np.float32, shape=source_seq.shape)
    records = []
    probe_frames = [0, 30, 60, 90, n-1]
    for t in range(n):
        delta = np.asarray(target_body[t]) - np.asarray(native_body[t])
        field = (delta[idx] * weights[..., None]).sum(1)
        output[t] = source_seq[t] + field
        if not np.isfinite(output[t]).all(): raise ValueError('nonfinite_transferred_garment')
        if t in probe_frames:
            source_d = cKDTree(native_body[t]).query(source_seq[t], k=1)[0]
            target_d = cKDTree(target_body[t]).query(output[t], k=1)[0]
            records.append(dict(asset_frame=t,
                source_to_body_vertex_distance_m=np.quantile(source_d,[.5,.95,1]).tolist(),
                transferred_to_body_vertex_distance_m=np.quantile(target_d,[.5,.95,1]).tolist(),
                cloth_bounds_m=[output[t].min(0).tolist(),output[t].max(0).tolist()],
                target_body_bounds_m=[np.asarray(target_body[t]).min(0).tolist(),np.asarray(target_body[t]).max(0).tolist()]))
    output.flush()
    ratios = []
    for t in probe_frames:
        original_edge = source_seq[t, faces[:, 1]] - source_seq[t, faces[:, 0]]
        new_edge = output[t, faces[:, 1]] - output[t, faces[:, 0]]
        valid = np.linalg.norm(original_edge, axis=1) > 1e-5
        ratio = np.linalg.norm(new_edge[valid], axis=1) / np.linalg.norm(original_edge[valid], axis=1)
        ratios.append(dict(asset_frame=t, edge_length_ratio_quantiles=np.quantile(ratio,[.01,.5,.99]).tolist()))
    # Inspect frame-to-frame motion without turning this into a zero-collision
    # criterion. The original dynamic cache remains the authority for folds.
    max_jump = 0.0
    max_jump_frame = None
    for t in range(1,n):
        step = np.linalg.norm(output[t]-output[t-1],axis=1).max()
        if step > max_jump: max_jump,max_jump_frame=float(step),t
    report = dict(status='numeric_transfer_passed_visual_pending',integration_only=True,
        source_native_clothing=str(a.native_clothing.resolve()),source_native_body=str(a.native_body.resolve()),
        target_body_manifest=str(a.target_manifest.resolve()),source_garment_total_frames=source_total,
        selected_frames=n,garment_vertices=source_seq.shape[1],garment_faces=len(faces),
        UV_vertices=len(uv),fixed_binding=True,original_simulation_deformation_preserved=True,
        formula='C_target[t,g] = C_source[t,g] + sum_w(B_target[t,v]-B_source[t,v]) using fixed frame0 source body surface triangles',
        approximation='body-shape transfer only, not re-simulated on M03; local penetration and drape must be visually reviewed',
        body_face_semantics_equal=True,source_and_target_time_index_equal=True,
        probes=records,edge_length_ratio=ratios,max_adjacent_garment_vertex_step_m=max_jump,
        max_adjacent_step_asset_frame=max_jump_frame,seconds=time.monotonic()-began,
        target_body_motion_unchanged=True)
    (out / 'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','selected_frames','garment_vertices','garment_faces','max_adjacent_garment_vertex_step_m','seconds']}))


if __name__=='__main__': main()
