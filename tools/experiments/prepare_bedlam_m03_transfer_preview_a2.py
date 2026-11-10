"""Freeze two-view comparison preview inputs for one BEDLAM2-to-M03 garment."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--accepted-asset', type=Path, required=True)
    ap.add_argument('--transfer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu-uuid', required=True)
    ap.add_argument('--diffuse', type=Path)
    ap.add_argument('--normal', type=Path)
    a = ap.parse_args()
    accepted = a.accepted_asset.resolve()
    transfer = a.transfer.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((accepted / 'preview_recipe.json').read_text())
    authority = json.loads((accepted / 'delivery_manifest.json').read_text())
    if authority['frame_count'] != 120:
        raise ValueError('frozen_M03_120_frame_asset_required')
    cloth = transfer / 'clothing_vertices.npy'
    cache = np.load(cloth, mmap_mode='r')
    if cache.shape[0] != 120: raise ValueError('transferred_cloth_timeline_mismatch')
    recipe = dict(baseline)
    recipe.update(source_clothing=str(transfer), clothing_cache=str(cloth),
                  native_clothing_material=dict(vertices_uv=str(transfer/'clothing_vertices_uv.npy'),
                      faces_uv=str(transfer/'clothing_faces_uv.npy'),
                      diffuse=str(a.diffuse.resolve()) if a.diffuse else None,
                      normal=str(a.normal.resolve()) if a.normal else None),
                  comparison_baseline=str((accepted/'preview_full/comparison.mp4').resolve()),
                  transfer_validation=str((transfer/'validation.json').resolve()))
    recipe['budget'] = dict(baseline['budget'], gpu_uuid=a.gpu_uuid, max_render_seconds=1800)
    recipe['probe_asset_frames'] = [0, 30, 60, 90, 119]
    recipe['preview_window'] = [0, 120]
    if not all(Path(v).is_file() for v in recipe['native_clothing_material'].values() if v):
        raise FileNotFoundError('native_material_dependency_missing')
    (out / 'preview_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    (out / 'timeline.json').write_text((accepted/'timeline.json').read_text())
    delivery = dict(schema='densekp_bedlam2_garment_transfer_candidate_a2_v1',
        asset_id='M03_KIT11_gr_eric_033_3XL_transfer_candidate',integration_only=True,
        character_family=authority['character_family'],motion_source_id=authority['motion_source_id'],
        source_group=authority['source_group'],frame_count=120,time_base=[1,30],
        target_body_manifest=str((accepted/'delivery_manifest.json').resolve()),
        target_body_vertices=authority['body_vertices'],target_body_faces=authority['body_faces'],
        target_R138=authority['body_R138'],target_head_transforms=authority['head_transforms'],
        target_groom=authority['groom_reference'],
        clothing_kind='retargeted_BEDLAM2_native_simulation_cache',
        original_body_motion='nl_2066_3XL/2205',original_outfit='gr_eric_033_3XL',
        original_clothing=str(Path(json.loads((transfer/'validation.json').read_text())['source_native_clothing']).resolve()),
        cloth_vertices=str(cloth),cloth_faces=str((transfer/'clothing_faces.npy').resolve()),
        cloth_uv_vertices=str((transfer/'clothing_vertices_uv.npy').resolve()),
        cloth_uv_faces=str((transfer/'clothing_faces_uv.npy').resolve()),
        garment_texture_01=recipe['native_clothing_material'],
        fixed_binding=str((transfer/'source_body_triangle_vertex_ids.npy').resolve()),
        approximation='source simulated cloth plus fixed source-body correspondence times target-minus-source body displacement; no new simulation',
        body_motion_unchanged=True,source_time_unchanged=True,dressed_human_status='pending_user',
        public_scene_status='not_consumed',license='BEDLAM2 non-commercial research; see official license')
    (out/'delivery_manifest.json').write_text(json.dumps(delivery,indent=2)+'\n')
    (out/'validation.json').write_text((transfer/'validation.json').read_text())
    (out/'a1_adapter_check.json').write_text(json.dumps(dict(status='pending_public_native_UV_material_consumer',
        scene_consumed=False,body_motion_authority_unchanged=True),indent=2)+'\n')
    head = np.load(authority['head_transforms'], mmap_mode='r')
    for stage, ids in [('probe', np.array(recipe['probe_asset_frames'], np.int32)),
                       ('full', np.arange(120, dtype=np.int32))]:
        folder = out/stage;folder.mkdir()
        np.save(folder/'asset_frames.npy', ids)
        np.save(folder/'head_transforms.npy', head[ids])
    print(json.dumps(dict(output=str(out),probe=recipe['probe_asset_frames'],
                          garment_texture_applied=bool(a.diffuse),source_body_motion_unchanged=True)))


if __name__ == '__main__': main()
