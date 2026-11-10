"""Optional sequence sidecars, including explicit logical/stored raster layout.

JSON paths resolve against the sidecar directory, never against the NPZ directory.
Validation reads tables/file metadata by default; image headers and packed metadata
are opt-in. It does not decode images, load models or certify exposure/identity.
"""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from .boxes import BBOX_KINDS, CROP_EXPANSIONS, crop_from_box
from densekp.core.datasets.image_layout import LAYOUT_FIELDS,parse_image_layout,read_layouts
from densekp_retrain.imaging.regions import require_crop_in_region

SCHEMA = "densekp_sequence_index_v1"
OBSERVATION_STATES = {"observed", "camera_missing", "person_absent", "empty"}


class SequenceIndexError(ValueError):
    """First invalid contract condition, with a stable machine-readable code."""
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _require(condition, code, detail=""):
    if not condition:
        raise SequenceIndexError(code, detail)


def _integer(value):
    return type(value) is int


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def exact_time(time_tick, time_base):
    """Integer ticks times rational seconds/tick; no float or filename matching."""
    _require(_integer(time_tick), "invalid_time_tick")
    _require(isinstance(time_base, list) and len(time_base) == 2
             and all(_integer(v) and v > 0 for v in time_base), "invalid_time_base")
    base = Fraction(*time_base)
    _require([base.numerator, base.denominator] == time_base, "noncanonical_time_base")
    return time_tick * base


def _unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        _require(value not in result, "duplicate_" + key, str(value))
        result[value] = row
    return result


def validate_sequence_index(episode, index, *, base_dir, check_files=True,
                            check_image_headers=False, check_packed=False):
    """Validate one sequence and one complete packed archive's row mapping.

    Unpacked root boxes may have packed_row_id=null. Missing observations have no
    boxes. All expected frame/person/camera slots must be explicit. No sidecar is
    required by legacy Dataset callers. Sync and identity remain evidence claims.
    """
    try:
        return _validate(episode, index, Path(base_dir), check_files,
                         check_image_headers, check_packed)
    except SequenceIndexError:
        raise
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise SequenceIndexError("malformed_structure", str(error)) from error


def _validate(ep, table, base, check_files, check_headers, check_packed):
    _require(ep["schema"] == table["schema"] == SCHEMA, "schema_mismatch")
    for key in ("episode_id", "sequence_id"):
        _require(_text(ep[key]) and ep[key] == table[key], "sequence_identity_mismatch", key)
    if 'scene_episode_id' in ep or 'rig_id' in ep or 'scene_episode_id' in table or 'rig_id' in table:
        _require(_text(ep.get('scene_episode_id')) and _text(ep.get('rig_id'))
                 and ep['scene_episode_id'] == table.get('scene_episode_id')
                 and ep['rig_id'] == table.get('rig_id'), 'scene_rig_identity_mismatch')
        _require(all(camera.get('rig_id') == ep['rig_id'] for camera in ep['cameras']),
                 'camera_rig_identity_mismatch')
    kind = ep["data_kind"]
    _require(kind in {"synthetic", "real", "synthetic_test_only"}, "invalid_data_kind")
    synthetic = kind != "real"
    _require(ep["temporal_kind"] in {"static", "sequence"}, "invalid_temporal_kind")
    _require(ep["time_semantics"] in {"static_anchor", "synthetic_clock", "nominal_index", "measured_pts"},
             "invalid_time_semantics")
    exact_time(0, ep["time_base"])
    _require(_text(ep["time_origin"]), "missing_time_origin")
    files = set()
    json_cache, header_cache, annotation_ids = {}, {}, {}

    def path(value):
        _require(_text(value), "missing_dependency_path")
        raw = Path(value)
        resolved = (base / raw).resolve()
        # Compact live inputs only. Historical recipe text is not recursively
        # interpreted as live dependencies (it may describe deleted EXRs).
        forbidden = {"source", "_temporary_ds_source", "<ephemeral-source-dir>"}
        _require(not forbidden.intersection(raw.parts + resolved.parts)
                 and not any(p.startswith("_temporary_") for p in raw.parts + resolved.parts)
                 and resolved.suffix.lower() != ".exr", "temporary_dependency", value)
        for root in ep.get("ephemeral_roots", []):
            temp = (base / root).resolve()
            _require(not resolved.is_relative_to(temp), "temporary_dependency", value)
        if check_files or check_headers or check_packed:
            _require(resolved.is_file(), "missing_long_term_file", str(resolved))
        files.add(str(resolved))
        return resolved

    def read_json(value):
        p = path(value)
        if p not in json_cache:
            json_cache[p] = json.loads(p.read_text())
        return json_cache[p]

    def size(value):
        _require(isinstance(value, list) and len(value) == 2
                 and all(_integer(v) and v > 0 for v in value), "invalid_image_size")
        return value

    cameras = _unique(ep["cameras"], "camera_id")
    people = _unique(ep["people"], "person_id")
    assets = _unique(ep["assets"], "asset_id")
    _require(bool(cameras) and bool(people), "empty_roster")
    for camera in cameras.values():
        _require(_text(camera["camera_id"]), "invalid_camera_id")
        size(camera["image_size_wh"])
        path(camera["rig_ref"])
        if camera["calibration_ref"] is None:
            _require(_text(camera["calibration_unknown_reason"]), "unexplained_unknown_calibration")
        else:
            path(camera["calibration_ref"])
    for asset in assets.values():
        _require(_text(asset["asset_id"]), "invalid_asset_id")
        path(asset["asset_ref"])
        if asset["motion_source_ref"] is None:
            _require(_text(asset["motion_source_unknown_reason"]), "unexplained_unknown_motion_source")
        else:
            path(asset["motion_source_ref"])
    for person in people.values():
        _require(_integer(person["person_id"]), "invalid_person_id")
        path(person["source_ref"])
        if person["character_family_id"] is None:
            _require(_text(person["family_unknown_reason"]), "unexplained_unknown_family")
        else:
            _require(_text(person["character_family_id"]), "invalid_family_id")
        _require(person["asset_id"] in assets or (not synthetic and person["asset_id"] is None),
                 "unknown_asset_id")
        status = person["identity_status"]
        _require(status in ({"synthetic_instance"} if synthetic else {"verified", "track_only", "unknown"}),
                 "invalid_identity_status")
        if status == "verified":
            path(person["identity_evidence_ref"])
        elif not synthetic:
            _require(_text(person["identity_reason"]), "unexplained_identity_status")
    for ref in ep["source_refs"]:
        path(ref)
    _require(bool(ep["source_refs"]), "missing_source_refs")
    if ep.get("recipe_ref") is not None:
        path(ep["recipe_ref"])
    sync = ep["synchronization"]
    status = sync["status"]
    _require(status in {"synthetic_same_state", "unverified", "verified_exposure"}, "invalid_sync_status")
    _require(_text(sync["basis"]), "missing_sync_basis")
    if status == "synthetic_same_state":
        _require(synthetic, "real_cannot_claim_synthetic_sync")
        path(sync["evidence_ref"])
    elif status == "verified_exposure":
        _require(not synthetic and ep["time_semantics"] == "measured_pts"
                 and sync["basis"] == "exposure_measurement", "unsupported_exposure_sync_claim")
        path(sync["evidence_ref"])
    else:
        _require(_text(sync["reason"]), "unexplained_unverified_sync")
        if sync.get("evidence_ref") is not None:
            path(sync["evidence_ref"])

    timeline = ep["timeline"]
    first, step, count = (timeline[k] for k in ("first_frame_index", "frame_step", "frame_count"))
    _require(_integer(first) and first >= 0 and _integer(step) and step > 0
             and _integer(count) and count > 0, "invalid_timeline")
    tick_step = timeline["expected_tick_step"]
    _require(tick_step is None or (_integer(tick_step) and tick_step > 0), "invalid_tick_step")
    frames = _unique(table["frames"], "frame_index")
    _require(len(frames) == count, "missing_frame_record")
    if ep["temporal_kind"] == "static":
        _require(count == 1 and ep["time_semantics"] == "static_anchor", "static_must_have_one_frame")
    else:
        _require(count >= 2 and ep["time_semantics"] != "static_anchor", "invalid_sequence_timeline")
    selected=None
    if ep.get('selected_timeline_ref') is not None and (check_files or check_headers or check_packed):
        selected=read_json(ep['selected_timeline_ref'])
        original=read_json(ep['original_timeline_ref'])
        _require(len(selected)==count,'selected_timeline_count_conflict')
        from .dynamic_scene import checked_matrices
        matrices=np.load(path(ep['scene_matrices_ref']),allow_pickle=False)
        if 'scene_matrix_person_ids' in ep:
            ids=ep['scene_matrix_person_ids']
            _require(len(ids)==len(people) and set(ids)==set(people),'scene_matrix_person_identity_conflict')
            _require(matrices.shape==(count,len(ids),4,4),'scene_matrix_shape_conflict')
            recipe=read_json(ep['recipe_ref'])
            for j,pid in enumerate(ids):
                checked_matrices(matrices[:,j],count)
                person=next(p for p in recipe['people'] if p['person_id']==pid)
                _require(np.array_equal(matrices[:,j],person['scene_from_model']),'scene_matrix_recipe_conflict')
        else:checked_matrices(matrices,count)
        originals={r['asset_frame']:r for r in original}
        origin=selected[0]['source_time_tick']
        for k,row in enumerate(selected):
            ref=originals[row['asset_frame']]
            _require(row['frame_index']==k and row['episode_time_tick']==row['source_time_tick']-origin,'selected_time_origin_conflict')
            _require(all(row[key]==ref[key] for key in ['source_time_tick','source_frame','time_base']),'selected_source_time_conflict')
    person_selected=None
    if ep.get('time_mapping_schema') is not None and (check_files or check_headers or check_packed):
        from .person_time import SCHEMA as TIME_SCHEMA,map_timeline
        from .dynamic_scene import person_matrices,frame_record
        _require(ep['time_mapping_schema']==TIME_SCHEMA,'unsupported_person_time_schema')
        recipe=read_json(ep['recipe_ref']);clock=read_json(ep['episode_timeline_ref'])
        _require(clock==recipe['episode_timeline'] and len(clock)==count,'mixed_clock_conflict')
        ids=ep['scene_matrix_person_ids'];matrices=np.load(path(ep['scene_matrices_ref']),allow_pickle=False)
        _require(ids==[p['person_id'] for p in recipe['people']] and set(ids)==set(people),'mixed_matrix_person_conflict')
        _require(matrices.shape==(count,len(ids),4,4),'mixed_matrix_shape_conflict')
        person_selected={}
        for j,p in enumerate(recipe['people']):
            pid=str(p['person_id']);rows=read_json(ep['person_timeline_refs'][pid]);source=read_json(ep['source_timeline_refs'][pid])
            _require(rows==map_timeline(source,clock,origin_record=rows[0]['source_origin_record']) and rows==recipe['person_timelines'][pid],'mixed_source_mapping_conflict')
            _require(np.array_equal(matrices[:,j],person_matrices(p,count)),'mixed_root_matrix_conflict')
            person_selected[p['person_id']]=rows
        for k in range(count):frame_record(recipe,k)
    last_tick = None
    for i, frame in enumerate(table["frames"]):
        _require(_integer(frame["frame_index"]) and frame["frame_index"] == first + i * step,
                 "frame_order_or_gap")
        tick = frame["time_tick"]
        exact_time(tick, ep["time_base"])
        if selected is not None:_require(tick==selected[i]['episode_time_tick'],'selected_frame_tick_conflict')
        if last_tick is not None:
            _require(tick > last_tick, "nonmonotonic_time")
            _require(tick_step is None or tick - last_tick == tick_step, "time_step_conflict")
        last_tick = tick
        _require(frame["status"] in {"available", "missing"}, "invalid_frame_status")
        if frame["status"] == "missing":
            _require(_text(frame["reason"]), "unexplained_missing_frame")

    observations = _unique(table["observations"], "observation_id")
    slots, camera_states, asset_frames = {}, {}, {}
    observed_count = Counter()
    file_fields = ("image", "annotations", "valid_mask", "person_mask", "raster_manifest")
    for oid, obs in observations.items():
        _require(_text(oid), "invalid_observation_id")
        if 'scene_episode_id' in ep:
            _require(obs.get('scene_episode_id') == ep['scene_episode_id']
                     and obs.get('rig_id') == ep['rig_id'], 'observation_scene_rig_identity_mismatch', oid)
        fid, pid, cid = obs["frame_index"], obs["person_id"], obs["camera_id"]
        _require(_integer(fid) and _integer(pid) and fid in frames and pid in people and cid in cameras,
                 "unknown_observation_slot")
        slot = (fid, pid, cid)
        _require(slot not in slots, "duplicate_observation", str(slot))
        slots[slot] = oid
        _require(_integer(obs["time_tick"]) and obs["time_tick"] == frames[fid]["time_tick"],
                 "observation_time_conflict", oid)
        if selected is not None:
            row=selected[fid]
            _require(obs['source_time_tick']==row['source_time_tick'] and obs['source_time_base']==row['time_base'] and obs['asset_frame_index']==row['asset_frame'] and obs['source_frame_index']==row['source_frame'],'observation_selected_time_conflict')
            _require(path(obs['selected_timeline_ref'])==path(ep['selected_timeline_ref']) and path(obs['scene_matrices_ref'])==path(ep['scene_matrices_ref']) and obs['scene_matrix_row']==fid and obs['timeline_row']==fid,'observation_scene_matrix_conflict')
            if 'scene_matrix_person_ids' in ep:
                _require(obs.get('scene_matrix_person_index')==ep['scene_matrix_person_ids'].index(pid),'observation_scene_person_conflict')
        if person_selected is not None:
            row=person_selected[pid][fid]
            _require(obs['time_tick']==row['episode_time_tick'] and row['time_base']==ep['time_base'],'mixed_observation_clock_conflict')
            for key in ['source_time_tick','source_time_base','source_record_index','source_origin_record','source_time_origin_tick','source_time_origin_base','mapping_policy','quantization_error_seconds','local_interval_seconds']:
                _require(obs[key]==row[key],'mixed_observation_source_conflict',key)
            _require(obs['asset_frame_index']==row['asset_frame'] and obs['source_frame_index']==row['source_frame'],'mixed_observation_asset_frame_conflict')
            _require(path(obs['person_timeline_ref'])==path(ep['person_timeline_refs'][str(pid)]) and obs['timeline_row']==fid,'mixed_observation_timeline_ref_conflict')
            _require(path(obs['scene_matrices_ref'])==path(ep['scene_matrices_ref']) and obs['scene_matrix_row']==fid and obs['scene_matrix_person_index']==ep['scene_matrix_person_ids'].index(pid),'mixed_observation_matrix_ref_conflict')
        person = people[pid]
        for key in ("asset_id", "source_ref", "character_family_id"):
            _require(obs[key] == person[key], "person_source_conflict", key)
        asset_frame = obs["asset_frame_index"]
        if asset_frame is None:
            _require(not synthetic and obs["asset_frame_status"] in {"unknown", "not_applicable"}
                     and _text(obs["asset_frame_reason"]), "unexplained_null_asset_frame")
        else:
            _require(_integer(asset_frame) and asset_frame >= 0 and obs["asset_frame_status"] == "known"
                     and obs["asset_id"] in assets, "invalid_asset_frame")
        if synthetic:
            same_person = (fid, pid)
            if same_person in asset_frames:
                _require(asset_frames[same_person] == asset_frame, "cross_camera_asset_frame_conflict")
            asset_frames[same_person] = asset_frame
        state = obs["status"]
        _require(state in OBSERVATION_STATES, "invalid_observation_status")
        observed_count[state] += 1
        if state != "observed":
            _require(_text(obs["reason"]), "unexplained_missing_observation")
        if frames[fid]["status"] == "missing":
            _require(state == "camera_missing", "missing_frame_has_observation")
        if state == "camera_missing":
            _require(all(obs[k] is None for k in file_fields) and obs["annotation_person_index"] is None
                     and obs["image_size_wh"] is None, "missing_camera_has_files")
        else:
            _require(size(obs["image_size_wh"]) == cameras[cid]["image_size_wh"], "image_size_conflict")
            layout=parse_image_layout(obs)
            if layout is not None:
                _require(list(layout.full_size_wh)==obs['image_size_wh'],'logical_image_size_conflict')
                path(obs['raster_storage_ref']);path(obs['physical_mask_ref'])
                if check_files or check_headers or check_packed:
                    stored=read_json(obs['raster_storage_ref'])
                    _require(all(np.array_equal(obs[k],stored[k]) for k in LAYOUT_FIELDS),'storage_metadata_conflict')
                    _require(path(stored['physical_mask_ref'])==path(obs['physical_mask_ref']),'physical_mask_reference_conflict')
                    if layout.kind=='roi_image':
                        _require(stored['render_coverage_xyxy']==list(layout.roi_xyxy),'storage_coverage_conflict')
                        coverage=read_json(obs['render_coverage_ref'])
                        _require(coverage['high_quality_region_xyxy']==list(layout.roi_xyxy),'render_coverage_conflict')

            for field in file_fields:
                if state == "person_absent" and field == "person_mask":
                    _require(obs[field] is None, "absent_person_has_mask")
                else:
                    path(obs[field])
            idx = obs["annotation_person_index"]
            _require((state == "person_absent" and idx is None)
                     or (state != "person_absent" and _integer(idx) and idx >= 0), "invalid_annotation_index")
            if check_files or check_headers or check_packed:
                manifest = read_json(obs["raster_manifest"])
                _require(manifest["final_erp_resolution"] == obs["image_size_wh"], "manifest_size_conflict")
            if check_packed and idx is not None:
                annotation_path = path(obs["annotations"])
                if annotation_path not in annotation_ids:
                    with np.load(annotation_path, allow_pickle=False) as archive:
                        annotation_ids[annotation_path] = archive["person_ids"]
                ids = annotation_ids[annotation_path]
                _require(idx < len(ids) and int(ids[idx]) == pid, "original_annotation_identity_mismatch")
            if check_headers:
                from PIL import Image
                for field in ("image", "valid_mask", "person_mask"):
                    if obs[field] is None:
                        continue
                    p = path(obs[field])
                    if p not in header_cache:
                        with Image.open(p) as image:
                            header_cache[p] = list(image.size)
                    _require(header_cache[p] == (list(layout.stored_size_wh) if layout else obs["image_size_wh"]), "file_header_size_conflict", str(p))
                if layout is not None:
                    p=path(obs['physical_mask_ref'])
                    if p not in header_cache:
                        with Image.open(p) as image:header_cache[p]=list(image.size)
                    _require(header_cache[p]==obs['image_size_wh'],'physical_mask_size_conflict')
        camera_key = (fid, cid)
        camera_value = (state == "camera_missing", obs["image"], obs["annotations"], obs["valid_mask"], obs["raster_manifest"],str({k:obs.get(k) for k in LAYOUT_FIELDS}))
        _require(camera_key not in camera_states or camera_states[camera_key] == camera_value,
                 "inconsistent_camera_observation")
        camera_states[camera_key] = camera_value
    _require(len(slots) == len(frames) * len(people) * len(cameras), "missing_observation_slot")

    boxes = _unique(table["boxes"], "box_id")
    packed = table["packed"]
    _require(_integer(packed["row_count"]) and packed["row_count"] >= 0, "invalid_packed_count")
    packed_path = path(packed["path"]) if packed["path"] is not None else None
    _require(packed_path is not None or packed["row_count"] == 0, "missing_packed_archive")
    row_map = {}
    box_observations = set()
    for bid, box in boxes.items():
        _require(_text(bid) and box["observation_id"] in observations, "unknown_box_observation")
        obs = observations[box["observation_id"]]
        _require(obs["status"] == "observed", "box_on_empty_observation")
        box_observations.add(box["observation_id"])
        _require(box["bbox_kind"] in BBOX_KINDS and _text(box["bbox_source"]), "invalid_bbox_kind_or_source")
        _require(box["crop_policy"] in CROP_EXPANSIONS, "invalid_crop_policy")
        coords = np.asarray(box["bbox_xyxy"], dtype=float)
        _require(coords.shape == (4,) and np.isfinite(coords).all()
                 and (coords[2:] > coords[:2]).all(), "invalid_bbox")
        w, h = obs["image_size_wh"]
        _require((coords[:2] >= 0).all() and coords[2] <= w and coords[3] <= h, "bbox_outside_raster")
        path(box["origin_ref"])
        _require(_integer(box["origin_row"]) and box["origin_row"] >= 0, "invalid_origin_row")
        if check_files or check_headers or check_packed:
            origin = read_json(box["origin_ref"])
            origin_rows = origin if isinstance(origin, list) else origin["rows"]
            _require(box["origin_row"] < len(origin_rows), "origin_row_missing")
            original = origin_rows[box["origin_row"]]
            original_box = original.get("given_bbox_xyxy", original.get("bbox_xyxy_exclusive"))
            _require(np.array_equal(coords, original_box) and original["person_id"] == obs["person_id"],
                     "origin_box_mismatch")
            _require(original.get("camera", obs["camera_id"]) == obs["camera_id"], "origin_camera_mismatch")
            _require(original.get("bbox_kind", box["bbox_kind"]) == box["bbox_kind"], "origin_kind_mismatch")
        parent = box["parent_box_id"]
        if parent is not None:
            _require(parent in boxes and parent != bid, "missing_or_self_parent_box")
            _require(boxes[parent]["observation_id"] == box["observation_id"], "derivative_sequence_conflict")
            _require(box["bbox_kind"] == "synthetic_perturbation", "invalid_derivative_kind")
        elif box["bbox_kind"] == "synthetic_perturbation":
            raise SequenceIndexError("derivative_missing_parent")
        seen = {bid}
        while parent is not None:
            _require(parent in boxes, "missing_or_self_parent_box")
            _require(parent not in seen, "parent_cycle")
            seen.add(parent)
            parent = boxes[parent]["parent_box_id"]
        row = box["packed_row_id"]
        if 'scene_episode_id' in ep:
            _require(box.get('scene_episode_id') == ep['scene_episode_id']
                     and box.get('rig_id') == ep['rig_id'], 'box_scene_rig_identity_mismatch', bid)
        if row is not None:
            _require(_integer(row) and row >= 0 and row not in row_map, "duplicate_or_invalid_packed_row")
            row_map[row] = box
    _require(set(row_map) == set(range(packed["row_count"])), "packed_row_coverage_mismatch")
    _require(box_observations == {oid for oid, obs in observations.items() if obs["status"] == "observed"},
             "observed_slot_without_box")
    if check_packed and packed_path is not None:
        with np.load(packed_path, allow_pickle=False) as archive:
            # Only small metadata columns, never proj_verts or decoded images.
            fields = {key: archive[key] for key in ("imgname", "person_id", "camera_id", "center", "scale", "scene_episode_id", "rig_id", *LAYOUT_FIELDS) if key in archive}
        layouts=read_layouts(fields,packed["row_count"])
        _require(all(len(v) == packed["row_count"] for v in fields.values()), "archive_row_count_mismatch")
        for i, box in row_map.items():
            obs = observations[box["observation_id"]]
            _require(int(fields["person_id"][i]) == obs["person_id"]
                     and str(fields["camera_id"][i]) == obs["camera_id"], "packed_identity_mismatch", str(i))
            if 'scene_episode_id' in ep:
                _require(str(fields['scene_episode_id'][i]) == ep['scene_episode_id']
                         and str(fields['rig_id'][i]) == ep['rig_id'], 'packed_scene_rig_identity_mismatch', str(i))
            _require(path(str(fields["imgname"][i])) == path(obs["image"]), "packed_image_mismatch", str(i))
            obs_layout=parse_image_layout(obs)
            _require((layouts is None)==(obs_layout is None),'packed_storage_layout_missing',str(i))
            if layouts is not None:
                _require(layouts[i]==obs_layout,'packed_storage_layout_conflict',str(i))
            center, scale = crop_from_box(box["bbox_xyxy"], image_size_wh=obs["image_size_wh"],
                                         boundary_policy="erp_reject_seam", crop_policy=box["crop_policy"], min_side_px=2)
            if obs_layout is not None and obs_layout.kind=='roi_image':
                require_crop_in_region(center,scale,obs_layout.roi_xyxy,obs_layout.full_size_wh)
            _require(np.array_equal(center, fields["center"][i]) and np.array_equal(scale, fields["scale"][i]),
                     "packed_crop_mismatch", str(i))
    return dict(status="passed", schema=SCHEMA, data_kind=kind,
                temporal_kind=ep["temporal_kind"], frame_count=len(frames),
                missing_frame_count=sum(f["status"] == "missing" for f in frames.values()),
                observation_counts=dict(observed_count), box_count=len(boxes), packed_rows=len(row_map),
                dependency_files=len(files), image_headers_read=len(header_cache),
                declared_synchronization=status, exposure_sync_validated=False,
                identity_confirmed_by_validator=False, synthetic_state_consistent=synthetic,
                warning="Seed is provenance only, not split independence; table consistency is not motion quality")


def row_context(episode, index, packed_row_id):
    """Resolve a validated sidecar row, without modifying Dataset's return value."""
    _require(_integer(packed_row_id) and packed_row_id >= 0, "invalid_packed_row_id")
    box = next((b for b in index["boxes"] if b["packed_row_id"] == packed_row_id), None)
    _require(box is not None, "unknown_packed_row_id")
    obs = next(o for o in index["observations"] if o["observation_id"] == box["observation_id"])
    person = next(p for p in episode["people"] if p["person_id"] == obs["person_id"])
    return dict(episode_id=episode["episode_id"], sequence_id=episode["sequence_id"],
                time_base=episode["time_base"], time_semantics=episode["time_semantics"],
                synchronization=episode["synchronization"], person=person, observation=obs, box=box)
