#!/usr/bin/env python3
"""审计 SMPL-X body mesh 与 UV OBJ 的 face-corner 拓扑兼容性。"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_obj(path: Path):
    vertices, uv, faces, faces_uv = [], [], [], []
    objects, groups, materials = [], [], []
    for raw in path.read_text(errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("v "):
            vertices.append([float(x) for x in s.split()[1:4]])
        elif s.startswith("vt "):
            uv.append([float(x) for x in s.split()[1:3]])
        elif s.startswith("f "):
            corners = s.split()[1:]
            if len(corners) != 3:
                raise ValueError(f"只支持三角形，发现 {len(corners)} 边面")
            vi, ti = [], []
            for corner in corners:
                fields = corner.split("/")
                vi.append(int(fields[0]) - 1)
                ti.append(int(fields[1]) - 1 if len(fields) > 1 and fields[1] else -1)
            faces.append(vi); faces_uv.append(ti)
        elif s.startswith("o "): objects.append(s[2:])
        elif s.startswith("g "): groups.append(s[2:])
        elif s.startswith("usemtl "): materials.append(s[7:])
    return (np.asarray(vertices), np.asarray(uv), np.asarray(faces, dtype=np.int64),
            np.asarray(faces_uv, dtype=np.int64), objects, groups, materials)


def components(vertex_count: int, faces: np.ndarray):
    parent = np.arange(vertex_count)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for a, b, c in faces:
        union(a, b); union(a, c)
    used = np.unique(faces)
    counts = {}
    for i in used: counts[find(i)] = counts.get(find(i), 0) + 1
    return sorted(counts.values(), reverse=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uv-obj", type=Path, required=True)
    p.add_argument("--geometry-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ov, uv, of, ofuv, objects, groups, materials = read_obj(a.uv_obj)
    gf = np.load(a.geometry_dir / "faces.npy")
    gv = np.load(a.geometry_dir / "vertices.npy")
    finite = bool(np.isfinite(uv).all())
    uv_valid = bool(ofuv.size and ofuv.min() >= 0 and ofuv.max() < len(uv))
    geometry_valid = bool(of.size and of.min() >= 0 and of.max() < gv.shape[1])
    face_match = bool(gf.shape == of.shape and np.array_equal(gf, of))
    cc = components(len(ov), of)
    report = {
        "uv_obj": str(a.uv_obj),
        "body_vertices_shape": list(gv.shape),
        "body_faces_shape": list(gf.shape),
        "obj_vertices_shape": list(ov.shape),
        "obj_uv_shape": list(uv.shape),
        "obj_faces_shape": list(of.shape),
        "obj_faces_uv_shape": list(ofuv.shape),
        "geometry_indices_valid": geometry_valid,
        "uv_indices_valid": uv_valid,
        "uv_finite": finite,
        "uv_range_u": [float(uv[:, 0].min()), float(uv[:, 0].max())],
        "uv_range_v": [float(uv[:, 1].min()), float(uv[:, 1].max())],
        "geometry_faces_exact_match": face_match,
        "topology_compatible": bool(face_match and geometry_valid and uv_valid and finite),
        "objects": objects,
        "groups": groups,
        "materials": materials,
        "connected_component_count": len(cc),
        "connected_component_vertex_counts": cc,
        "eye_material_region_explicit": bool(any("eye" in x.lower() for x in materials + groups + objects)),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["topology_compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
