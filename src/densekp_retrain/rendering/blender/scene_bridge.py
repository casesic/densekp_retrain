"""bpy-side scene bridge: link a .blend scene, control visibility and ground.

Runs inside Blender (imported by tools/fisheye/render_equirectangular_scene_source.py).
Scene objects always get pass_index=0 so the person mask (pass_index 1 body,
2 clothing) stays person-only, matching the Phase 17 warp pipeline.
"""

import bpy

GROUND_KEYWORDS = (
    "ground",
    "floor",
    "plane",
    "terrain",
    "pavement",
    "concrete",
    "yard",
    "courtyard",
    "road",
)


def link_blend_scene(blend_path, collection=None):
    before = {o.name for o in bpy.data.objects}
    requested = []
    with bpy.data.libraries.load(str(blend_path), link=False) as (src, dst):
        if collection:
            if collection not in src.collections:
                raise RuntimeError(
                    f"collection {collection!r} not in {blend_path}: {src.collections}"
                )
            dst.collections = [collection]
        else:
            dst.objects = list(src.objects)
            requested = [n for n in dst.objects if n]
    linked = []
    if collection:
        for coll in [
            c for c in bpy.data.collections if c.name == collection and c.library is not None
        ]:
            if coll.name not in {c.name for c in bpy.context.scene.collection.children}:
                bpy.context.scene.collection.children.link(coll)
            linked += [o.name for o in coll.all_objects]
    else:
        for name in requested:
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.name not in before:
                bpy.context.scene.collection.objects.link(obj)
                linked.append(name)
    if not linked:
        raise RuntimeError(f"no objects linked from {blend_path} (collection={collection})")
    return sorted(set(linked))


def set_visibility(names, hide_objects, ground_mode, ground_keywords=GROUND_KEYWORDS):
    hidden = []
    for name in names:
        obj = bpy.data.objects[name]
        hide = name in hide_objects or (
            ground_mode == "remove_ground" and any(k in name.lower() for k in ground_keywords)
        )
        obj.hide_render = hide
        obj.pass_index = 0
        if hide:
            hidden.append(name)
    return hidden


def scene_bounds(names):
    import mathutils

    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for name in names:
        obj = bpy.data.objects[name]
        if obj.type != "MESH" or obj.hide_render:
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
    if lo[0] == float("inf"):
        return None
    return {"min": lo, "max": hi, "size": [hi[i] - lo[i] for i in range(3)]}
