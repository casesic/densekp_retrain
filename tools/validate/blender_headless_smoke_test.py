"""用法：blender -b --python tools/validate/blender_headless_smoke_test.py"""

import json
import os
import time
from pathlib import Path

import bpy


output_dir = Path(os.environ.get("DENSEKP_SMOKE_OUTPUT", "outputs/headless_smoke")).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=(0, 0, 0))
sphere = bpy.context.object
material = bpy.data.materials.new("SmokeMaterial")
material.diffuse_color = (0.08, 0.35, 0.8, 1.0)
sphere.data.materials.append(material)

bpy.ops.object.light_add(type="AREA", location=(2.5, -2.5, 4.0))
bpy.context.object.data.energy = 700
bpy.context.object.data.shape = "DISK"
bpy.context.object.data.size = 3.0

bpy.ops.object.camera_add(location=(0, -4, 0))
camera = bpy.context.object
camera.rotation_euler = (1.5707963268, 0, 0)
bpy.context.scene.camera = camera

scene = bpy.context.scene
requested_engine = os.environ.get("DENSEKP_SMOKE_ENGINE", "BLENDER_EEVEE")
scene.render.engine = requested_engine
devices = []
if requested_engine == "CYCLES":
    cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
    requested_backend = os.environ.get("DENSEKP_SMOKE_DEVICE", "OPTIX")
    cycles_preferences.compute_device_type = requested_backend
    cycles_preferences.get_devices()
    for device in cycles_preferences.devices:
        if device.type == requested_backend:
            device.use = True
        devices.append({"name": device.name, "type": device.type, "use": bool(device.use)})
    scene.cycles.device = "GPU"
    scene.cycles.samples = 16
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(output_dir / "blender_rgb.png")

started = time.perf_counter()
bpy.ops.render.render(write_still=True)
report = {
    "display": os.environ.get("DISPLAY"),
    "engine": scene.render.engine,
    "render_seconds": time.perf_counter() - started,
    "device": "GPU" if requested_engine == "CYCLES" else "CPU/default",
    "cycles_devices": devices,
}
(output_dir / "blender_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
