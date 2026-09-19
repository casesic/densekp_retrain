"""Scene asset loading interface for Phase 18 3D scene integration.

A scene asset is a static Blender .blend file with an outdoor house/courtyard
build in the same world convention as Phase 16/17: meters, Z-up inside Blender,
body resting on Z=0. No scene asset ships with this repository; the loader
validates the spec and reports exactly what has to be provided instead of
faking a scene.

Required asset content (documented here so the asset can be prepared offline):
  - at least one mesh object forming the static environment (house/courtyard)
  - optionally a ground object; if absent set ground_mode='add_procedural_ground'
  - units: 1 Blender unit = 1 meter, scene sized >= 24x24 m around the origin
  - no cameras/lights are required; lighting comes from --hdri or default sun
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

GROUND_KEYWORDS=('ground','floor','plane','terrain','pavement','concrete','yard','courtyard','road')

class SceneAssetMissing(RuntimeError): pass

@dataclass(frozen=True)
class SceneAssetSpec:
    blend_path: Path
    collection: str|None=None
    hide_objects: tuple[str,...]=()
    ground_mode: str='scene_native'
    body_location: tuple[float,float,float]=(0.,0.,0.)
    body_yaw_deg: float=0.

    @staticmethod
    def from_dict(value):
        spec=SceneAssetSpec(blend_path=Path(value['blend_path']),collection=value.get('collection'),hide_objects=tuple(value.get('hide_objects',())),ground_mode=value.get('ground_mode','scene_native'),body_location=tuple(value.get('body_location',(0.,0.,0.))),body_yaw_deg=float(value.get('body_yaw_deg',0.)))
        if spec.ground_mode not in ('scene_native','remove_ground','add_procedural_ground'): raise ValueError(f'unknown ground_mode: {spec.ground_mode}')
        return spec

    @staticmethod
    def from_file(path):
        raw=json.loads(Path(path).read_text()); raw.pop('notes',None); return SceneAssetSpec.from_dict(raw)

    def validate(self, root=Path('.')):
        if not self.blend_path.is_file():
            raise SceneAssetMissing(f'scene asset not found: {self.blend_path.resolve()}\n'
                'Phase 18 requires a static outdoor house/courtyard .blend:\n'
                '  - mesh objects for the static environment, meters, scene >= 24x24 m around origin\n'
                '  - ground object optional (ground_mode=add_procedural_ground otherwise)\n'
                '  - configure configs/fisheye_scene_p0.json -> blend_path, rerun tools/run_fisheye_scene_p0.py')
        if self.ground_mode!='scene_native' and self.hide_objects: pass
        return self

    def describe(self):
        return {'blend_path':str(self.blend_path.resolve()),'collection':self.collection,'hide_objects':list(self.hide_objects),'ground_mode':self.ground_mode,'body_location':list(self.body_location),'body_yaw_deg':self.body_yaw_deg}
