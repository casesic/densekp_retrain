"""Adapt the CC0 Poly Haven scene without modifying the original .blend.

blender -b --disable-autoexec assets/scenes/photoreal/hidden_alley/ph_hidden_alley.blend \
  --python tools/scene/prepare_hidden_alley.py -- --samples 32 --width 1600
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy



def main():
    ROOT=Path(__file__).resolve().parents[2]
    ASSET=ROOT/'assets/scenes/photoreal/hidden_alley'
    OUT=ROOT/'outputs/scenes/photoreal_hidden_alley'
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--width',type=int,default=2048)
    p.add_argument('--skip-render',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True,exist_ok=True)
    scene=bpy.data.scenes['Scene']
    bpy.context.window.scene=scene

    # Resolve the supplied maps by basename, recording all missing dependencies.
    files={f.name:f for f in (ASSET/'textures').rglob('*') if f.is_file()}
    missing=[]
    for img in bpy.data.images:
        if img.source!='FILE' or img.packed_file:
            continue
        basename=img.filepath.replace('\\','/').rsplit('/',1)[-1]
        original=Path(bpy.path.abspath(img.filepath))
        found=original if original.is_file() else files.get(basename)
        if found:
            img.filepath=str(found)
        else:
            missing.append({'image':img.name,'path':img.filepath})
    if missing:
        (OUT/'missing_images.json').write_text(json.dumps(missing,indent=2))
        raise RuntimeError(f'{len(missing)} missing textures; refusing incomplete render')

    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=a.samples
    scene.cycles.use_denoising=True
    scene.cycles.use_adaptive_sampling=True
    scene.cycles.adaptive_threshold=.025
    scene.cycles.max_bounces=8
    scene.render.threads_mode='FIXED'
    scene.render.threads=16
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False
    # The artist's compositor references a separate Fog render scene. Disable this
    # screen-space overlay so both perspective and spherical images use 3D lighting.
    scene.render.use_compositing=False
    scene.render.use_sequencer=False
    scene.render.use_border=False
    scene.render.use_crop_to_border=False
    scene['source_url']='https://blog.polyhaven.com/hidden-alley/'
    scene['asset_credit']='Poly Haven and Hidden Alley community artists; CC0'
    scene['adaptation']='Original geometry, materials and lighting; portable texture paths and panorama camera. Original source .blend preserved.'
    camera=scene.camera
    camera.name='Camera_Artist_Perspective'
    camera.data.clip_end=1000
    coll=bpy.data.collections.new('Render_Cameras')
    scene.collection.children.link(coll)
    pano=camera.copy()
    pano.data=camera.data.copy()
    pano.name='Camera_Equirectangular_360'
    coll.objects.link(pano)
    pano.data.type='PANO'
    pano.data.panorama_type='EQUIRECTANGULAR'
    pano.rotation_euler=(math.pi/2,0,0)
    pano.location.z=1.65
    for obj in scene.objects:
        obj.pass_index=0

    scene.camera=pano
    scene.render.resolution_x=a.width
    scene.render.resolution_y=a.width//2
    scene.render.filepath=str(OUT/'panorama.png')
    # Keep all maps relative to the adapted scene, avoiding duplicate packed files.
    for img in bpy.data.images:
        if img.source=='FILE' and img.filepath:
            img.filepath=bpy.path.relpath(img.filepath,start=str(ASSET))
    bpy.ops.wm.save_as_mainfile(filepath=str(ASSET/'hidden_alley_ready.blend'))
    report={'source':scene['source_url'],'credit':scene['asset_credit'],
            'source_blend':'ph_hidden_alley.blend','adapted_blend':'hidden_alley_ready.blend',
            'objects':len(scene.objects),'materials':len(bpy.data.materials),
            'images':len([i for i in bpy.data.images if i.source=='FILE']),
            'missing_images':missing,'samples':a.samples,'device':'CPU',
            'camera_perspective':list(camera.location),'camera_panorama':list(pano.location),
            'compositor_disabled':True,'texture_paths':'relative; keep textures directory next to blend'}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not a.skip_render:
        for cam,name,w,h in [(camera,'perspective.png',a.width,round(a.width*9/16)),
                             (pano,'panorama.png',a.width,a.width//2)]:
            scene.camera=cam
            scene.render.resolution_x=w
            scene.render.resolution_y=h
            scene.render.filepath=str(OUT/name)
            print('RENDER_START',name,flush=True)
            bpy.ops.render.render(write_still=True)
            print('RENDER_DONE',name,flush=True)
    print('HIDDEN_ALLEY_COMPLETE',json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
