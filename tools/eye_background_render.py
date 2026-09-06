"""Material-only eyes and linear differential shadow compositing for P0."""
import json
from pathlib import Path

import bpy
import numpy as np


def procedural_eyes(mesh, faces, labels, body_component, mapping_path):
    # This supplied SMPL albedo contains photographed eyes above the native
    # SMPL-X apertures. Locally sample adjacent forehead skin; never change UVs
    # or the dataset texture. Feather only these two albedo regions.
    skin = bpy.data.materials.get('OfficialBodySkin')
    nodes, links = skin.node_tree.nodes, skin.node_tree.links
    original = next(n for n in nodes if n.type == 'TEX_IMAGE')
    if Path(original.image.filepath).name != 'skin_m_white_01_ALB.png':
        raise RuntimeError('Local eye albedo correction is calibrated only for skin_m_white_01_ALB.png')
    uv = nodes.new('ShaderNodeTexCoord')
    offset = nodes.new('ShaderNodeVectorMath'); offset.operation = 'ADD'
    links.new(uv.outputs['UV'], offset.inputs[0]); offset.inputs[1].default_value = (0, .075, 0)
    adjacent = nodes.new('ShaderNodeTexImage'); adjacent.image = original.image
    links.new(offset.outputs['Vector'], adjacent.inputs['Vector'])
    color = original.outputs['Color']
    for center in ((.239, .821, 0), (.354, .821, 0)):
        delta = nodes.new('ShaderNodeVectorMath'); delta.operation = 'SUBTRACT'
        links.new(uv.outputs['UV'], delta.inputs[0]); delta.inputs[1].default_value = center
        scale = nodes.new('ShaderNodeVectorMath'); scale.operation = 'MULTIPLY'
        links.new(delta.outputs['Vector'], scale.inputs[0]); scale.inputs[1].default_value = (1/.037, 1/.023, 0)
        radius = nodes.new('ShaderNodeVectorMath'); radius.operation = 'LENGTH'
        links.new(scale.outputs['Vector'], radius.inputs[0])
        feather = nodes.new('ShaderNodeMapRange'); feather.clamp = True; feather.interpolation_type = 'SMOOTHSTEP'
        links.new(radius.outputs['Value'], feather.inputs['Value'])
        feather.inputs['From Min'].default_value = .65; feather.inputs['From Max'].default_value = 1.
        feather.inputs['To Min'].default_value = 1.; feather.inputs['To Max'].default_value = 0.
        mix = nodes.new('ShaderNodeMixRGB'); links.new(feather.outputs[0], mix.inputs[0])
        links.new(color, mix.inputs[1]); links.new(adjacent.outputs['Color'], mix.inputs[2]); color = mix.outputs[0]
    links.new(color, nodes.get('Principled BSDF').inputs['Base Color'])
    mapping = np.load(mapping_path)
    if not np.array_equal(mapping['faces'], faces):
        raise RuntimeError('Procedural eye mapping topology mismatch')
    coords = mapping['coordinates']
    attribute = mesh.attributes.new('EyeRestCoordinates', 'FLOAT_VECTOR', 'POINT')
    attribute.data.foreach_set('vector', coords.astype(np.float32).ravel())
    material = bpy.data.materials.new('ProceduralNeutralEye')
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Metallic'].default_value = 0
    bsdf.inputs['Roughness'].default_value = .36
    bsdf.inputs['Specular IOR Level'].default_value = .2
    bsdf.inputs['Emission Strength'].default_value = 0
    source = nodes.new('ShaderNodeAttribute'); source.attribute_name = attribute.name
    split = nodes.new('ShaderNodeSeparateXYZ'); links.new(source.outputs['Vector'], split.inputs[0])
    planar = nodes.new('ShaderNodeVectorMath'); planar.operation = 'MULTIPLY'
    planar.inputs[1].default_value = (1, 1, 0); links.new(source.outputs['Vector'], planar.inputs[0])
    radius = nodes.new('ShaderNodeVectorMath'); radius.operation = 'LENGTH'
    links.new(planar.outputs['Vector'], radius.inputs[0])
    front = nodes.new('ShaderNodeMath'); front.operation = 'GREATER_THAN'
    links.new(split.outputs['Z'], front.inputs[0]); front.inputs[1].default_value = 0
    def disk(size):
        less = nodes.new('ShaderNodeMath'); less.operation = 'LESS_THAN'
        links.new(radius.outputs['Value'], less.inputs[0]); less.inputs[1].default_value = size
        mask = nodes.new('ShaderNodeMath'); mask.operation = 'MULTIPLY'
        links.new(less.outputs[0], mask.inputs[0]); links.new(front.outputs[0], mask.inputs[1])
        return mask.outputs[0]
    # All radii in metres in the native shaped rest-eye frame; fixed to the mesh.
    outer = nodes.new('ShaderNodeMixRGB')
    outer.inputs[1].default_value = (.62, .64, .65, 1)
    outer.inputs[2].default_value = (.015, .025, .028, 1)
    links.new(disk(.0057), outer.inputs[0])
    iris = nodes.new('ShaderNodeMixRGB')
    links.new(outer.outputs[0], iris.inputs[1]); iris.inputs[2].default_value = (.055, .115, .13, 1)
    links.new(disk(.0051), iris.inputs[0])
    pupil = nodes.new('ShaderNodeMixRGB')
    links.new(iris.outputs[0], pupil.inputs[1]); pupil.inputs[2].default_value = (.002, .003, .004, 1)
    links.new(disk(.0022), pupil.inputs[0]); links.new(pupil.outputs[0], bsdf.inputs['Base Color'])
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        if np.all(labels[faces[polygon.index]] != body_component):
            polygon.material_index = len(mesh.materials)-1


def save_linear(scene, path):
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    bpy.data.images['Render Result'].save_render(str(path), scene=scene)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'


def read_linear(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    w, h = image.size
    pixels = np.empty(w*h*4, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(h, w, 4)


def write_display(scene, pixels, path):
    h, w = pixels.shape[:2]
    image = bpy.data.images.new('LinearComposite', width=w, height=h, alpha=True, float_buffer=True)
    image.pixels.foreach_set(np.ascontiguousarray(pixels).ravel())
    image.save_render(str(path), scene=scene)
    bpy.data.images.remove(image)


def shadow_composite(scene, output_node, body, clothing, ground, out, frame):
    """Use actual ground-render attenuation, retaining the existing geometry pass."""
    raw = out/'appearance_raw'; raw.mkdir(exist_ok=True)
    # Called immediately after the existing RGB/depth render with the ground.
    shadowed = raw/f'{frame:06d}_receiver.exr'
    save_linear(scene, shadowed)
    output_node.mute = True
    # Clean receiver uses exactly the same plane, lighting, camera and color space.
    clean = raw/'clean_receiver.exr'
    if frame == 0 or not clean.exists():
        body.hide_render = clothing.hide_render = True
        bpy.ops.render.render()
        save_linear(scene, clean)
        body.hide_render = clothing.hide_render = False
    ground.hide_render = True
    bpy.ops.render.render()
    no_ground = raw/f'{frame:06d}_no_ground.exr'
    save_linear(scene, no_ground)
    receiver = read_linear(shadowed); clean_pixels = read_linear(clean); background = read_linear(no_ground)
    # Render Layers Alpha from the earlier transparent body+clothing pass.
    foreground = read_linear(raw/f'{frame:06d}_foreground.exr')
    occupied = foreground[..., 3] > .001
    protected = occupied.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            protected |= np.roll(np.roll(occupied, dy, 0), dx, 1)
    luminance = np.array([.2126, .7152, .0722], np.float32)
    numerator = receiver[..., :3] @ luminance
    denominator = clean_pixels[..., :3] @ luminance
    ratio = np.clip(numerator / np.maximum(denominator, 1e-6), 0, 1)
    ratio[protected | (denominator < 1e-6)] = 1
    background[..., :3] *= ratio[..., None]
    background[..., 3] = 1
    write_display(scene, background, out/'rgb'/f'{frame:06d}.png')
    debug = np.ones_like(background); debug[..., :3] = 1-ratio[..., None]
    write_display(scene, debug, raw/f'{frame:06d}_shadow.png')
    np.save(raw/f'{frame:06d}_shadow_attenuation.npy', 1-ratio)
    output_node.mute = False
    scene.render.film_transparent = True
    return {'mode':'linear_receiver_ratio_composite', 'shadow_pixels':int(np.count_nonzero(ratio<.99)),
            'max_attenuation':float(1-ratio.min()), 'foreground_protected':True}
