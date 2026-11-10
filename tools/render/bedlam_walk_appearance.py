"""Appearance-only native-eye shader; no body or eye mesh edits."""
import bpy
import numpy as np


def add_native_eyes(mesh, faces, appearance):
    with np.load(appearance['eye_mapping']) as z:
        np.testing.assert_array_equal(z['faces'], faces)
        coords=z['coordinates'];labels=z['component_labels']
    attr=mesh.attributes.new('EyeRestCoordinates','FLOAT_VECTOR','POINT')
    attr.data.foreach_set('vector',coords.ravel())
    material=bpy.data.materials.new('NativeEyes_BrownIris');material.use_nodes=True
    nodes=material.node_tree.nodes;links=material.node_tree.links
    bs=nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.36
    bs.inputs['Specular IOR Level'].default_value=.2
    source=nodes.new('ShaderNodeAttribute');source.attribute_name=attr.name
    split=nodes.new('ShaderNodeSeparateXYZ');links.new(source.outputs['Vector'],split.inputs[0])
    planar=nodes.new('ShaderNodeVectorMath');planar.operation='MULTIPLY';planar.inputs[1].default_value=(1,1,0)
    links.new(source.outputs['Vector'],planar.inputs[0])
    radius=nodes.new('ShaderNodeVectorMath');radius.operation='LENGTH';links.new(planar.outputs['Vector'],radius.inputs[0])
    front=nodes.new('ShaderNodeMath');front.operation='GREATER_THAN';front.inputs[1].default_value=0
    links.new(split.outputs['Z'],front.inputs[0]);color=None
    for size,fill in [(.0057,(.015,.025,.028,1)),(appearance['iris_radius_m'],appearance['iris_color']),
                      (appearance['pupil_radius_m'],(.002,.003,.004,1))]:
        less=nodes.new('ShaderNodeMath');less.operation='LESS_THAN';less.inputs[1].default_value=size
        links.new(radius.outputs['Value'],less.inputs[0])
        mask=nodes.new('ShaderNodeMath');mask.operation='MULTIPLY'
        links.new(less.outputs[0],mask.inputs[0]);links.new(front.outputs[0],mask.inputs[1])
        mix=nodes.new('ShaderNodeMixRGB');links.new(mask.outputs[0],mix.inputs[0])
        if color is None:mix.inputs[1].default_value=appearance['sclera_color']
        else:links.new(color,mix.inputs[1])
        mix.inputs[2].default_value=fill;color=mix.outputs[0]
    links.new(color,bs.inputs['Base Color']);mesh.materials.append(material)
    count=0
    for polygon in mesh.polygons:
        if np.all(labels[faces[polygon.index]]>0):polygon.material_index=len(mesh.materials)-1;count+=1
    return count
