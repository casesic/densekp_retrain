"""Frozen pre-refactor studio material branch for the A2 comparison only."""
import bpy
import numpy as np

from densekp_retrain.rendering.blender.materials import add_clothing_materials


def add_before_materials(mesh, cloth, native_material):
    if native_material:
        uv=np.load(native_material['vertices_uv']);uf=np.load(native_material['faces_uv'])
        cf=np.asarray([tuple(mesh.loops[i].vertex_index for i in p.loop_indices) for p in mesh.polygons])
        if uf.shape!=cf.shape or uf.min()<0 or uf.max()>=len(uv):raise ValueError('native_clothing_uv_mismatch')
        layer=mesh.uv_layers.new(name='OriginalBEDLAM_UV')
        layer.data.foreach_set('uv',uv[uf].reshape(-1))
        material=bpy.data.materials.new('TransferredBEDLAMGarment');material.use_nodes=True
        nodes=material.node_tree.nodes;links=material.node_tree.links
        bs=nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.58
        bs.inputs['Base Color'].default_value=(.23,.35,.48,1)
        if native_material.get('diffuse'):
            image=nodes.new('ShaderNodeTexImage');image.image=bpy.data.images.load(native_material['diffuse'])
            image.image.colorspace_settings.name='sRGB';links.new(image.outputs['Color'],bs.inputs['Base Color'])
        if native_material.get('normal'):
            image=nodes.new('ShaderNodeTexImage');image.image=bpy.data.images.load(native_material['normal'])
            image.image.colorspace_settings.name='Non-Color'
            normal=nodes.new('ShaderNodeNormalMap');links.new(image.outputs['Color'],normal.inputs['Color'])
            links.new(normal.outputs['Normal'],bs.inputs['Normal'])
        mesh.materials.append(material)
    else:add_clothing_materials(mesh,cloth)
