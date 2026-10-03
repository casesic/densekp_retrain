"""Blender BVH broad phase plus triangle-edge narrow phase, independent of kNN.

Reports proper non-coplanar crossings. Blender BVH can omit coplanar overlap;
the third return value is therefore None (unknown), never a zero pass count.
Self tests exclude vertex-adjacent faces.
"""
import numpy as np
from mathutils.bvhtree import BVHTree


def crossings(vertices_a, faces_a, vertices_b=None, faces_b=None):
    self_test = vertices_b is None
    if self_test:
        vertices_b, faces_b = vertices_a, faces_a
    tree_a = BVHTree.FromPolygons(vertices_a.tolist(), faces_a.tolist(), all_triangles=True)
    tree_b = tree_a if self_test else BVHTree.FromPolygons(
        vertices_b.tolist(), faces_b.tolist(), all_triangles=True)
    pairs = np.asarray(tree_a.overlap(tree_b), dtype=np.int32).reshape(-1, 2)
    if self_test and len(pairs):
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        adjacent = (faces_a[pairs[:, 0], :, None] == faces_b[pairs[:, 1], None, :]).any((1, 2))
        pairs = pairs[~adjacent]
    if not len(pairs):
        return pairs, np.empty((0, 3)), None
    a = vertices_a[faces_a[pairs[:, 0]]].astype(np.float64)
    b = vertices_b[faces_b[pairs[:, 1]]].astype(np.float64)
    hit = np.zeros(len(pairs), bool)
    locations = np.zeros((len(pairs), 3))
    for source, target in ((a, b), (b, a)):
        edge1, edge2 = target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]
        for corner in range(3):
            origin = source[:, corner]
            direction = source[:, (corner + 1) % 3] - origin
            h = np.cross(direction, edge2)
            determinant = np.einsum('ij,ij->i', edge1, h)
            valid = np.abs(determinant) > 1e-14
            inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
            delta = origin - target[:, 0]
            u = inverse * np.einsum('ij,ij->i', delta, h)
            q = np.cross(delta, edge1)
            v = inverse * np.einsum('ij,ij->i', direction, q)
            t = inverse * np.einsum('ij,ij->i', edge2, q)
            valid &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
            valid &= (t > 1e-7) & (t < 1 - 1e-7)
            locations[valid] = origin[valid] + t[valid, None] * direction[valid]
            hit |= valid
    return pairs[hit], locations[hit], None
