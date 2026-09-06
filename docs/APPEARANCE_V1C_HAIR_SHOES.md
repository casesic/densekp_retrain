# Appearance V1-C — Hair + Shoes audit

Status: `BLOCKED`. No asset passed the compatibility gate, so no
`outputs/p0_dressed_full_appearance/` render was produced.

## Hair

The official source archive is `b2_groom_abc.tar.gz`, available after signing
in and accepting the BEDLAM2 license. The public repository and public website
do not publish its byte size, so the archive size is recorded as unknown and no
downloader was generated under the 40 GB rule.

The archive contains Alembic strand curves. BEDLAM2 imports them into Unreal as
Groom assets with a fixed import rotation of `[-90, 0, 180]`. The groom source
is reusable across identities, but it is not directly body-independent: the
official pipeline creates a GeometryCache groom binding for every target body
shape. That binding uses a neutral locked-head authoring reference and a
shape-only target GeometryCache in the default T-pose. At render time the groom
component and subject-specific binding are attached to the animated body
GeometryCache. It is neither a world-static hair mesh nor a per-motion hair
simulation.

The current body is BEDLAM v1 SMPL-X male. BEDLAM2 uses the neutral locked-head,
no-head-bun model and its own T-pose target caches. The occurrence of the same
`rp_aaron_posed_009` name in BEDLAM2 code is insufficient evidence that the
current mesh is the exact binding target. No groom ABC, target binding, source
reference cache, or compatible Blender binding path exists locally. Therefore
hair compatibility for `rp_aaron_posed_009 / 1095` is not verified.

## Shoes

The official source archive is `b2_shoes.tar.gz`. It is gated by the BEDLAM2
license. Its authenticated URL is present in the official processing repository,
but its size is not published publicly, so no downloader was generated.

BEDLAM2 shoes are not static, rigged, or skinned shoe meshes. Each shoe is a set
of color, normal, and object-space displacement maps applied to the body. The
official pipeline replaces the normal foot template with a 10,475-vertex
toeless/stocking-foot neutral locked-head template, drives that modified body
with SMPL-X motion, and lifts the complete body by a shoe-specific vertical
offset. Foot and ankle articulation therefore comes from the modified SMPL-X
body itself.

This cannot be applied to the current standard-foot BEDLAM v1 body without
changing body vertices and translation. Doing so would violate the explicit
geometry, motion, and projection constraints. Shoe compatibility is therefore
not verified and integration is blocked.

## Preserved regression state

The existing `outputs/p0_dressed_skin_textured/` baseline remains unchanged.
Its recorded checks remain:

- body geometry unchanged: `YES`
- clothing geometry unchanged: `YES`
- projection regression: `PASS`
- maximum projection error: `3.3463941410797083e-05 px`

Depth, person-mask, and visibility checks with new occluders are not applicable
because no new geometry was integrated. No before/after images were generated;
duplicating the baseline would not provide evidence of a hair/shoe integration.

## Recommended path

For official BEDLAM2 hair and shoes, migrate an entire selected pair to the
BEDLAM2 neutral locked-head/toeless body and use the official Unreal groom
binding and shoe material pipeline. For the current BEDLAM v1 Blender pipeline,
a separate generic rigged hair-and-shoe asset pipeline is more controllable, but
it would not be an official BEDLAM appearance baseline.

Appearance V1-D can proceed independently with Poly Haven's
`abandoned_church` 4K HDR. It is the first entry in both local BEDLAM HDRI
whitelists, is available as a 24.42 MB HDR, and is CC0.
