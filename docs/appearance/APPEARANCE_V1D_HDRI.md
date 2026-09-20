# Appearance V1-D — HDRI environment

Status: `PASS`.

The deterministic environment baseline uses Poly Haven's CC0
`abandoned_church_4k.hdr`. The downloaded Radiance HDR is 4096×2048,
27,183,288 bytes, and Blender interprets it as Linear Rec.709. The renderer now
selects `simple` or `hdri` through CLI flags; the default remains `simple`. A
single-frame pixel regression confirmed that explicit simple mode reproduces the
previous RGB exactly.

The selection matrix compared HDRI-only strengths 0.5, 1.0, and 1.5, plus the
original studio lights and a weak fill at strength 1.0. HDRI-only had acceptable
exposure but insufficient contact shadow in Eevee. The studio setup overexposed
the skin. The final setting is strength 1.0, rotation 0°, visible background,
and one 90 W area fill. The neutral rough ground remains enabled.

All 16 original frames were rendered. Body and clothing geometry, camera,
DenseKP 2D/camera coordinates, depth, mask, and visibility match V1-B exactly.
The analytic-versus-Blender maximum projection error remains
`3.3463941410797083e-05 px`.

Color management remained unchanged: sRGB display, AgX view transform, no look,
exposure 0, gamma 1. The existing V1-B timing was 402.362 ms/frame and V1-D was
441.278 ms/frame, including both passes and first-frame warmup.

Visually, the real environment removes the debug background, gives the face
warmer directional modeling, and preserves readable clothing folds without
overexposure. The weak fill adds a subtle foot-contact shadow. The remaining
limitation is a visible horizontal boundary between the neutral ground and HDRI;
perspective and material continuity are imperfect, so the result still has some
composited appearance.

The next appearance phase should use a small deterministic HDRI set and improve
ground/background integration before camera stress tests.
