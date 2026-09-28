# Urban Street environment

## Asset-backed revision (2026-09-20)

The procedural version below is retained for comparison; its original visual
acceptance was too generous. A separate `urban_street_assets.blend` now uses
the CC0 Poly Haven apartment facade kit (178 MB download, 14 2K texture maps)
and selected lamps, potted plants and air-conditioning units from the existing
Hidden Alley source. Native facade UVs and evaluated detailed geometry are
retained; modules share meshes to avoid unnecessary duplication.

Build/reproduction instructions and licensing are in the
[asset README](../../assets/scenes/photoreal/urban_street/README.md#asset-backed-revision).
New previews/reports: `outputs/scenes/urban_street_assets/`.
New integration spec: `assets/scenes/photoreal/urban_street/scene_spec_assets.json`.
The original blend, previews and default spec are unchanged.

Actual Blender 5.1.1 CPU Cycles renders: 48 samples, three 1280×720 perspective
views and one 1280×640 ERP. All four were opened for visual inspection after the
second pass (which reduced repeated doorways and varied facade finishes).
Saved blend: 219,499,893 bytes; 2,572 objects; 3,632,150 evaluated mesh triangles;
38 packed file textures. Render times: street 29.34 s, sidewalk 29.31 s,
long street 29.37 s, ERP 21.66 s. The preview stage took 110.94 s; this excludes
downloads and the approximately six-second asset assembly/save stage. Peak
process RSS was 6,735 MiB, with no GPU used. Detailed source hashes are in
`asset_manifest.json`, not the baseline renderer's empty `texture_sources` field.
The reopened scene passed 1,160 support rays with zero missing textures and zero
sampled corridor intrusions. Seven existing unit tests and the workspace audit
also passed. These checks do not certify realism or continuous collision safety.

The revision addresses facade detail, not every cause of artificial appearance.
It has residential/traditional rather than contemporary retail styling;
repeated modules, simplified trees, uniform pavement and distant block geometry
remain limitations. Geometry validation must not be interpreted as a photorealism
certificate. No people, vehicles, training or annotation changes are included.

## Procedural baseline (historical results)

The second designated outdoor training environment alongside Cycling Forest.
The current deliverable is an environment-only modern street: road, sidewalks,
mixed-use facades, sparse street furniture, and six trees. Hidden Alley and
Riverfront remain separate source/reference scenes.

See the [asset README](../../assets/scenes/photoreal/urban_street/README.md) for
provenance, scale, marker coordinates, build/validation commands and limitations.

Scene: `assets/scenes/photoreal/urban_street/urban_street.blend`.
Previews and machine-readable evidence: `outputs/scenes/urban_street/`.

The existing runtime Blender executable helper and subprocess logger are reused.
The forest validator is a top-level script tied to its sinusoidal road and
scatter graph, so it is not imported or copied: the urban validator implements
explicit-mesh road/sidewalk footprint checks. No scene framework is added.

Perspective and native Blender ERP previews are supported. The future body
pipeline is not run. All environment objects keep `pass_index=0`.
For future integration, use the existing SceneAssetSpec contract with
`ground_mode=scene_native` and `collection=null`; placement on sidewalks needs
the documented +0.15 m support height. Preview lights/cameras should be selected
or hidden intentionally by a future caller. No camera or annotation mathematics
are changed for this scene.

## Delivered validation (2026-09-20)

Blender 5.1.1, CPU Cycles, seed 73, 32 samples. The saved scene was reopened by
the independent validator with `--require-previews`: PASS, 1,160 support rays,
zero intrusions and zero missing textures. Ten spawn and five camera reference
empties passed checks. Road is 7×160 m; each sidewalk is 3.5×160 m. The sampled
clear walking strip is 1.9 m wide over 140 m on each side. Twenty-six main
frontage buildings are 9.7–19.0 m high, plus the distant cross-street backdrop.

| Measurement | Result |
|---|---:|
| Saved blend, with five packed maps | 12,927,745 bytes (12.33 MiB) |
| Scene objects | 3,461 |
| Evaluated mesh triangles (text curves excluded) | 143,684 |
| Street-level preview, 1280×720 | 15.58 s |
| Sidewalk preview, 1280×720 | 16.51 s |
| Long-street preview, 1280×720 | 16.25 s |
| ERP preview, 1280×640 | 11.84 s |
| Full build plus four previews | 88.26 s |
| Peak Blender process RSS | 1,739 MiB |

All four final images were opened for visual inspection. Scale, eye height,
ground contact, pavement clearance, texture presence and horizon closure were
checked. The first render had overly bright glass and repetitive window bays;
the delivered version reduces exposure, changes glazing, varies window widths
and heights, and adds selected Juliet railings and shop transoms. No obvious
floating/embedded props, missing-texture magenta, open ground boundary, severe
overexposure or dark unusable regions were found in these views. At the ERP
nadir the centreline marking is stretched by the projection; it is not a hole.

Remaining visual limitations are modular frontage repetition, simplified tree
foliage and reflective panels without interiors. This is an environment baseline
for review, not evidence of photorealism at every close-up viewpoint.

Workspace audit and all seven existing unit/numeric tests passed. GPU rendering,
continuous collision checking, arbitrary-camera coverage and integration with
future human/fisheye annotation production were not tested. A failed desktop
thumbnail cache write in the sandbox did not affect blend saving or rendering.
