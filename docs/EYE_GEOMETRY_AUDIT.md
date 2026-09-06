# Eye geometry audit

The reported eye-position artifact is not a geometry, object-transform, or
parenting error. The two eyes are native connected components of the same
10,475-vertex SMPL-X mesh, with 546 vertices and 1,088 faces each. They are not
separate Blender objects and receive the same object matrix exactly once.

Joint weights and spatial checks identify the positive-X component as the left
eye (joint 23) and negative-X component as the right eye (joint 24), each with
mean weight 0.99999988. Their rest-space centers are approximately
`(+0.032674, 0.339730, 0.068347)` and
`(-0.032675, 0.339730, 0.068346)` metres. Center separation is 65.349 mm and
mirror symmetry error is 0.00135 mm. Replaying the source SMPL-X parameters
reproduces all vertices exactly. Across the 16 frames, eye separation changes by
only 0.114 mm and both centers remain within 2.014 mm of the fitted head motion.

The visual artifact came from both eye components inheriting the reddish legacy
skin material without sclera, iris, or pupil cues. HDRI specular response made
the uniform reddish spheres appear displaced. Geometry was not edited.

After the geometry gate passed, the eye texture stage tested the archive's
documented `SMPLX_eye.png` RGBA body-UV overlay. Its effective alpha region does
not cover the verified eye-component UV islands; its rendered preview was
pixel-identical to the neutral-eye preview. The final deterministic fallback
assigns both eye components an off-white nonmetallic material with roughness 0.3
and IOR level 0.25. This removes the red artifact but does not claim a finished
iris/pupil texture.

The 16-frame material-only regression passed: body and clothing geometry,
camera, projection, depth, mask, and visibility are unchanged. Maximum
analytic-versus-Blender projection error remains
`3.3463941410797083e-05 px`.
