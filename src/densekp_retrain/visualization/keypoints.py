"""Candidate point panels and contact sheets with historical raster behavior."""

import cv2
import numpy as np
from PIL import Image, ImageDraw


def render_candidate_panel(item, prediction, supervised, visible, label):
    """Draw full-square normalized XY on a 512x512 panel.

    item img_disp is BGR. Visibility colors GT; supervised selects points.
    The center-column guides do not shift the full-square coordinates.
    """
    panel = Image.fromarray(cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB)).resize((512, 512))
    draw = ImageDraw.Draw(panel)
    target_pixels = (item["proj_verts_cropped"][:, :2] + 0.5) * 512
    prediction_pixels = (prediction[:, :2] + 0.5) * 512
    for point_index in np.flatnonzero(supervised):
        target_point = target_pixels[point_index]
        predicted_point = prediction_pixels[point_index]
        color = (40, 255, 70) if visible[point_index] else (255, 175, 40)
        if np.isfinite(predicted_point).all() and np.abs(predicted_point).max() < 100000:
            draw.line([tuple(target_point), tuple(predicted_point)], fill=(230, 190, 70), width=1)
            draw.ellipse(
                (
                    predicted_point[0] - 2,
                    predicted_point[1] - 2,
                    predicted_point[0] + 2,
                    predicted_point[1] + 2,
                ),
                fill=(255, 60, 70),
            )
        draw.ellipse(
            (target_point[0] - 3, target_point[1] - 3, target_point[0] + 3, target_point[1] + 3),
            fill=color,
        )
    for x in (64, 448):
        draw.line((x, 25, x, 512), fill=(180, 180, 180))
    draw.rectangle((0, 0, 512, 24), fill="black")
    draw.text((5, 5), label, fill="white")
    return panel


def save_contact_sheets(output, fixed_ids, worst_ids):
    "Reload saved JPEG panels in fixed order and write the two historical sheets."
    for name, case_ids in [("fixed_controls", fixed_ids), ("worst_samples", worst_ids)]:
        sheet = Image.new("RGB", (1024, 512 * ((len(case_ids) + 1) // 2)), (25, 25, 25))
        for index, case_id in enumerate(case_ids):
            sheet.paste(
                Image.open(output / case_id / "prediction.jpg"), (index % 2 * 512, index // 2 * 512)
            )
        sheet.save(output / f"{name}.jpg")
