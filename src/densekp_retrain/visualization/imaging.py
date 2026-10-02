"""Post-processing previews; drawing never determines labels or valid masks."""

from PIL import Image, ImageDraw


def draw_overlay(rgb, xy, visible, in_view, ids):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    colors = [(255, 80, 70), (60, 255, 100), (80, 150, 255), (255, 220, 60)]
    for j, person_id in enumerate(ids):
        for point, seen, present in zip(xy[j], visible[j], in_view[j]):
            if not present:
                continue
            x, y = point
            radius = 2
            color = colors[j % len(colors)] if seen else (170, 170, 170)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.text(
            (8, 8 + 16 * j),
            f"person {person_id}: visible {int(visible[j].sum())} / in-view {int(in_view[j].sum())}",
            fill=colors[j % len(colors)],
        )
    return image


def draw_omni_overlay(display, projected, inside, visible):
    overlay = Image.fromarray(display)
    draw = ImageDraw.Draw(overlay)
    for j, (uv, ok) in enumerate(zip(projected, inside)):
        if ok:
            u, v = uv
            color = (20, 245, 70) if visible[j] else (255, 60, 40)
            draw.ellipse((u - 8, v - 8, u + 8, v + 8), fill=color)
            if j % 20 == 0:
                draw.text((u + 9, v + 4), str(j), fill=color)
    return overlay
