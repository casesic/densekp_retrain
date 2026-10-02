"""Historical periodic ERP sampling and measured reference-lens extraction."""

from pathlib import Path
from dataclasses import dataclass
import json
import cv2
import numpy as np
from PIL import Image


def remap_periodic_erp(
    image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interpolation: int
) -> np.ndarray:
    """Sample longitude periodically without wrapping the ERP poles."""
    padded = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    padded_x = np.where(map_x >= 0, map_x + 1, -1).astype(np.float32)
    return cv2.remap(padded, padded_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)


def lens_mask_from_reference(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    """Keep the real lens footprint, without copying any scene RGB into training."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"reference fisheye not readable: {path}")
    if frame.shape[:2] != expected_size:
        raise ValueError(
            f"reference fisheye size {frame.shape[:2]} != {expected_size}: {path}"
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rough = (gray > 5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"cannot find lens footprint: {path}")
    mask = np.zeros_like(rough)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return mask != 0
