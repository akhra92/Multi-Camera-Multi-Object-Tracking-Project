"""Camera geometry: projecting image points onto the ground plane (z = 0) and back.

A pinhole camera maps a world point X to pixels by  x ~ K (R X + t).  For points on the ground (Z = 0) this reduces
to a 3x3 homography  H = K [r1 r2 t]  (r1, r2 = first two columns of R), which is invertible: a pixel that lies on
the ground (a person's feet) maps to exactly one ground position. That is what lets several cameras agree on where a
person stands.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    name: str
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 rotation, world -> camera
    t: np.ndarray  # (3,) translation, world -> camera (same unit as the world, here cm)

    @property
    def ground_to_image_h(self) -> np.ndarray:
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])

    @property
    def image_to_ground_h(self) -> np.ndarray:
        return np.linalg.inv(self.ground_to_image_h)

    def image_to_ground(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) ground coordinates (world units)."""
        return _apply_h(self.image_to_ground_h, uv)

    def ground_to_image(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) ground coordinates -> (N, 2) pixels."""
        return _apply_h(self.ground_to_image_h, xy)


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    if pts.size == 0:
        return np.zeros((0, 2))
    p = np.column_stack([pts, np.ones(len(pts))]) @ h.T
    return p[:, :2] / p[:, 2:3]


def foot_points(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of (N, 4) x1, y1, x2, y2 boxes: where a standing person touches the ground."""
    boxes = np.atleast_2d(boxes)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
