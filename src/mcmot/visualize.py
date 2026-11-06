"""Result video: the 7 camera views with global ids, plus a bird's-eye view (BEV) of the ground plane."""
import colorsys

import cv2
import numpy as np

from .wildtrack import AREA_CM

TILE_W, TILE_H = 640, 360


def id_color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.75, 0.95)  # golden-ratio hues: neighbours differ
    return int(b * 255), int(g * 255), int(r * 255)


def draw_camera(img: np.ndarray, rows: list[dict], cam: int) -> np.ndarray:
    img = img.copy()
    for r in rows:
        x1, y1, x2, y2 = map(int, r["box"])
        c = id_color(r["global_id"])
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
        cv2.putText(img, str(r["global_id"]), (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 4)
    img = cv2.resize(img, (TILE_W, TILE_H))
    cv2.putText(img, f"C{cam + 1}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return img


def draw_bev(positions: dict[int, np.ndarray], gt: list[dict], history: dict[int, list]) -> np.ndarray:
    """Ground plane, long side horizontal. Grey rings = ground truth, coloured dots + trails = tracker."""
    x0, y0, x1, y1 = AREA_CM
    canvas = np.full((TILE_H, TILE_W, 3), 30, np.uint8)
    to_px = lambda xy: (int((xy[1] - y0) / (y1 - y0) * (TILE_W - 1)), int((xy[0] - x0) / (x1 - x0) * (TILE_H - 1)))
    for p in gt:
        cv2.circle(canvas, to_px(p["world"]), 7, (140, 140, 140), 1)
    for g in positions:  # trails of the people present now (last 10 positions = 5 s)
        pts = history.get(g, [])
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array([to_px(p) for p in pts[-10:]], np.int32)], False, id_color(g), 1)
    for g, xy in positions.items():
        cv2.circle(canvas, to_px(xy), 4, id_color(g), -1)
    cv2.putText(canvas, "ground plane (12 m x 36 m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def mosaic(tiles: list[np.ndarray]) -> np.ndarray:
    """3 x 3 grid: 7 cameras, BEV, one empty tile."""
    tiles = tiles + [np.zeros_like(tiles[0])] * (9 - len(tiles))
    return np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(3)])
