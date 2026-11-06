"""WILDTRACK dataset (EPFL): 7 synchronised, calibrated, overlapping cameras filming pedestrians on a square.

Layout after unzipping:
    Image_subsets/C1..C7/00000000.png ... 00001995.png   annotated frames (every 5th frame = 2 fps), 1920x1080,
                                                          already undistorted
    annotations_positions/00000000.json ...               one file per frame: every person with a ground-grid
                                                          position and a box in each camera that sees them
    calibrations/intrinsic_zero/intr_<cam>.xml            camera matrix (zero distortion, matches the images)
    calibrations/extrinsic/extr_<cam>.xml                 rvec / tvec, world unit = cm

Ground plane: a 480 x 1440 grid of 2.5 cm cells starting at (-300, -900) cm, i.e. a 12 m x 36 m area.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from .geometry import Camera

CAMERAS = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]  # C1 .. C7
GRID_W, GRID_H = 480, 1440
CELL_CM = 2.5
ORIGIN_CM = (-300.0, -900.0)
AREA_CM = (ORIGIN_CM[0], ORIGIN_CM[1], ORIGIN_CM[0] + GRID_W * CELL_CM, ORIGIN_CM[1] + GRID_H * CELL_CM)
FRAME_STEP = 5  # annotated frames are every 5th frame of the 10 fps video
FPS = 2


def position_to_world(position_id: int) -> tuple[float, float]:
    """Ground-grid cell index -> (x, y) in cm."""
    return (ORIGIN_CM[0] + CELL_CM * (position_id % GRID_W), ORIGIN_CM[1] + CELL_CM * (position_id // GRID_W))


def in_area(xy: np.ndarray, margin_cm: float = 0.0) -> np.ndarray:
    xy = np.atleast_2d(xy)
    x0, y0, x1, y1 = AREA_CM
    return ((xy[:, 0] >= x0 - margin_cm) & (xy[:, 0] <= x1 + margin_cm)
            & (xy[:, 1] >= y0 - margin_cm) & (xy[:, 1] <= y1 + margin_cm))


def _find_dir(root: Path, name: str) -> Path:
    for p in [root, *root.iterdir()]:  # the zip may add one top-level folder
        if p.is_dir():
            for q in p.iterdir():
                if q.is_dir() and q.name.lower() == name.lower():
                    return q
    raise FileNotFoundError(f"no '{name}' folder under {root}; download WILDTRACK with scripts/download_wildtrack.sh")


def _load_camera(calib_dir: Path, name: str) -> Camera:
    fs = cv2.FileStorage(str(calib_dir / "intrinsic_zero" / f"intr_{name}.xml"), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    fs.release()
    root = ET.parse(calib_dir / "extrinsic" / f"extr_{name}.xml").getroot()
    rvec = np.array([float(v) for v in re.split(r"\s+", root.find("rvec").text.strip())])
    tvec = np.array([float(v) for v in re.split(r"\s+", root.find("tvec").text.strip())])
    R, _ = cv2.Rodrigues(rvec)
    return Camera(name, K, R, tvec)


class Wildtrack:
    def __init__(self, root: str | Path):
        root = Path(root)
        self.image_dir = _find_dir(root, "Image_subsets")
        self.annotation_dir = _find_dir(root, "annotations_positions")
        calib_dir = _find_dir(root, "calibrations")
        self.cameras = [_load_camera(calib_dir, name) for name in CAMERAS]
        self.frames = sorted(int(p.stem) for p in self.annotation_dir.glob("*.json"))

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    def split(self, which: str) -> list[int]:
        """'all', or the standard 90 / 10 split ('train' = first 360 frames, 'test' = last 40)."""
        n = int(len(self.frames) * 0.9)
        return {"all": self.frames, "train": self.frames[:n], "test": self.frames[n:]}[which]

    def image_path(self, cam: int, frame: int) -> Path:
        return self.image_dir / f"C{cam + 1}" / f"{frame:08d}.png"

    def read_images(self, frame: int) -> list[np.ndarray]:
        """BGR images of all cameras at one frame."""
        return [cv2.imread(str(self.image_path(c, frame))) for c in range(self.num_cameras)]

    def annotations(self, frame: int) -> list[dict]:
        """People at one frame: {'person_id', 'world': (x, y) cm, 'boxes': {cam: (x1, y1, x2, y2)}}.
        Only cameras that see the person have a box."""
        people = json.loads((self.annotation_dir / f"{frame:08d}.json").read_text())
        out = []
        for p in people:
            boxes = {v["viewNum"]: (v["xmin"], v["ymin"], v["xmax"], v["ymax"])
                     for v in p["views"] if v["xmin"] != -1}
            out.append({"person_id": p["personID"], "world": position_to_world(p["positionID"]), "boxes": boxes})
        return out
