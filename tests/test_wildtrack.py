"""Checks on the real dataset; skipped when data/Wildtrack is not downloaded."""
from pathlib import Path

import numpy as np
import pytest

from mcmot.geometry import foot_points

ROOT = Path("data/Wildtrack")
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="WILDTRACK not downloaded")


def test_calibration_projects_ground_truth_onto_feet():
    """Ground-truth ground positions, projected into each camera, land near the bottom of the annotated boxes."""
    from mcmot.wildtrack import Wildtrack

    ds = Wildtrack(ROOT)
    assert ds.num_cameras == 7 and len(ds.frames) == 400
    errors = []
    for p in ds.annotations(ds.frames[0]):
        for cam, box in p["boxes"].items():
            uv = ds.cameras[cam].ground_to_image(np.array([p["world"]]))[0]
            height = box[3] - box[1]
            errors.append(np.linalg.norm(uv - foot_points(np.array([box]))[0]) / height)
    assert np.median(errors) < 0.15  # within 15 % of the person's height
