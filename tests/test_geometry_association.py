import numpy as np

from mcmot.association import GroundTracker, Observation, fuse
from mcmot.evaluate import evaluate, iou_distance
from mcmot.geometry import Camera, foot_points


def make_camera(yaw_deg: float, pos) -> Camera:
    """Camera at `pos` (cm, z up) looking down at the ground at 30 degrees, rotated by `yaw`."""
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
    yaw, tilt = np.radians(yaw_deg), np.radians(120)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    R = Rx @ Rz
    return Camera("synthetic", K, R, -R @ np.asarray(pos, float))


def test_ground_projection_round_trip():
    cam = make_camera(20, [0, -1500, 600])
    xy = np.array([[0, 0], [250, 400], [-300, 800]], float)
    np.testing.assert_allclose(cam.image_to_ground(cam.ground_to_image(xy)), xy, atol=1e-6)


def test_foot_point_is_bottom_centre():
    np.testing.assert_allclose(foot_points(np.array([[10, 20, 30, 120]])), [[20, 120]])


def test_fuse_groups_cameras_by_ground_position():
    obs = [Observation(0, 1, np.array([0.0, 0.0])), Observation(1, 5, np.array([40.0, 0.0])),
           Observation(2, 9, np.array([0.0, 30.0])), Observation(0, 2, np.array([1000.0, 0.0])),
           Observation(1, 6, np.array([1030.0, 0.0]))]
    groups = sorted(sorted(g) for g in fuse(obs, radius=100))
    assert groups == [[0, 1, 2], [3, 4]]


def test_fuse_keeps_one_box_per_camera():
    obs = [Observation(0, 1, np.array([0.0, 0.0])), Observation(0, 2, np.array([20.0, 0.0])),
           Observation(1, 1, np.array([10.0, 0.0]))]
    assert all(len({obs[k].cam for k in g}) == len(g) for g in fuse(obs, radius=100))


def test_ground_tracker_keeps_identity_while_walking():
    tr = GroundTracker(fuse_radius=100, max_step=100, min_length=1, min_cameras=2)
    for t in range(6):  # two people walking in parallel, 10 m apart, seen by cameras 0 and 1
        obs = [Observation(c, 10 * c + p, np.array([60.0 * t + 5 * c, 1000.0 * p])) for p in (0, 1) for c in (0, 1)]
        tr.update(t, 5 * t, obs)
    assign, ground = tr.results()
    assert len({g for d in ground.values() for g in d}) == 2
    assert assign[(0, 0, 0)] == assign[(25, 1, 10)] != assign[(25, 0, 1)]
    np.testing.assert_allclose(ground[25][assign[(25, 0, 0)]], [302.5, 0])


def test_iou_distance_threshold():
    d = iou_distance([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15]], 0.5)
    assert d[0, 0] == 0 and np.isnan(d[0, 1])


def test_perfect_prediction_scores_one():
    gt = {f: [{"person_id": 3, "world": (10.0 * f, 0.0), "boxes": {0: (0, 0, 10, 20), 1: (5, 5, 15, 25)}}]
          for f in range(4)}
    rows = [{"frame": f, "cam": c, "local_id": 1, "global_id": 0, "box": b}
            for f, people in gt.items() for c, b in people[0]["boxes"].items()]
    ground = {f: {0: np.array(people[0]["world"])} for f, people in gt.items()}
    m = evaluate(gt, rows, ground, num_cameras=2)
    for name in ("C1", "C2", "cross_camera", "ground"):
        assert m[name]["mota"] == 1.0 and m[name]["idf1"] == 1.0
