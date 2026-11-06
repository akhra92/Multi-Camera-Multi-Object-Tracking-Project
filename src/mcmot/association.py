"""Cross-camera association: turn per-camera DeepSORT tracks into global identities on the ground plane.

Two steps, run frame by frame:

1. Fuse (space). Every confirmed DeepSORT track in every camera gives one ground point (its foot point projected
   with the calibration). Points from different cameras closer than `fuse_radius` are merged into one person, with
   complete linkage (every pair in a group must be close) and at most one point per camera (a person is one box per
   camera). The person's position is the mean of the group.

2. Track (time). Fused people are matched to the global tracks of the previous frames with the Hungarian algorithm.
   The cost combines
     - ground distance to the track's predicted position (constant velocity), gated at `max_step`, and
     - ID votes: the share of the group's (camera, DeepSORT id) pairs that already belonged to that global track.
   Votes carry DeepSORT's appearance-based continuity across frames; distance handles new or re-appearing tracks.
   Unmatched people start new global tracks; tracks unseen for `max_age` frames end. Global tracks shorter than
   `min_length` frames are dropped as noise.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Observation:
    """One confirmed DeepSORT track in one camera at one frame."""
    cam: int
    local_id: int
    ground: np.ndarray  # (x, y) cm


def fuse(obs: list[Observation], radius: float) -> list[list[int]]:
    """Group observations of one frame into people (lists of indices)."""
    n = len(obs)
    pts = np.array([o.ground for o in obs]).reshape(n, 2)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    group = list(range(n))
    members = {i: [i] for i in range(n)}
    pairs = sorted((d[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                   if obs[i].cam != obs[j].cam and d[i, j] <= radius)
    for _, i, j in pairs:
        a, b = group[i], group[j]
        if a == b:
            continue
        A, B = members[a], members[b]
        if {obs[k].cam for k in A} & {obs[k].cam for k in B}:
            continue  # would put two boxes of one camera on one person
        if d[np.ix_(A, B)].max() > radius:
            continue  # complete linkage: the merged group must stay compact
        for k in B:
            group[k] = a
        members[a] = A + members.pop(b)
    return list(members.values())


@dataclass
class GlobalTrack:
    gid: int
    positions: dict[int, np.ndarray] = field(default_factory=dict)  # frame index -> (x, y)
    last: int = 0

    def predict(self, t: int) -> np.ndarray:
        ts = sorted(self.positions)
        p = self.positions[ts[-1]]
        if len(ts) > 1:
            v = (p - self.positions[ts[-2]]) / (ts[-1] - ts[-2])
            p = p + v * min(t - ts[-1], 2)  # extrapolate at most 2 steps
        return p


class GroundTracker:
    def __init__(self, fuse_radius=75.0, max_step=150.0, vote_weight=1.0, max_age=5, min_length=3, min_cameras=1):
        self.fuse_radius, self.max_step, self.vote_weight = fuse_radius, max_step, vote_weight
        self.max_age, self.min_length, self.min_cameras = max_age, min_length, min_cameras
        self.tracks: list[GlobalTrack] = []
        self.local_to_global: dict[tuple[int, int], int] = {}
        self.assign: dict[tuple[int, int, int], int] = {}  # (frame, cam, local id) -> global id
        self.frame_of: dict[int, int] = {}  # frame index -> dataset frame number
        self._next = 0

    def update(self, t: int, frame: int, obs: list[Observation]) -> None:
        """t = frame index (0, 1, 2, ...), frame = dataset frame number."""
        groups = fuse(obs, self.fuse_radius) if obs else []
        groups = [g for g in groups if len(g) >= self.min_cameras]  # seen by too few cameras = likely noise
        centers = [np.mean([obs[k].ground for k in g], axis=0) for g in groups]
        active = [tr for tr in self.tracks if t - tr.last <= self.max_age]
        cost = np.full((len(groups), len(active)), 1e6)
        for i, g in enumerate(groups):
            keys = [(obs[k].cam, obs[k].local_id) for k in g]
            for j, tr in enumerate(active):
                dist = np.linalg.norm(centers[i] - tr.predict(t))
                votes = sum(self.local_to_global.get(k) == tr.gid for k in keys) / len(keys)
                if dist <= self.max_step or votes > 0:
                    cost[i, j] = dist / self.max_step - self.vote_weight * votes
        matched = {}
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    matched[i] = active[j]
        for i, g in enumerate(groups):
            tr = matched.get(i)
            if tr is None:
                tr = GlobalTrack(self._next)
                self._next += 1
                self.tracks.append(tr)
            tr.positions[t] = centers[i]
            tr.last = t
            for k in g:
                key = (obs[k].cam, obs[k].local_id)
                self.local_to_global[key] = tr.gid
                self.assign[(frame, *key)] = tr.gid
        self.frame_of[t] = frame

    def results(self) -> tuple[dict[tuple[int, int, int], int], dict[int, dict[int, np.ndarray]]]:
        """(frame, cam, local id) -> global id, and frame -> {global id: (x, y)}, without too-short tracks."""
        keep = {tr.gid for tr in self.tracks if len(tr.positions) >= self.min_length}
        ground: dict[int, dict[int, np.ndarray]] = {}
        for tr in self.tracks:
            if tr.gid in keep:
                for t, xy in tr.positions.items():
                    ground.setdefault(self.frame_of[t], {})[tr.gid] = xy
        return {k: g for k, g in self.assign.items() if g in keep}, ground
