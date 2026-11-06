"""MOT metrics with py-motmetrics (MOTA, IDF1, ...).

Three views of the result:
  per_camera   boxes in each camera, local DeepSORT ids (IoU >= 0.5)       -> single-camera tracking quality
  cross_camera boxes in all cameras together, global ids (IoU >= 0.5)      -> MTMC IDF1: one person must keep the
                                                                              same id in every camera
  ground       ground-plane positions, global ids (distance <= 1 m)        -> the usual WILDTRACK tracking metric

Distance matrices are computed here (motmetrics' own helpers use np.asfarray, removed in NumPy 2).
"""
import motmetrics as mm
import numpy as np

METRICS = ["mota", "idf1", "motp", "precision", "recall", "num_switches", "mostly_tracked", "mostly_lost",
           "num_false_positives", "num_misses", "num_unique_objects"]


def iou_distance(gt: np.ndarray, pred: np.ndarray, iou_threshold: float) -> np.ndarray:
    """1 - IoU between (N, 4) and (M, 4) x1 y1 x2 y2 boxes; NaN (= cannot match) below the IoU threshold."""
    if len(gt) == 0 or len(pred) == 0:
        return np.empty((len(gt), len(pred)))
    gt, pred = np.asarray(gt, float)[:, None], np.asarray(pred, float)[None]
    iw = np.clip(np.minimum(gt[..., 2], pred[..., 2]) - np.maximum(gt[..., 0], pred[..., 0]), 0, None)
    ih = np.clip(np.minimum(gt[..., 3], pred[..., 3]) - np.maximum(gt[..., 1], pred[..., 1]), 0, None)
    inter = iw * ih
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    iou = inter / (area(gt) + area(pred) - inter)
    d = 1.0 - iou
    d[iou < iou_threshold] = np.nan
    return d


def euclidean_distance(gt: np.ndarray, pred: np.ndarray, max_dist: float) -> np.ndarray:
    if len(gt) == 0 or len(pred) == 0:
        return np.empty((len(gt), len(pred)))
    d = np.linalg.norm(np.asarray(gt, float)[:, None] - np.asarray(pred, float)[None], axis=2)
    d[d > max_dist] = np.nan
    return d


def summarize(accs: dict[str, mm.MOTAccumulator]) -> dict[str, dict[str, float]]:
    mh = mm.metrics.create()
    df = mh.compute_many(list(accs.values()), names=list(accs), metrics=METRICS)
    return {name: {k: float(v) for k, v in row.items()} for name, row in df.iterrows()}


def evaluate(gt: dict, boxes: list[dict], ground: dict, num_cameras: int, iou_threshold: float = 0.5,
             ground_dist_cm: float = 100.0) -> dict:
    """
    gt:     frame -> list of {'person_id', 'world', 'boxes': {cam: box}}   (Wildtrack.annotations)
    boxes:  rows {'frame', 'cam', 'local_id', 'global_id', 'box'}
    ground: frame -> {global id: (x, y)}
    """
    by_frame_cam: dict[tuple, list] = {}
    for r in boxes:
        by_frame_cam.setdefault((r["frame"], r["cam"]), []).append(r)

    per_cam = {f"C{c + 1}": mm.MOTAccumulator(auto_id=False) for c in range(num_cameras)}
    cross = mm.MOTAccumulator(auto_id=False)
    ground_acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(gt):
        people = gt[frame]
        for c in range(num_cameras):
            g = [(p["person_id"], p["boxes"][c]) for p in people if c in p["boxes"]]
            pr = by_frame_cam.get((frame, c), [])
            d = iou_distance([b for _, b in g], [r["box"] for r in pr], iou_threshold)
            gid = [i for i, _ in g]
            per_cam[f"C{c + 1}"].update(gid, [r["local_id"] for r in pr], d, frameid=frame)
            # offset frame ids per camera so all cameras form one long sequence with shared identities
            cross.update(gid, [r["global_id"] for r in pr], d, frameid=c * 10**7 + frame)
        gp = ground.get(frame, {})
        d = euclidean_distance([p["world"] for p in people], list(gp.values()), ground_dist_cm)
        ground_acc.update([p["person_id"] for p in people], list(gp), d, frameid=frame)

    result = summarize({**per_cam, "cross_camera": cross, "ground": ground_acc})
    return result
