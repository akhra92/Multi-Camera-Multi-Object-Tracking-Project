"""Single-camera tracking with DeepSORT (one tracker per camera).

DeepSORT = Kalman filter (predicts where each track moves) + Hungarian matching of predictions to new detections,
using an appearance embedding (cosine distance) gated by motion, then IoU for the leftovers. A track is confirmed
after `n_init` consecutive hits and deleted after `max_age` missed frames.

One appearance embedder is shared by all cameras (loaded once, batched per image).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackObs:
    """One confirmed track matched to a detection in one frame."""
    track_id: int
    box: np.ndarray  # x1 y1 x2 y2 (the detection box)
    score: float
    feature: np.ndarray  # L2-normalised appearance embedding


class AppearanceEmbedder:
    """ImageNet MobileNetV2 bottleneck features (DeepSORT-realtime's default embedder), L2-normalised."""

    def __init__(self, gpu: bool = True):
        from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

        self.model = MobileNetv2_Embedder(half=gpu, max_batch_size=64, bgr=True, gpu=gpu)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 1280), dtype=np.float32)
        h, w = image.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            crops.append(image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])
        feats = np.asarray(self.model.predict(crops), dtype=np.float32)
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)


class CameraTracker:
    def __init__(self, max_age=5, n_init=2, max_iou_distance=0.7, max_cosine_distance=0.3, nn_budget=50):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance,
                                max_cosine_distance=max_cosine_distance, nn_budget=nn_budget, embedder=None)

    def update(self, boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray) -> list[TrackObs]:
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, feats = boxes[keep], scores[keep], feats[keep]
        dets = [([b[0], b[1], b[2] - b[0], b[3] - b[1]], float(s), "person") for b, s in zip(boxes, scores)]
        # `others` carries the detection index, so each track can be tied back to its box and embedding
        tracks = self.tracker.update_tracks(dets, embeds=list(feats), others=list(range(len(dets))))
        out = []
        for t in tracks:
            i = t.get_det_supplementary()
            if t.is_confirmed() and t.time_since_update == 0 and i is not None:
                out.append(TrackObs(int(t.track_id), boxes[i], float(scores[i]), feats[i]))
        return out
