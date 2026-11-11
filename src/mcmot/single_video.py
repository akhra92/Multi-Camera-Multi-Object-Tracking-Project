"""Tracking in one ordinary video (no calibration): YOLO person detection + DeepSORT, used by the Streamlit app.

Without calibration there is no ground plane and no cross-camera association, so ids are DeepSORT's local ids.
Output videos are H.264 (via the ffmpeg binary bundled with imageio-ffmpeg) so browsers can play them;
OpenCV's pip wheels can only write mp4v, which browsers do not decode.
"""
from pathlib import Path

import cv2
import numpy as np

from .detection import PersonDetector
from .tracking import AppearanceEmbedder, CameraTracker
from .visualize import draw_boxes


class H264Writer:
    """Write BGR frames to a browser-playable .mp4 (H.264, yuv420p)."""

    def __init__(self, path: str | Path, fps: float, quality: float = 7):
        """quality: 0 (smallest) .. 10 (best)."""
        self.path, self.fps, self.quality, self._gen = str(path), fps, quality, None

    def write(self, frame: np.ndarray) -> None:
        if self._gen is None:
            import imageio_ffmpeg

            h, w = frame.shape[:2]
            self._gen = imageio_ffmpeg.write_frames(self.path, (w, h), fps=self.fps, codec="libx264",
                                                    pix_fmt_out="yuv420p", quality=self.quality, macro_block_size=2,
                                                    ffmpeg_log_level="error")
            self._gen.send(None)
        self._gen.send(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()


def video_info(path: str | Path) -> tuple[float, int, int, int]:
    """(fps, frame count, width, height)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    info = (cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def track_video(path: str | Path, out_path: str | Path, det: PersonDetector, emb: AppearanceEmbedder,
                stride: int = 1, max_frames: int | None = None, width: int = 960, track_seconds: float = 2.5,
                progress=None) -> list[dict]:
    """Track people in every `stride`-th frame (at most `max_frames` of them), resized to `width` px.
    Writes the annotated video to `out_path` and returns rows {'frame', 'track_id', 'box', 'score'}.
    Tracks survive `track_seconds` without a detection."""
    fps, total, w, h = video_info(path)
    width = min(width, w) // 2 * 2
    height = round(h * width / w / 2) * 2  # even sizes for yuv420p
    n = -(-total // stride) if total > 0 else max_frames or 0
    if max_frames:
        n = min(n, max_frames)
    out_fps = fps / stride
    tracker = CameraTracker(max_age=max(1, round(track_seconds * out_fps)), n_init=2)
    writer = H264Writer(out_path, out_fps)
    cap = cv2.VideoCapture(str(path))
    rows, i, done = [], 0, 0
    try:
        while max_frames is None or done < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                boxes, scores = det([frame])[0]
                tracks = tracker.update(boxes, scores, emb(frame, boxes))
                current = [{"frame": i, "track_id": t.track_id, "box": t.box.tolist(), "score": t.score}
                           for t in tracks]
                rows += current
                writer.write(draw_boxes(frame, current, id_key="track_id"))
                done += 1
                if progress:
                    progress(done, max(n, done))
            i += 1
    finally:
        cap.release()
        writer.close()
    return rows
