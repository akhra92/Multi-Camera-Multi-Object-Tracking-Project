"""End-to-end multi-camera multi-object tracking on WILDTRACK.

    python -m mcmot.run --config configs/wildtrack.yaml [--set frames=test detector.imgsz=960] [--no-video]

Steps per frame: read the 7 images -> YOLO person detection -> appearance embeddings -> DeepSORT per camera ->
foot point -> ground plane -> fuse cameras + ground-plane tracking (global ids). Then evaluation and video.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .association import GroundTracker, Observation
from .detection import PersonDetector
from .evaluate import evaluate
from .geometry import foot_points
from .tracking import AppearanceEmbedder, CameraTracker
from .visualize import draw_bev, draw_camera, mosaic
from .wildtrack import FPS, Wildtrack, in_area


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:  # key.sub=value
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def track_all(ds: Wildtrack, frames: list[int], cfg: dict) -> tuple[list[dict], dict]:
    """Detection + DeepSORT in every camera + ground-plane association.
    Returns box rows (with global ids) and frame -> {global id: (x, y)}."""
    det = PersonDetector(**cfg["detector"])
    emb = AppearanceEmbedder(gpu=cfg["detector"]["device"] != "cpu")
    trackers = [CameraTracker(**cfg["tracker"]) for _ in range(ds.num_cameras)]
    a = cfg["association"]
    ground_tracker = GroundTracker(a["fuse_radius_cm"], a["max_step_cm"], a["vote_weight"], a["max_age"],
                                   a["min_length"], a["min_cameras"])
    rows = []
    for t, frame in enumerate(tqdm(frames, desc="detect + track")):
        images = ds.read_images(frame)
        obs = []
        for cam, (img, (boxes, scores)) in enumerate(zip(images, det(images))):
            for o in trackers[cam].update(boxes, scores, emb(img, boxes)):
                xy = ds.cameras[cam].image_to_ground(foot_points(o.box))[0]
                if not in_area(xy, a["area_margin_cm"])[0]:
                    continue  # WILDTRACK only annotates people inside the 12 m x 36 m area
                rows.append({"frame": frame, "cam": cam, "local_id": o.track_id, "box": o.box.tolist(),
                             "score": o.score, "gx": xy[0], "gy": xy[1]})
                obs.append(Observation(cam, o.track_id, xy))
        ground_tracker.update(t, frame, obs)
    assign, ground = ground_tracker.results()
    rows = [{**r, "global_id": assign[k]} for r in rows if (k := (r["frame"], r["cam"], r["local_id"])) in assign]
    return rows, ground


def write_video(ds: Wildtrack, frames: list[int], rows: list[dict], ground: dict, gt: dict, path: Path) -> None:
    by_frame: dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    writer = None
    history: dict[int, list] = {}
    for frame in tqdm(frames, desc="video"):
        images = ds.read_images(frame)
        tiles = [draw_camera(img, [r for r in by_frame.get(frame, []) if r["cam"] == c], c)
                 for c, img in enumerate(images)]
        for g, xy in ground.get(frame, {}).items():
            history.setdefault(g, []).append(xy)
        tiles.append(draw_bev(ground.get(frame, {}), gt[frame], history))
        canvas = mosaic(tiles)
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, canvas.shape[1::-1])
        writer.write(canvas)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/wildtrack.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. tracker.max_age=10")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = Wildtrack(cfg["data_root"])
    frames = ds.split(cfg["frames"])
    gt = {f: ds.annotations(f) for f in frames}
    print(f"WILDTRACK: {ds.num_cameras} cameras, {len(frames)} frames ({cfg['frames']}), "
          f"{len({p['person_id'] for f in frames for p in gt[f]})} people in ground truth")

    rows, ground = track_all(ds, frames, cfg)
    print(f"{len({(r['cam'], r['local_id']) for r in rows})} single-camera tracks -> "
          f"{len({g for d in ground.values() for g in d})} global identities")

    pd.DataFrame([{**{k: v for k, v in r.items() if k != "box"},
                   **dict(zip(["x1", "y1", "x2", "y2"], r["box"]))} for r in rows]).to_csv(
        out / "tracks_per_camera.csv", index=False)
    pd.DataFrame([{"frame": f, "global_id": g, "x_cm": xy[0], "y_cm": xy[1]}
                  for f, d in sorted(ground.items()) for g, xy in d.items()]).to_csv(
        out / "tracks_ground.csv", index=False)

    e = cfg["evaluation"]
    metrics = evaluate(gt, rows, ground, ds.num_cameras, e["iou_threshold"], e["ground_dist_cm"])
    (out / "metrics.json").write_text(json.dumps({"config": cfg, "metrics": metrics}, indent=1))
    table = pd.DataFrame(metrics).T[["mota", "idf1", "precision", "recall", "num_switches", "num_unique_objects"]]
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    if not args.no_video:
        write_video(ds, frames, rows, ground, gt, out / "tracking.mp4")
    print(f"done in {time.time() - t0:.0f} s, results in {out}/")


if __name__ == "__main__":
    main()
