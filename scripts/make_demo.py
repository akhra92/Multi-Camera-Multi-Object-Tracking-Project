"""Demo images for the README from a finished run (no models needed, reads the saved tracks).

    uv run python scripts/make_demo.py [--run outputs/wildtrack] [--cams 1 3 6] [--start 905] [--frames 40]

Writes assets/demo.gif (3 cameras + bird's-eye view, animated) and assets/demo_all_cameras.jpg (all 7 views at one
frame). Box colour and number = global identity: the same person has the same number in every camera.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from mcmot.visualize import draw_bev, draw_camera, mosaic
from mcmot.wildtrack import Wildtrack


def load_run(run: Path):
    d = pd.read_csv(run / "tracks_per_camera.csv")
    boxes: dict[tuple[int, int], list] = {}
    for r in d.itertuples():
        boxes.setdefault((r.frame, r.cam), []).append({"global_id": r.global_id, "box": [r.x1, r.y1, r.x2, r.y2]})
    ground: dict[int, dict] = {}
    for r in pd.read_csv(run / "tracks_ground.csv").itertuples():
        ground.setdefault(r.frame, {})[r.global_id] = np.array([r.x_cm, r.y_cm])
    return boxes, ground


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/wildtrack")
    ap.add_argument("--data", default="data/Wildtrack")
    ap.add_argument("--cams", type=int, nargs=3, default=[1, 3, 6], help="three cameras for the GIF (1-7)")
    ap.add_argument("--start", type=int, default=905, help="first dataset frame number")
    ap.add_argument("--frames", type=int, default=40, help="number of annotated frames (2 per second)")
    ap.add_argument("--fps", type=float, default=4, help="GIF playback speed (the data is 2 fps)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()

    ds = Wildtrack(args.data)
    boxes, ground = load_run(Path(args.run))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    frames = [f for f in ds.frames if f >= args.start][:args.frames]
    cams = [c - 1 for c in args.cams]

    history: dict[int, list] = {}
    for f in ds.frames:  # trails need the positions before the first GIF frame too
        if f >= frames[0]:
            break
        for g, xy in ground.get(f, {}).items():
            history.setdefault(g, []).append(xy)

    gif = []
    for f in frames:
        images = ds.read_images(f)
        for g, xy in ground.get(f, {}).items():
            history.setdefault(g, []).append(xy)
        tiles = [draw_camera(images[c], boxes.get((f, c), []), c) for c in cams]
        tiles.append(draw_bev(ground.get(f, {}), ds.annotations(f), history))
        canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        h = round(canvas.shape[0] * args.width / canvas.shape[1])
        gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(canvas, (args.width, h), interpolation=cv2.INTER_AREA),
                                                cv2.COLOR_BGR2RGB)))
        if f == frames[len(frames) // 2]:  # one still with all seven cameras
            still = [draw_camera(img, boxes.get((f, c), []), c) for c, img in enumerate(images)]
            still.append(draw_bev(ground.get(f, {}), ds.annotations(f), history))
            m = mosaic(still)[: 3 * 360]  # 3 x 3 grid; the last tile is empty
            cv2.imwrite(str(out / "demo_all_cameras.jpg"), cv2.resize(m, (1440, 810), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])

    # one shared palette for all frames keeps colours stable and the file small
    palette = gif[len(gif) // 2].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in gif]
    frames_p[0].save(out / "demo.gif", save_all=True, append_images=frames_p[1:], duration=round(1000 / args.fps),
                     loop=0, optimize=True)
    for name in ("demo.gif", "demo_all_cameras.jpg"):
        print(f"{out / name}: {(out / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
