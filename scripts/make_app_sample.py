"""Build the small default data the Streamlit app ships with (so it runs without the 6.8 GB dataset).

    uv run python scripts/make_app_sample.py [--data data/Wildtrack] [--start 905] [--frames 20] [--video-frames 60]

Writes, under assets/sample/:
    wildtrack/       a WILDTRACK-layout subset: all 7 cameras for `--frames` annotated frames, images at half
                     resolution (960x540 JPEG), intrinsics and annotation boxes scaled to match, extrinsics as-is.
                     mcmot.wildtrack.Wildtrack reads it like the full dataset.
    default_video.mp4  camera C1 for `--video-frames` frames (2 fps), the app's default single video.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from mcmot.single_video import H264Writer
from mcmot.wildtrack import CAMERAS, FPS, Wildtrack

SCALE = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Wildtrack")
    ap.add_argument("--start", type=int, default=905, help="first dataset frame number")
    ap.add_argument("--frames", type=int, default=20, help="annotated frames in the multi-camera sample")
    ap.add_argument("--video-frames", type=int, default=60, help="frames in the default single video")
    ap.add_argument("--out", default="assets/sample")
    args = ap.parse_args()

    ds = Wildtrack(args.data)
    out = Path(args.out)
    root = out / "wildtrack"
    if root.exists():
        shutil.rmtree(root)
    frames = [f for f in ds.frames if f >= args.start]

    for f in frames[:args.frames]:
        for c, img in enumerate(ds.read_images(f)):
            d = root / "Image_subsets" / f"C{c + 1}"
            d.mkdir(parents=True, exist_ok=True)
            small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"{f:08d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        people = json.loads((ds.annotation_dir / f"{f:08d}.json").read_text())
        for p in people:
            for v in p["views"]:
                if v["xmin"] != -1:
                    for k in ("xmin", "ymin", "xmax", "ymax"):
                        v[k] = round(v[k] * SCALE)
        (root / "annotations_positions").mkdir(parents=True, exist_ok=True)
        (root / "annotations_positions" / f"{f:08d}.json").write_text(json.dumps(people))

    calib = ds.annotation_dir.parent / "calibrations"
    shutil.copytree(calib / "extrinsic", root / "calibrations" / "extrinsic")
    (root / "calibrations" / "intrinsic_zero").mkdir(parents=True)
    for name, cam in zip(CAMERAS, ds.cameras):
        K = cam.K.copy()
        K[:2] *= SCALE  # pixel coordinates halve: focal lengths and principal point scale with the image
        fs = cv2.FileStorage(str(root / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml"),
                             cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", np.zeros((5, 1)))
        fs.release()

    writer = H264Writer(out / "default_video.mp4", FPS, quality=5)
    for f in frames[:args.video_frames]:
        img = cv2.imread(str(ds.image_path(0, f)))
        writer.write(cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    writer.close()

    size = lambda p: sum(q.stat().st_size for q in p.rglob("*") if q.is_file()) / 1e6
    print(f"{root}: {size(root):.1f} MB; {out / 'default_video.mp4'}: "
          f"{(out / 'default_video.mp4').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
