"""Convert the Bosch CNC Machining dataset from HDF5 (.h5) to .npy in the folder layout.

Dataset: https://github.com/boschresearch/CNC_Machining  (data: CC BY 4.0; cite Tnani et al., Procedia CIRP 107, 2022)
Vibration from 3 brownfield CNC milling machines (M01-M03) in production, each running 15 processes (OP00-OP14),
recorded in 6 periods of 6 months (Feb_2019 ... Aug_2021). One tri-axial accelerometer (Bosch CISS) inside the
machine, X / Y / Z at 2 kHz (values in mg; Z includes gravity, about -1000). 1,632 good and 70 bad process runs of
about 20-70 s each. Get it with:
    git clone https://github.com/boschresearch/CNC_Machining data/cnc_machining_raw
Files: data/<machine>/<process>/<good|bad>/M01_Aug_2019_OP01_000.h5, dataset "vibration_data" of (samples, 3).

Output (configs/cnc_machining.yaml), one "machine" per CNC machine, normal runs split by recording period so that
the test periods are later than (and never mixed with) the training periods:
    <out>/M01/train/normal/M01_Aug_2019_OP01_000.npy       good runs outside --test-periods
    <out>/M01/test/normal/M01_Feb_2021_OP01_003.npy        good runs in --test-periods
    <out>/M01/test/anomaly/OP01/M01_Aug_2019_OP01_000.npy  all bad runs; fault type = process
Each .npy holds (samples, 3) float32 at 2 kHz, columns X, Y, Z.

Usage (needs h5py, installed by uv sync):
    python scripts/convert_cnc_machining.py --src data/cnc_machining_raw --out data/cnc_machining
    python scripts/convert_cnc_machining.py --src data/cnc_machining_raw --test-periods Aug_2021
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 2000
PERIODS = ["Feb_2019", "Aug_2019", "Feb_2020", "Aug_2020", "Feb_2021", "Aug_2021"]

_NAME = re.compile(r"^(?P<machine>M\d+)_(?P<period>[A-Z][a-z]{2}_\d{4})_(?P<process>OP\d+)_(?P<idx>\d+)$")


def read_vibration(path: Path) -> np.ndarray:
    """(samples, 3) float32 array from one CNC Machining .h5 file."""
    with h5py.File(path, "r") as f:
        data = np.asarray(f["vibration_data"][:], dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"expected (samples, 3), got {data.shape}")
    return data


def convert(src: Path, out: Path, test_periods: list[str]) -> dict[str, int]:
    files = sorted(src.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files under {src}; git clone the repository there first")
    counts = {"converted": 0, "existing": 0, "skipped": 0}
    for path in tqdm(files, desc="cnc_machining"):
        m = _NAME.match(path.stem)
        label = path.parent.name
        if m is None or label not in ("good", "bad"):
            print(f"skipping {path}: unknown file name or label folder")
            counts["skipped"] += 1
            continue
        if label == "bad":
            folder = Path("test", "anomaly", m["process"])
        else:
            folder = Path("test" if m["period"] in test_periods else "train", "normal")
        dest = out / m["machine"] / folder / f"{path.stem}.npy"
        if dest.exists():
            counts["existing"] += 1
            continue
        try:
            data = read_vibration(path)
        except Exception as e:
            print(f"skipping {path.name}: {e}")
            counts["skipped"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, data)
        counts["converted"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/cnc_machining_raw", help="clone of the CNC_Machining repository")
    parser.add_argument("--out", default="data/cnc_machining")
    parser.add_argument("--test-periods", nargs="+", default=["Feb_2021", "Aug_2021"], choices=PERIODS,
                        help="periods whose good runs are held out for testing; the others are used for training")
    args = parser.parse_args()
    if set(args.test_periods) == set(PERIODS):
        parser.error("--test-periods must leave at least one period for training")

    counts = convert(Path(args.src), Path(args.out), args.test_periods)
    print(f"{counts} -> {args.out} ({SAMPLE_RATE} Hz, columns: X, Y, Z)")


if __name__ == "__main__":
    main()
