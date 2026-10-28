"""Convert the Paderborn (KAt) bearing dataset from MATLAB .mat files to .npy in the folder layout.

Dataset: https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download
32 ball bearings (type 6203) on an electric motor test rig: 6 healthy, 12 with artificial damage, 14 with damage
from accelerated lifetime tests. Each bearing has one .rar archive (download all of them and extract with 7-Zip
into one folder). Inside: <bearing>/N15_M07_F10_KA01_1.mat, i.e. operating condition (rotation speed N x100 rpm,
load torque M x0.1 Nm, radial force F x100 N), bearing code and measurement number (20 per condition, 4 s each).
Signals: vibration_1 (bearing housing accelerometer), phase_current_1/2 at 64 kHz; force, speed, torque at 4 kHz.

Output (configs/paderborn.yaml), one "machine" per operating condition, split by bearing so that no bearing is
in both train and test:
    <out>/N15_M07_F10/train/normal/K001_N15_M07_F10_1.npy         healthy bearings not in --test-healthy
    <out>/N15_M07_F10/test/normal/K005_...npy                     healthy bearings in --test-healthy
    <out>/N15_M07_F10/test/anomaly/outer_race/artificial/KA01_...npy
    <out>/N15_M07_F10/test/anomaly/inner_race/lifetime/KI04_...npy  (fault type = component/damage method)
Each .npy holds (samples, channels) float32 at 64 kHz, one column per --signals entry.

Usage:
    python scripts/convert_paderborn.py --src data/paderborn_raw --out data/paderborn
    python scripts/convert_paderborn.py --src data/paderborn_raw --signals vibration_1 phase_current_1 phase_current_2
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

SAMPLE_RATE = 64000
SIGNALS_64K = ["vibration_1", "phase_current_1", "phase_current_2"]

# Bearing code -> (damaged component, damage method); healthy bearings map to None.
# From Lessmeier et al. (2016), "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems...".
BEARINGS = {
    **{f"K00{i}": None for i in range(1, 7)},
    **{code: ("outer_race", "artificial") for code in ("KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09")},
    **{code: ("inner_race", "artificial") for code in ("KI01", "KI03", "KI05", "KI07", "KI08")},
    **{code: ("outer_race", "lifetime") for code in ("KA04", "KA15", "KA16", "KA22", "KA30")},
    **{code: ("inner_race", "lifetime") for code in ("KI04", "KI14", "KI16", "KI17", "KI18", "KI21")},
    **{code: ("combined", "lifetime") for code in ("KB23", "KB24", "KB27")},
}

_NAME = re.compile(r"^(?P<cond>N\d+_M\d+_F\d+)_(?P<bearing>K[A-Z]?\d+)_(?P<idx>\d+)$")


def read_signals(path: Path, signals: list[str]) -> np.ndarray:
    """(samples, len(signals)) float32 array from one Paderborn .mat file."""
    mat = loadmat(path, simplify_cells=True)
    record = next(v for k, v in mat.items() if not k.startswith("__"))
    channels = {}
    for ch in np.atleast_1d(record["Y"]):
        values = list(ch.values())  # fields: Name, Type?, Data, Unit, Raster (looked up by name when present)
        name = str(ch.get("Name", values[0])).strip().lower()
        channels[name] = np.asarray(ch.get("Data", values[2]), dtype=np.float32).ravel()
    missing = [s for s in signals if s not in channels]
    if missing:
        raise KeyError(f"signal(s) {missing} not in file; it has {sorted(channels)}")
    n = min(channels[s].size for s in signals)  # 64 kHz channels can differ by a sample or two
    return np.stack([channels[s][:n] for s in signals], axis=1)


def convert(src: Path, out: Path, signals: list[str], test_healthy: list[str]) -> dict[str, int]:
    files = sorted(src.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no .mat files under {src}; extract the .rar archives there first")
    counts = {"converted": 0, "existing": 0, "skipped": 0}
    for path in tqdm(files, desc="paderborn"):
        m = _NAME.match(path.stem)
        if m is None or m["bearing"] not in BEARINGS:
            print(f"skipping {path.name}: unknown file name or bearing code")
            counts["skipped"] += 1
            continue
        damage = BEARINGS[m["bearing"]]
        if damage is None:
            folder = Path("test" if m["bearing"] in test_healthy else "train") / "normal"
        else:
            folder = Path("test", "anomaly", *damage)
        dest = out / m["cond"] / folder / f"{m['bearing']}_{m['cond']}_{m['idx']}.npy"
        if dest.exists():
            counts["existing"] += 1
            continue
        try:
            data = read_signals(path, signals)
        except Exception as e:  # a few files are known to be unreadable, e.g. KA08/N15_M01_F10_KA08_2.mat
            print(f"skipping {path.name}: {e}")
            counts["skipped"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, data)
        counts["converted"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/paderborn_raw", help="folder with the extracted bearing folders")
    parser.add_argument("--out", default="data/paderborn")
    parser.add_argument("--signals", nargs="+", default=["vibration_1"], choices=SIGNALS_64K,
                        help="64 kHz signals to keep, one column each")
    parser.add_argument("--test-healthy", nargs="+", default=["K005", "K006"],
                        help="healthy bearings held out for testing; the others are used for training")
    args = parser.parse_args()
    healthy = [b for b, damage in BEARINGS.items() if damage is None]
    if set(args.test_healthy) - set(healthy) or set(args.test_healthy) == set(healthy):
        parser.error(f"--test-healthy must be a strict subset of {healthy}")

    counts = convert(Path(args.src), Path(args.out), args.signals, args.test_healthy)
    print(f"{counts} -> {args.out} ({SAMPLE_RATE} Hz, columns: {', '.join(args.signals)})")


if __name__ == "__main__":
    main()
