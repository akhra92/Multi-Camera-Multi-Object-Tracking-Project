"""Generate tiny synthetic datasets to smoke-test the pipeline without downloading anything.

dcase    - DCASE layout, sound only. Normal = motor hum + noise; anomaly adds a whine or clicks.
           Target-domain clips run at a different speed (domain shift).
mafaulda - MaFaulDa layout, 8-column CSV (tachometer, 2 x 3-axis accelerometer, microphone) at 50 kHz.
           Faults: imbalance (strong 1x rotation), misalignment (strong 2x), bearing outer race
           (impacts at 3x rotation ringing a 3 kHz resonance). Faults are clearer in vibration than in sound.
folder   - generic folder layout (configs/folder.yaml): <out>/pump/normal/*.wav and
           <out>/pump/anomaly/{whine,clicks}/*.wav, split into train/test automatically.

    python scripts/make_synthetic.py dcase --out data/synthetic
    python -m msad.run --machine synth --model gmm knn ae --set data_root=data/synthetic ae.epochs=5
    python scripts/make_synthetic.py mafaulda --out data/synthetic_mafaulda
    python -m msad.run --config configs/mafaulda.yaml --modality sound vibration fusion --model knn ae \
        --set data_root=data/synthetic_mafaulda ae.epochs=5
    python scripts/make_synthetic.py folder --out data/synthetic_folder
    python -m msad.run --config configs/folder.yaml --model gmm knn ae --set data_root=data/synthetic_folder ae.epochs=5
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def machine_sound(rng, f0, seconds, anomaly, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    y = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3)) / h for h in range(1, 8))
    y = y * (1 + 0.05 * np.sin(2 * np.pi * 3 * t)) + rng.normal(0, 0.3, t.size)
    if anomaly:  # True = random kind, or "whine" / "clicks"
        kind = anomaly if isinstance(anomaly, str) else ("whine" if rng.random() < 0.5 else "clicks")
        if kind == "whine":
            y += 0.25 * np.sin(2 * np.pi * rng.uniform(2500, 4000) * t)
        else:
            clicks = np.zeros_like(t)
            clicks[:: int(sr / rng.uniform(15, 30))] = 3.0
            y += np.convolve(clicks, np.hanning(64), mode="same")
    return (0.1 * y / np.abs(y).max()).astype(np.float32)


def make_dcase(out: Path, machine: str, seconds: float, rng):
    sr = 16000
    root = out / machine
    speeds = {"source": 50.0, "target": 60.0}
    plan = [("train", "source", "normal", 90), ("train", "target", "normal", 10)] + [
        ("test", d, lab, 25) for d in speeds for lab in ("normal", "anomaly")
    ]
    for split, domain, label, n in plan:
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            f0 = speeds[domain] * rng.uniform(0.97, 1.03)
            y = machine_sound(rng, f0, seconds, label == "anomaly", sr)
            name = f"section_00_{domain}_{split}_{label}_{i:04d}_speed_{int(speeds[domain])}.wav"
            sf.write(root / split / name, y, sr)
    print(f"wrote synthetic DCASE-style dataset to {root}")


def mafaulda_record(rng, f_rot, seconds, fault, sr=50000):
    t = np.arange(int(sr * seconds)) / sr
    tach = 5.0 * (np.sin(2 * np.pi * f_rot * t) > 0.99)
    one_x, two_x = 1.0, 0.3
    if fault == "imbalance":
        one_x *= rng.uniform(2.5, 4.0)
    if fault == "misalignment":
        two_x *= rng.uniform(4.0, 6.0)
    shaft = one_x * np.sin(2 * np.pi * f_rot * t) + two_x * np.sin(4 * np.pi * f_rot * t + 1.0)
    impacts = np.zeros_like(t)
    if fault == "outer_race":
        impacts[:: int(sr / (2.998 * f_rot))] = rng.uniform(4.0, 6.0)
        ring = np.exp(-np.arange(400) / 60) * np.sin(2 * np.pi * 3000 * np.arange(400) / sr)
        impacts = np.convolve(impacts, ring, mode="same")
    accel = [g * shaft + 0.5 * impacts * g + rng.normal(0, 0.2, t.size) for g in (0.6, 1.0, 0.8, 0.5, 0.9, 0.7)]
    mic = 0.2 * shaft + 0.05 * impacts + machine_sound(rng, f_rot * 4, seconds, False, sr) * 5 + rng.normal(0, 0.3, t.size)
    return np.stack([tach, *accel, mic], axis=1).astype(np.float32)


def make_mafaulda(out: Path, seconds: float, rng):
    groups = {
        "normal": ("", 30),
        "imbalance/10g": ("imbalance", 8),
        "horizontal-misalignment/1.0mm": ("misalignment", 8),
        "underhang/outer_race/6g": ("outer_race", 8),
    }
    for folder, (fault, n) in groups.items():
        (out / folder).mkdir(parents=True, exist_ok=True)
        for f_rot in np.linspace(15, 55, n):
            data = mafaulda_record(rng, f_rot * rng.uniform(0.99, 1.01), seconds, fault)
            np.savetxt(out / folder / f"{f_rot:.4f}.csv", data, delimiter=",", fmt="%.5f")
    print(f"wrote synthetic MaFaulDa-style dataset to {out}")


def make_folder(out: Path, seconds: float, rng):
    sr = 16000
    plan = {"normal": (False, 40), "anomaly/whine": ("whine", 10), "anomaly/clicks": ("clicks", 10)}
    for folder, (anomaly, n) in plan.items():
        (out / "pump" / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            y = machine_sound(rng, 50.0 * rng.uniform(0.97, 1.03), seconds, anomaly, sr)
            sf.write(out / "pump" / folder / f"rec_{i:03d}.wav", y, sr)
    print(f"wrote synthetic folder-layout dataset to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["dcase", "mafaulda", "folder"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--machine", default="synth", help="dcase only")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    if args.kind == "dcase":
        make_dcase(Path(args.out or "data/synthetic"), args.machine, args.seconds, rng)
    elif args.kind == "mafaulda":
        make_mafaulda(Path(args.out or "data/synthetic_mafaulda"), args.seconds, rng)
    else:
        make_folder(Path(args.out or "data/synthetic_folder"), args.seconds, rng)


if __name__ == "__main__":
    main()
