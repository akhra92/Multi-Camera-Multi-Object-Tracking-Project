"""Convert IMAD-DS (multi-sensor, with domain shift) from Parquet to .npy plus a manifest CSV.

Dataset: https://doi.org/10.5281/zenodo.12665499  (license: CC BY-SA 4.0)
Two scaled industrial machines, BrushlessMotor.7z (1.6 GB) and RoboticArm.7z (3.5 GB); extract them with 7-Zip
into one folder. Recorded with an STEVAL-STWINBX1 sensor node: microphone (imp23absu_mic, 16 kHz), 3-axis
accelerometer (ism330dhcx_acc, ~7 kHz) and 3-axis gyroscope (ism330dhcx_gyro, ~7 kHz), one Parquet file per sensor
and segment with a Time column (seconds). CSVs list the segments of each split:
    <src>/<Machine>/train/attributes_normal_{source,target}_train.csv
    <src>/<Machine>/test/attributes_{normal,anomaly}_{source,target}_test.csv
Source domain = usual operating and noise conditions; target domain = changed load/speed or background noise,
with few training segments.

The sensors run at different rates, so each segment is cut to the time range all sensors cover and the
accelerometer (and gyroscope) is interpolated onto the microphone's timestamps. Output, at 16 kHz:
    <out>/<Machine>/<split>/<normal|anomaly>/<segment_id>.npy    columns: mic, acc x/y/z[, gyro x/y/z]
    <out>/manifest.csv    path,label,split,fault,machine,domain,domain_shift_op,domain_shift_env
The manifest's domain column gives AUC(source) and AUC(target) as in DCASE. Above ~3.5 kHz (half the
accelerometer's rate) the interpolated vibration channels carry no information; configs/imad_ds.yaml caps them.

Usage (needs pyarrow, installed by uv sync):
    python scripts/convert_imad_ds.py --src data/imad_ds_raw --out data/imad_ds
    python scripts/convert_imad_ds.py --src data/imad_ds_raw --out data/imad_ds --gyro
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

MIC, ACC, GYRO = "imp23absu_mic", "ism330dhcx_acc", "ism330dhcx_gyro"
SAMPLE_RATE = 16000
GYRO_WARM_UP_S = 0.035  # the gyroscope's first ~35 ms are unreliable (IMAD-DS baseline)
MANIFEST_COLUMNS = ["path", "label", "split", "fault", "machine", "domain", "domain_shift_op", "domain_shift_env"]

_ATTRS = re.compile(r"^attributes_(?P<label>normal|anomaly)_(?P<domain>source|target)_(?P<split>train|test)\.csv$")


def read_sensor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(time in seconds, (samples, channels) values) of one sensor file, sorted by time."""
    df = pd.read_parquet(path).sort_values("Time")
    values = df.drop(columns="Time").to_numpy(dtype=np.float64)
    return df["Time"].to_numpy(dtype=np.float64), values


def merge_sensors(files: dict[str, Path], gyro: bool) -> np.ndarray:
    """(samples, channels) float32 at the microphone's rate: mic, acc x/y/z[, gyro x/y/z]."""
    sensors = {name: read_sensor(path) for name, path in files.items()}
    start = max(t[0] for t, _ in sensors.values())
    if gyro:
        start = max(start, sensors[GYRO][0][0] + GYRO_WARM_UP_S)
    end = min(t[-1] for t, _ in sensors.values())
    t_mic, mic = sensors[MIC]
    keep = (t_mic >= start) & (t_mic <= end)
    if not keep.any():
        raise ValueError("sensors share no common time range")
    columns = [mic[keep, :1]]
    for name in (ACC, GYRO) if gyro else (ACC,):
        t, values = sensors[name]
        columns.append(np.stack([np.interp(t_mic[keep], t, values[:, c]) for c in range(values.shape[1])], axis=1))
    return np.concatenate(columns, axis=1).astype(np.float32)


def machine_dirs(src: Path) -> list[Path]:
    if (src / "train").is_dir():
        return [src]
    return sorted(p for p in src.iterdir() if (p / "train").is_dir()) if src.is_dir() else []


def convert(src: Path, out: Path, gyro: bool = False) -> pd.DataFrame:
    machines = machine_dirs(src)
    if not machines:
        raise FileNotFoundError(f"no <Machine>/train/ folders under {src}; extract BrushlessMotor.7z / RoboticArm.7z there")
    sensors = [MIC, ACC, GYRO] if gyro else [MIC, ACC]
    rows, skipped = [], 0
    for mdir in machines:
        for csv_path in sorted(mdir.glob("*/attributes_*.csv")):
            m = _ATTRS.match(csv_path.name)
            if m is None:
                continue
            meta = pd.read_csv(csv_path, index_col=0)
            for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"{mdir.name} {csv_path.stem[11:]}"):
                segment_id = Path(row[MIC]).stem.replace(f"{MIC}_", "")
                rel = Path(mdir.name, m["split"], m["label"], f"{segment_id}.npy")
                if not (out / rel).exists():
                    try:
                        data = merge_sensors({s: csv_path.parent / row[s] for s in sensors}, gyro)
                    except Exception as e:  # the baseline also skips unreadable segments
                        print(f"skipping {mdir.name}/{m['split']} segment {segment_id}: {e}")
                        skipped += 1
                        continue
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    np.save(out / rel, data)
                rows.append({
                    "path": rel.as_posix(),
                    "label": m["label"],
                    "split": m["split"],
                    "fault": str(row.get("anomaly_label", "")) if m["label"] == "anomaly" else "",
                    "machine": mdir.name,
                    "domain": m["domain"],
                    "domain_shift_op": row.get("domain_shift_op", ""),
                    "domain_shift_env": row.get("domain_shift_env", ""),
                })
    if skipped:
        print(f"skipped {skipped} unreadable segment(s)")

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path = out / "manifest.csv"
    if manifest_path.exists():  # keep rows of machines converted in an earlier run
        old = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        manifest = pd.concat([old[~old["machine"].isin([d.name for d in machines])], manifest], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/imad_ds_raw", help="folder with the extracted BrushlessMotor/, RoboticArm/")
    parser.add_argument("--out", default="data/imad_ds")
    parser.add_argument("--gyro", action="store_true", help="also keep the gyroscope (channels 4-6)")
    args = parser.parse_args()

    manifest = convert(Path(args.src), Path(args.out), args.gyro)
    summary = manifest.groupby(["machine", "split", "domain", "label"]).size()
    print(summary.to_string())
    print(f"-> {args.out}/manifest.csv ({SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
