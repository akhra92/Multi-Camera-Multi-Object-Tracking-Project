"""Download and unzip MIMII (Hitachi) from Zenodo: sound of fans, pumps, slide rails and valves.

Dataset: https://doi.org/10.5281/zenodo.3384388  (license: CC BY-SA 4.0)
4 machine types x 4 machine IDs (id_00, id_02, id_04, id_06), 8-channel microphone array, 16 kHz, 10 s per file.
Each machine type is recorded with factory noise mixed in at three levels (SNR -6, 0 and 6 dB), one zip each
(6.6-10.4 GB). Faulty recordings sit in abnormal/ folders, which configs/mimii.yaml reads directly.

Usage:
    python scripts/download_mimii.py --machines fan --snr 6        # -> data/mimii/6dB/fan/id_00/{normal,abnormal}
    python scripts/download_mimii.py --machines all --snr -6 0 6   # everything, ~96 GB zipped
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

RECORD_URL = "https://zenodo.org/api/records/3384388/files/{snr}_dB_{machine}.zip/content"
MACHINES = ["fan", "pump", "slider", "valve"]
SNRS = ["-6", "0", "6"]


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", nargs="+", default=["fan"], help=f"any of {MACHINES} or 'all'")
    parser.add_argument("--snr", nargs="+", default=["6"], help=f"noise level(s) in dB: {SNRS}")
    parser.add_argument("--out", default="data/mimii")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    machines = MACHINES if args.machines == ["all"] else args.machines
    if set(machines) - set(MACHINES):
        parser.error(f"unknown machine types: {sorted(set(machines) - set(MACHINES))}")
    if set(args.snr) - set(SNRS):
        parser.error(f"unknown SNR(s): {sorted(set(args.snr) - set(SNRS))}; choose from {SNRS}")

    for snr in args.snr:
        out = Path(args.out) / f"{snr}dB"
        out.mkdir(parents=True, exist_ok=True)
        for machine in machines:
            if any(out.rglob(f"{machine}/id_*")):
                print(f"{snr} dB {machine}: already present, skipping")
                continue
            zip_path = out / f"{snr}_dB_{machine}.zip"
            if not zip_path.exists():
                download(RECORD_URL.format(snr=snr, machine=machine), zip_path)
            print(f"{snr} dB {machine}: extracting")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out)
            if not args.keep_zip:
                zip_path.unlink()
            roots = sorted({p.parent for p in out.rglob("id_*") if p.parent.name == machine})
            for root in roots:
                print(f"  data_root for configs/mimii.yaml: {root.as_posix()}")


if __name__ == "__main__":
    main()
