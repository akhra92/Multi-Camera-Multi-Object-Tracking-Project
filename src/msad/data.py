"""Datasets. Each one lists clips and loads raw signals for the modalities it has.

DCASE 2024 Task 2 (sound only):
    <data_root>/<machine>/train/section_00_source_train_normal_0000_<attrs>.wav
    <data_root>/<machine>/test/section_00_target_test_anomaly_0012_<attrs>.wav
    Unlabelled evaluation files (e.g. section_00_0000.wav) get label=None.

MaFaulDa (sound + vibration, recorded together on one machine fault simulator):
    <data_root>/normal/<rotation_hz>.csv
    <data_root>/imbalance/10g/<rotation_hz>.csv
    <data_root>/underhang/outer_race/6g/<rotation_hz>.csv
    CSV columns: 0 tachometer, 1-3 underhang accelerometer (axial, radial, tangential),
    4-6 overhang accelerometer (axial, radial, tangential), 7 microphone. 50 kHz, 5 s.

Any other dataset ("folder"): see FolderDataset below - no code needed, only a folder layout or a manifest CSV.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

MODALITIES = ("sound", "vibration")


@dataclass(frozen=True)
class Clip:
    path: Path
    machine: str
    section: str
    domain: str | None  # "source" | "target" | None
    label: int | None   # 0 = normal, 1 = anomaly, None = unknown
    attrs: str = ""
    fault: str = ""     # fault type for anomalous clips, when the dataset provides it
    segment: int | None = None  # index of a fixed-length segment within the file


# ---------------------------------------------------------------- DCASE 2024 Task 2

_DCASE_PATTERN = re.compile(
    r"section_(?P<section>\d+)_(?P<domain>source|target)_(?P<split>train|test)"
    r"_(?P<label>normal|anomaly)_(?P<idx>\d+)(?:_(?P<attrs>.*))?\.wav$"
)


def parse_clip(path: Path, machine: str) -> Clip:
    m = _DCASE_PATTERN.search(path.name)
    if m is None:
        section = re.search(r"section_(\d+)", path.name)
        return Clip(path, machine, section.group(1) if section else "00", None, None)
    return Clip(
        path=path,
        machine=machine,
        section=m["section"],
        domain=m["domain"],
        label=int(m["label"] == "anomaly"),
        attrs=m["attrs"] or "",
    )


def list_clips(data_root: str | Path, machine: str, split: str) -> list[Clip]:
    folder = Path(data_root) / machine / split
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} not found. Download data first: python scripts/download_dcase2024.py --machines {machine}"
        )
    clips = [parse_clip(p, machine) for p in sorted(folder.glob("*.wav"))]
    if not clips:
        raise FileNotFoundError(f"no .wav files in {folder}")
    return clips


def list_machines(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    return sorted(p.name for p in root.iterdir() if (p / "train").is_dir()) if root.is_dir() else []


class DCASE:
    modalities = ("sound",)

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_root"])

    def machines(self) -> list[str]:
        return list_machines(self.root)

    def clips(self, machine: str, split: str) -> list[Clip]:
        return list_clips(self.root, machine, split)

    def file_clips(self, path: str | Path) -> list[Clip]:
        path = Path(path)
        return [parse_clip(path, path.parent.parent.name)]

    def load(self, clip: Clip, modality: str) -> tuple[np.ndarray, int]:
        """Return (channels, samples) signal and its sample rate."""
        y, sr = librosa.load(clip.path, sr=None, mono=True)
        return y[None, :], sr


# ---------------------------------------------------------------- MaFaulDa

_BEARING_GROUPS = ("underhang", "overhang")


@lru_cache(maxsize=None)
def _count_rows(path: Path) -> int:
    """Number of samples in a CSV without parsing it (much faster than reading the numbers)."""
    data = path.read_bytes()
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _fault_type(rel: Path) -> str:
    """normal / imbalance / horizontal-misalignment / vertical-misalignment / underhang/outer_race / ..."""
    parts = rel.parts
    return "/".join(parts[:2]) if parts[0] in _BEARING_GROUPS else parts[0]


class MaFaulDa:
    modalities = ("sound", "vibration")
    machine = "mafaulda"
    sample_rate = 50000

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_root"])
        self.cfg = cfg["mafaulda"]
        self.seed = cfg.get("seed", 0)
        self._splits: dict[str, list[tuple[Path, int, str]]] | None = None

    def machines(self) -> list[str]:
        return [self.machine]

    def _make_splits(self) -> dict[str, list[tuple[Path, int, str]]]:
        """Split by file (never by segment) so no recording appears in both train and test."""
        if not (self.root / "normal").is_dir():
            raise FileNotFoundError(
                f"{self.root / 'normal'} not found. Download data first: python scripts/download_mafaulda.py"
            )
        rng = np.random.default_rng(self.seed)
        normal = sorted((self.root / "normal").glob("*.csv"))
        order = rng.permutation(len(normal))
        n_test = max(1, round(len(normal) * self.cfg["test_fraction"]))
        test = [(normal[i], 0, "") for i in sorted(order[:n_test])]
        train = [(normal[i], 0, "") for i in sorted(order[n_test:])]

        by_fault: dict[str, list[Path]] = {}
        for p in sorted(self.root.rglob("*.csv")):
            fault = _fault_type(p.relative_to(self.root))
            if fault != "normal":
                by_fault.setdefault(fault, []).append(p)
        limit = self.cfg.get("files_per_fault")
        for fault, files in sorted(by_fault.items()):
            if limit and len(files) > limit:
                files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
            test += [(p, 1, fault) for p in files]
        return {"train": train, "test": test}

    def _segments(self, path: Path, label: int | None, fault: str) -> list[Clip]:
        n_samples = _count_rows(path)
        seg_len = int(self.cfg["segment_seconds"] * self.sample_rate)
        speed = path.stem
        return [Clip(path, self.machine, "00", None, label, f"{speed}Hz", fault, i)
                for i in range(max(1, n_samples // seg_len))]

    def clips(self, machine: str, split: str) -> list[Clip]:
        if self._splits is None:
            self._splits = self._make_splits()
        return [c for p, label, fault in self._splits[split] for c in self._segments(p, label, fault)]

    def file_clips(self, path: str | Path) -> list[Clip]:
        return self._segments(Path(path), None, "")

    @staticmethod
    @lru_cache(maxsize=4)  # segments of one file are loaded one after another
    def _read(path: Path) -> np.ndarray:
        return pd.read_csv(path, header=None, dtype=np.float32, engine="c").to_numpy()

    def load(self, clip: Clip, modality: str) -> tuple[np.ndarray, int]:
        data = self._read(clip.path)
        if clip.segment is not None:
            seg_len = int(self.cfg["segment_seconds"] * self.sample_rate)
            data = data[clip.segment * seg_len : (clip.segment + 1) * seg_len]
        cols = self.cfg["vibration_channels"] if modality == "vibration" else [self.cfg["sound_channel"]]
        return data[:, cols].T.copy(), self.sample_rate


# ---------------------------------------------------------------- Any dataset: folder layout or manifest

AUDIO_EXTS = (".wav", ".flac", ".ogg")
SIGNAL_EXTS = (*AUDIO_EXTS, ".csv", ".npy")


def _info(path: Path, csv_header: bool, sample_rate: float | None) -> tuple[int, float]:
    """(number of samples, sample rate) without loading the whole signal."""
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        info = sf.info(path)
        return info.frames, info.samplerate
    if sample_rate is None:
        raise ValueError(f"{path.name}: set folder.sample_rate in the config ({ext} files store no sample rate)")
    if ext == ".csv":
        return _count_rows(path) - int(csv_header), sample_rate
    if ext == ".npy":
        return np.load(path, mmap_mode="r").shape[0], sample_rate
    raise ValueError(f"unsupported file type {ext!r} ({path}); use one of {SIGNAL_EXTS}")


@lru_cache(maxsize=4)  # segments of one file are loaded one after another
def _read_signal(path: Path, csv_header: bool, sample_rate: float | None) -> tuple[np.ndarray, float]:
    """(samples, channels) float32 array and its sample rate."""
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        return sf.read(path, dtype="float32", always_2d=True)
    _, sr = _info(path, csv_header, sample_rate)
    if ext == ".csv":
        data = pd.read_csv(path, header=0 if csv_header else None, dtype=np.float32, engine="c").to_numpy()
    else:
        data = np.load(path).astype(np.float32)
    return (data[:, None] if data.ndim == 1 else data), sr


def _signal_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SIGNAL_EXTS) if folder.is_dir() else []


class FolderDataset:
    """Use any dataset without writing code: arrange its files in a standard layout, or list them in a manifest.

    Folder layout, per machine type (files may sit in deeper sub-folders):
        <data_root>/<machine>/normal/...              normal recordings, split into train/test by file
        <data_root>/<machine>/anomaly/<fault>/...     faulty recordings (test only); folder name = fault type
    or, when the dataset has an official split:
        <data_root>/<machine>/train/normal/...
        <data_root>/<machine>/test/normal/...
        <data_root>/<machine>/test/anomaly/<fault>/...
    With a single machine type, the normal/ (or train/) folders may sit directly in <data_root>.
    abnormal/ is accepted in place of anomaly/ (as in MIMII).

    Manifest (folder.manifest): a CSV with columns path,label and optional split,fault,machine,domain. path is
    relative to data_root; label is normal/anomaly (or abnormal, 0/1); an empty split means "split
    automatically"; domain is source/target (AUC reported per domain, as in DCASE) or empty.

    Files: .wav .flac .ogg (sample rate read from the file), .csv and .npy (one column per channel; set
    folder.sample_rate). folder.sound_channels / folder.vibration_channels pick the channels for each sensor.
    """

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_root"])
        self.cfg = cfg["folder"]
        self.seed = cfg.get("seed", 0)
        self.modalities = tuple(m for m in MODALITIES if self.cfg.get(f"{m}_channels") is not None)
        self._splits: dict[str, dict[str, list[tuple[Path, int, str, str | None]]]] = {}

    # -- listing ------------------------------------------------------------------------------------

    def _manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg["manifest"], dtype=str, keep_default_na=False)
        missing = {"path", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"manifest {self.cfg['manifest']} is missing column(s) {sorted(missing)}")
        for col, default in (("split", ""), ("fault", ""), ("machine", self.root.name), ("domain", "")):
            if col not in df.columns:
                df[col] = default
        df["machine"] = df["machine"].replace("", self.root.name)
        return df

    def machines(self) -> list[str]:
        if self.cfg.get("manifest"):
            return sorted(self._manifest()["machine"].unique())
        if any((self.root / d).is_dir() for d in ("normal", "train")):
            return [self.root.name]
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "normal").is_dir() or (p / "train").is_dir())

    def _machine_dir(self, machine: str) -> Path:
        if machine == self.root.name and any((self.root / d).is_dir() for d in ("normal", "train")):
            return self.root
        return self.root / machine

    def _records(self, machine: str) -> list[tuple[Path, int, str | None, str, str | None]]:
        """(path, label, split or None for automatic, fault type, domain) for every file of a machine type."""
        if self.cfg.get("manifest"):
            df = self._manifest()
            out = []
            for row in df[df["machine"] == machine].itertuples():
                label = {"normal": 0, "0": 0, "anomaly": 1, "abnormal": 1, "1": 1}.get(row.label.strip().lower())
                if label is None:
                    raise ValueError(f"manifest label must be normal/anomaly or 0/1, got {row.label!r} ({row.path})")
                domain = row.domain.strip().lower() or None
                if domain not in (None, "source", "target"):
                    raise ValueError(f"manifest domain must be source, target or empty, got {row.domain!r} ({row.path})")
                path = Path(row.path) if Path(row.path).is_absolute() else self.root / row.path
                out.append((path, label, row.split.strip().lower() or None, row.fault, domain))
            return out

        mdir = self._machine_dir(machine)

        def anomalies(parent: Path, split: str | None):
            return [(p, 1, split, "/".join(p.relative_to(folder).parent.parts), None)
                    for folder in (parent / "anomaly", parent / "abnormal") for p in _signal_files(folder)]

        if (mdir / "train").is_dir():
            train_dir = mdir / "train" / "normal" if (mdir / "train" / "normal").is_dir() else mdir / "train"
            return ([(p, 0, "train", "", None) for p in _signal_files(train_dir)]
                    + [(p, 0, "test", "", None) for p in _signal_files(mdir / "test" / "normal")]
                    + anomalies(mdir / "test", "test"))
        if (mdir / "normal").is_dir():
            return [(p, 0, None, "", None) for p in _signal_files(mdir / "normal")] + anomalies(mdir, "test")
        raise FileNotFoundError(f"{mdir} has neither normal/ nor train/ (see the FolderDataset docstring)")

    def _make_splits(self, machine: str) -> dict[str, list[tuple[Path, int, str, str | None]]]:
        """Split by file (never by segment) so no recording appears in both train and test."""
        records = self._records(machine)
        if not records:
            raise FileNotFoundError(f"no {'/'.join(SIGNAL_EXTS)} files found for machine {machine!r}")
        rng = np.random.default_rng(self.seed)
        splits = {"train": [], "test": []}
        auto_normal = [r for r in records if r[2] is None and r[1] == 0]
        order = rng.permutation(len(auto_normal))
        n_test = round(len(auto_normal) * self.cfg.get("test_fraction", 0.3)) if auto_normal else 0
        test_idx = set(order[:max(1, n_test)]) if auto_normal else set()
        for i, (p, label, _, fault, domain) in enumerate(auto_normal):
            splits["test" if i in test_idx else "train"].append((p, label, fault, domain))
        for p, label, split, fault, domain in records:
            if split is None and label == 0:
                continue
            if split == "train" and label == 1:
                continue  # training uses normal data only
            splits["train" if split == "train" else "test"].append((p, label, fault, domain))

        limit = self.cfg.get("files_per_fault")
        if limit:
            by_fault: dict[str, list] = {}
            for rec in splits["test"]:
                if rec[1] == 1:
                    by_fault.setdefault(rec[2], []).append(rec)
            kept = [r for r in splits["test"] if r[1] == 0]
            for fault, recs in sorted(by_fault.items()):
                kept += [recs[i] for i in sorted(rng.choice(len(recs), min(limit, len(recs)), replace=False))]
            splits["test"] = kept
        if not splits["train"]:
            raise ValueError(f"machine {machine!r} has no normal training files")
        return splits

    def _segments(self, path: Path, machine: str, label: int | None, fault: str,
                  domain: str | None = None) -> list[Clip]:
        seconds = self.cfg.get("segment_seconds")
        if not seconds:
            return [Clip(path, machine, "00", domain, label, "", fault)]
        n_samples, sr = _info(path, self.cfg.get("csv_header", False), self.cfg.get("sample_rate"))
        n = max(1, n_samples // int(seconds * sr))
        return [Clip(path, machine, "00", domain, label, "", fault, i) for i in range(n)]

    def clips(self, machine: str, split: str) -> list[Clip]:
        if machine not in self._splits:
            self._splits[machine] = self._make_splits(machine)
        return [c for p, label, fault, domain in self._splits[machine][split]
                for c in self._segments(p, machine, label, fault, domain)]

    def file_clips(self, path: str | Path) -> list[Clip]:
        return self._segments(Path(path), self.root.name, None, "")

    # -- loading ------------------------------------------------------------------------------------

    def load(self, clip: Clip, modality: str) -> tuple[np.ndarray, int]:
        data, sr = _read_signal(clip.path, self.cfg.get("csv_header", False), self.cfg.get("sample_rate"))
        if clip.segment is not None:
            seg_len = int(self.cfg["segment_seconds"] * sr)
            data = data[clip.segment * seg_len : (clip.segment + 1) * seg_len]
        channels = self.cfg.get(f"{modality}_channels")
        if channels is None:
            raise ValueError(f"folder.{modality}_channels is not set, so this dataset has no {modality} data")
        if max(channels) >= data.shape[1]:
            raise ValueError(f"{clip.path.name} has {data.shape[1]} channel(s); folder.{modality}_channels={channels}")
        return data[:, channels].T.copy(), sr


DATASETS = {"dcase2024": DCASE, "mafaulda": MaFaulDa, "folder": FolderDataset}


def get_dataset(cfg: dict):
    name = cfg.get("dataset", "dcase2024")
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name](cfg)
