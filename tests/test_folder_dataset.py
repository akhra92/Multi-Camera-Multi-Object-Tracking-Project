"""Tests for the generic folder / manifest dataset (msad.data.FolderDataset)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from msad.data import FolderDataset, get_dataset
from msad.run import load_config, run_one

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "folder.yaml"
SR = 8000


def tone(rng, freq, seconds=1.0, channels=1, anomaly=False):
    t = np.arange(int(SR * seconds)) / SR
    y = np.sin(2 * np.pi * freq * t)[:, None] * np.ones(channels) + rng.normal(0, 0.1, (t.size, channels))
    if anomaly:
        y += 0.8 * np.sin(2 * np.pi * 2500 * t)[:, None]
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def write_wavs(folder: Path, n: int, rng, anomaly=False):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sf.write(folder / f"r{i:02d}.wav", tone(rng, 100 + i, anomaly=anomaly), SR)


def make_cfg(tmp_path: Path, root: Path, **folder):
    cfg = load_config(CONFIG, [f"data_root={root}", f"cache_dir={tmp_path / 'cache'}",
                               f"output_dir={tmp_path / 'out'}", "ae.epochs=2"])
    cfg["folder"].update(folder)
    return cfg


def files(clips):
    return {c.path for c in clips}


def test_auto_split_layout(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "ds"
    write_wavs(root / "pump" / "normal", 10, rng)
    write_wavs(root / "pump" / "anomaly" / "bearing", 3, rng, anomaly=True)
    write_wavs(root / "pump" / "anomaly" / "belt" / "loose", 2, rng, anomaly=True)
    ds = get_dataset(make_cfg(tmp_path, root))

    assert isinstance(ds, FolderDataset)
    assert ds.machines() == ["pump"]
    assert ds.modalities == ("sound",)
    train, test = ds.clips("pump", "train"), ds.clips("pump", "test")
    assert not files(train) & files(test)
    assert len(train) == 7 and all(c.label == 0 for c in train)
    assert sum(c.label == 0 for c in test) == 3
    assert sorted({c.fault for c in test if c.label == 1}) == ["bearing", "belt/loose"]


def test_explicit_split_in_data_root(tmp_path):
    rng = np.random.default_rng(1)
    root = tmp_path / "bench"
    write_wavs(root / "train" / "normal", 4, rng)
    write_wavs(root / "test" / "normal", 2, rng)
    write_wavs(root / "test" / "anomaly", 2, rng, anomaly=True)
    ds = get_dataset(make_cfg(tmp_path, root))

    assert ds.machines() == ["bench"]
    assert len(ds.clips("bench", "train")) == 4
    test = ds.clips("bench", "test")
    assert [c.label for c in test].count(1) == 2 and all(c.fault == "" for c in test if c.label == 1)


def test_csv_channels_segments_and_fusion(tmp_path):
    rng = np.random.default_rng(2)
    root = tmp_path / "rig"
    for folder, n, anomaly in (("normal", 8, False), ("anomaly/imbalance", 3, True)):
        (root / "m" / folder).mkdir(parents=True)
        for i in range(n):
            data = tone(rng, 60 + i, seconds=2.0, channels=4, anomaly=anomaly)  # 3 accel axes + microphone
            np.savetxt(root / "m" / folder / f"{i}.csv", data, delimiter=",", fmt="%.5f")
    cfg = make_cfg(tmp_path, root, sample_rate=SR, vibration_channels=[0, 1, 2], sound_channels=[3],
                   segment_seconds=1.0)
    ds = get_dataset(cfg)

    assert ds.modalities == ("sound", "vibration")
    train = ds.clips("m", "train")
    assert len(train) == 2 * len(files(train))  # 2 s files -> two 1 s segments each
    vib, sr = ds.load(train[0], "vibration")
    snd, _ = ds.load(train[0], "sound")
    assert vib.shape == (3, SR) and snd.shape == (1, SR) and sr == SR

    result = run_one(ds, "m", "knn", "fusion", cfg)
    assert 0.0 <= result["auc_all"] <= 1.0 and "auc_fault/imbalance" in result


def test_csv_needs_sample_rate(tmp_path):
    root = tmp_path / "x"
    (root / "normal").mkdir(parents=True)
    np.savetxt(root / "normal" / "a.csv", np.zeros((100, 1)), delimiter=",")
    np.savetxt(root / "normal" / "b.csv", np.zeros((100, 1)), delimiter=",")
    ds = get_dataset(make_cfg(tmp_path, root, segment_seconds=0.01))
    with pytest.raises(ValueError, match="sample_rate"):
        ds.clips("x", "train")


def test_manifest_and_npy(tmp_path):
    rng = np.random.default_rng(3)
    root = tmp_path / "raw"
    root.mkdir()
    rows = ["path,label,split,fault"]
    for i in range(6):
        np.save(root / f"n{i}.npy", tone(rng, 80 + i)[:, 0])  # 1-D arrays
        rows.append(f"n{i}.npy,normal,,")
    for i in range(2):
        np.save(root / f"f{i}.npy", tone(rng, 80, anomaly=True)[:, 0])
        rows.append(f"f{i}.npy,anomaly,test,gear")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows))
    cfg = make_cfg(tmp_path, root, manifest=str(manifest), sample_rate=SR)
    ds = get_dataset(cfg)

    assert ds.machines() == ["raw"]
    train, test = ds.clips("raw", "train"), ds.clips("raw", "test")
    assert len(train) + sum(c.label == 0 for c in test) == 6
    assert {c.fault for c in test if c.label == 1} == {"gear"}
    signal, _ = ds.load(train[0], "sound")
    assert signal.shape == (1, SR)

    result = run_one(ds, "raw", "gmm", "sound", cfg)
    assert {"auc_all", "pauc", "hmean", "f1"} <= set(result)


def test_files_per_fault_cap(tmp_path):
    rng = np.random.default_rng(4)
    root = tmp_path / "cap"
    write_wavs(root / "normal", 6, rng)
    write_wavs(root / "anomaly" / "a", 5, rng, anomaly=True)
    ds = get_dataset(make_cfg(tmp_path, root, files_per_fault=2))
    assert sum(c.label == 1 for c in ds.clips("cap", "test")) == 2
