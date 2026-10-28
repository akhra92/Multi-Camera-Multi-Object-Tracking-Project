"""MIMII (abnormal/ folders), Paderborn and IMAD-DS converters, on tiny synthetic stand-ins for each dataset."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy.io import savemat

from msad.data import get_dataset
from msad.run import load_config, run_one

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cfg(tmp_path: Path, config: str, root: Path, **folder):
    cfg = load_config(ROOT / "configs" / config, [f"data_root={root}", f"cache_dir={tmp_path / 'cache'}",
                                                  f"output_dir={tmp_path / 'out'}", "ae.epochs=2"])
    cfg["folder"].update(folder)
    return cfg


def signal(rng, n, channels=1, anomaly=False, sr=16000):
    t = np.arange(n) / sr
    y = np.sin(2 * np.pi * 50 * t)[:, None] + rng.normal(0, 0.1, (n, channels))
    if anomaly:
        y += np.sin(2 * np.pi * 1500 * t)[:, None]
    return (0.2 * y / np.abs(y).max()).astype(np.float32)


# ---------------------------------------------------------------- MIMII: <machine id>/{normal,abnormal}/*.wav

def test_mimii_abnormal_folders(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "mimii" / "6dB" / "fan"
    for mid in ("id_00", "id_02"):
        for label, n in (("normal", 8), ("abnormal", 3)):
            (root / mid / label).mkdir(parents=True)
            for i in range(n):
                sf.write(root / mid / label / f"{i:08d}.wav", signal(rng, 8000, 8, label == "abnormal"), 16000)
    cfg = make_cfg(tmp_path, "mimii.yaml", root)
    ds = get_dataset(cfg)

    assert ds.machines() == ["id_00", "id_02"]
    test = ds.clips("id_00", "test")
    assert sum(c.label == 1 for c in test) == 3 and all(c.fault == "" for c in test)
    assert all(c.label == 0 for c in ds.clips("id_00", "train"))
    assert ds.load(test[0], "sound")[0].shape == (1, 8000)  # first of the 8 microphones
    assert 0.0 <= run_one(ds, "id_00", "gmm", "sound", cfg)["auc_all"] <= 1.0


def test_manifest_abnormal_label_and_domain(tmp_path):
    rng = np.random.default_rng(1)
    root = tmp_path / "raw"
    root.mkdir()
    rows = ["path,label,split,domain"]
    for i, (label, split, domain) in enumerate([("normal", "train", "source")] * 6 + [("normal", "train", "target")] * 2
                                               + [("normal", "test", "source"), ("normal", "test", "target")] * 2
                                               + [("abnormal", "test", "source"), ("1", "test", "target")] * 2):
        sf.write(root / f"{i}.wav", signal(rng, 8000, anomaly=label != "normal"), 16000)
        rows.append(f"{i}.wav,{label},{split},{domain}")
    (tmp_path / "m.csv").write_text("\n".join(rows))
    cfg = make_cfg(tmp_path, "folder.yaml", root, manifest=str(tmp_path / "m.csv"))
    ds = get_dataset(cfg)

    test = ds.clips("raw", "test")
    assert {c.domain for c in test} == {"source", "target"} and sum(c.label == 1 for c in test) == 4
    result = run_one(ds, "raw", "knn", "sound", cfg)
    assert {"auc_source", "auc_target", "hmean"} <= set(result)

    (tmp_path / "bad.csv").write_text("path,label,domain\n0.wav,normal,elsewhere\n")
    with pytest.raises(ValueError, match="domain"):
        get_dataset(make_cfg(tmp_path, "folder.yaml", root, manifest=str(tmp_path / "bad.csv"))).clips("raw", "train")


# ---------------------------------------------------------------- Paderborn: .mat structs -> .npy folder layout

def write_paderborn_mat(path: Path, rng, anomaly: bool, n=32000):
    """Same nesting as the real files: <stem>.Y is a 1 x k struct array with Name/Type/Data/Unit/Raster."""
    channels = [("force", n // 16), ("phase_current_1", n), ("vibration_1", n + 1)]
    y = np.zeros((1, len(channels)), dtype=[(f, "O") for f in ("Name", "Type", "Data", "Unit", "Raster")])
    for i, (name, length) in enumerate(channels):
        data = signal(rng, length, anomaly=anomaly and name == "vibration_1", sr=64000)[:, 0]
        y[0, i] = (name, "", data[None, :], "", "HostService")
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {path.stem: {"Info": "synthetic", "Y": y}})


def test_paderborn_converter(tmp_path):
    rng = np.random.default_rng(2)
    src, out = tmp_path / "paderborn_raw", tmp_path / "paderborn"
    for bearing in ("K001", "K002", "K005", "KA01", "KI04"):
        for idx in (1, 2, 3):
            write_paderborn_mat(src / bearing / f"N15_M07_F10_{bearing}_{idx}.mat", rng, bearing != "K00" + bearing[-1])
    (src / "KA08").mkdir()
    (src / "KA08" / "N15_M07_F10_KA08_2.mat").write_bytes(b"not a mat file")  # like the one broken real file
    write_paderborn_mat(src / "X" / "N15_M07_F10_K999_1.mat", rng, False)    # unknown bearing code

    convert = load_script("convert_paderborn").convert
    counts = convert(src, out, ["vibration_1", "phase_current_1"], test_healthy=["K005"])
    assert counts == {"converted": 15, "existing": 0, "skipped": 2}
    cond = out / "N15_M07_F10"
    assert len(list((cond / "train" / "normal").glob("K00[12]_*.npy"))) == 6
    assert len(list((cond / "test" / "normal").glob("K005_*.npy"))) == 3
    assert len(list((cond / "test" / "anomaly" / "outer_race" / "artificial").glob("KA01_*.npy"))) == 3
    assert len(list((cond / "test" / "anomaly" / "inner_race" / "lifetime").glob("KI04_*.npy"))) == 3
    assert np.load(next(cond.rglob("*.npy"))).shape == (32000, 2)  # truncated to the shorter 64 kHz channel
    assert convert(src, out, ["vibration_1", "phase_current_1"], ["K005"])["existing"] == 15

    cfg = make_cfg(tmp_path, "paderborn.yaml", out, segment_seconds=0.25)
    ds = get_dataset(cfg)
    assert ds.machines() == ["N15_M07_F10"] and ds.modalities == ("vibration",)
    assert len(ds.clips("N15_M07_F10", "train")) == 6 * 2  # 0.5 s files -> two 0.25 s clips
    result = run_one(ds, "N15_M07_F10", "ae", "vibration", cfg)
    assert {"auc_fault/outer_race/artificial", "auc_fault/inner_race/lifetime"} <= set(result)


# ---------------------------------------------------------------- IMAD-DS: Parquet per sensor + attribute CSVs

def write_imad_segment(folder: Path, seg: str, rng, anomaly: bool, seconds=0.5):
    for sensor, sr, n_ch, t0 in (("imp23absu_mic", 16000, 1, 0.0), ("ism330dhcx_acc", 7063, 3, 0.01),
                                 ("ism330dhcx_gyro", 7063, 3, 0.0)):
        t = t0 + np.arange(int(sr * (seconds - t0))) / sr
        values = signal(rng, t.size, n_ch, anomaly, sr)
        df = pd.DataFrame(values, columns=[f"{sensor}_{c}" for c in "xyz"[:n_ch]]).assign(Time=t)
        df.sample(frac=1, random_state=0).to_parquet(folder / f"{sensor}_{seg}.parquet")  # unsorted on purpose


def make_imad_ds(src: Path, rng) -> None:
    plan = [("train", "normal", "source", 6), ("train", "normal", "target", 2), ("test", "normal", "source", 2),
            ("test", "anomaly", "source", 2), ("test", "normal", "target", 2), ("test", "anomaly", "target", 2)]
    for split, label, domain, n in plan:
        folder = src / "BrushlessMotor" / split
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            seg = f"{split}_{label}_{domain}_{i}"
            write_imad_segment(folder, seg, rng, label == "anomaly")
            rows.append({s: f"{s}_{seg}.parquet" for s in ("imp23absu_mic", "ism330dhcx_acc", "ism330dhcx_gyro")}
                        | {"split_label": f"{label.title()}_{domain.title()}_{split.title()}",
                           "anomaly_label": "normal" if label == "normal" else "imbalance",
                           "domain_shift_op": "load_1" if domain == "source" else "load_3",
                           "domain_shift_env": "noise_off"})
        pd.DataFrame(rows).to_csv(folder / f"attributes_{label}_{domain}_{split}.csv")


def test_imad_ds_converter(tmp_path):
    rng = np.random.default_rng(3)
    src, out = tmp_path / "imad_ds_raw", tmp_path / "imad_ds"
    make_imad_ds(src, rng)
    convert = load_script("convert_imad_ds").convert

    manifest = convert(src, out)
    assert len(manifest) == 16 and set(manifest["machine"]) == {"BrushlessMotor"}
    assert set(manifest.loc[manifest["label"] == "anomaly", "fault"]) == {"imbalance"}
    assert (manifest.groupby(["split", "domain"]).size() == pd.Series(
        {("test", "source"): 4, ("test", "target"): 4, ("train", "source"): 6, ("train", "target"): 2})).all()
    data = np.load(out / manifest["path"][0])
    assert data.shape[1] == 4 and abs(data.shape[0] - 0.49 * 16000) <= 16  # mic + acc, from the acc start at 10 ms

    with_gyro = np.load(out.with_name("gyro") / convert(src, out.with_name("gyro"), gyro=True)["path"][0])
    assert with_gyro.shape[1] == 7 and with_gyro.shape[0] < data.shape[0]  # 35 ms gyroscope warm-up dropped

    cfg = make_cfg(tmp_path, "imad_ds.yaml", out, manifest=str(out / "manifest.csv"))
    ds = get_dataset(cfg)
    assert ds.machines() == ["BrushlessMotor"] and ds.modalities == ("sound", "vibration")
    assert len(ds.clips("BrushlessMotor", "train")) == 8
    result = run_one(ds, "BrushlessMotor", "knn", "fusion", cfg)
    assert {"auc_source", "auc_target", "auc_fault/imbalance"} <= set(result)
