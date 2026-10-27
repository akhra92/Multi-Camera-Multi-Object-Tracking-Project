# Machine Fault Detection from Sound and Vibration

Unsupervised detection of faulty machines from **sound (microphones)** and **vibration (accelerometers)**, alone or
combined. Models are trained on **normal operation only**, then flag anything that sounds or vibrates differently.

| Dataset | Sensors | Used for |
|---|---|---|
| DCASE 2024 Task 2 | sound | benchmark on 7 machine types with published results to compare against |
| MaFaulDa | sound + vibration, recorded together | comparing sound vs. vibration vs. fusion on the same machine |
| Any other dataset | sound and/or vibration | no code needed: folder layout or manifest CSV (`configs/folder.yaml`) |

## Setup

Linux / GPU server (needs an NVIDIA driver with CUDA 12.8+, check with `nvidia-smi`; for older drivers change
the PyTorch index in `pyproject.toml` to `cu126` or `cu124`):
```bash
uv sync
source .venv/bin/activate
```

Windows: if the project folder path is long, PyTorch's CUDA DLLs fail to load ("filename too long").
Put the venv at a short path instead:
```bash
set UV_PROJECT_ENVIRONMENT=C:\venvs\msad
uv sync
```

Using several GPUs: run one machine type / modality per GPU, e.g.
`CUDA_VISIBLE_DEVICES=1 python -m msad.run --machine valve --model ae`.

## Data

**DCASE 2024 Task 2 development set** (sound). 7 machine types (bearing, fan, gearbox, slider, valve, ToyCar,
ToyTrain), 10-second clips at 16 kHz. Per machine: 1000 normal training clips (990 source domain + 10 target domain,
i.e. changed operating conditions) and 200 labelled test clips. License: **CC BY-NC-SA 4.0 (non-commercial)**.
```bash
python scripts/download_dcase2024.py --machines all          # ~2.2 GB
```

**MaFaulDa** (sound + vibration, [UFRJ](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html)). A SpectraQuest
machinery fault simulator with two 3-axis accelerometers (underhang and overhang bearing) and a microphone, recorded
together at 50 kHz, 5 s per file. 49 normal recordings (737-3686 rpm) and 1902 faulty ones: imbalance, horizontal
and vertical misalignment, and cage / outer race / ball faults on both bearings. Cite the authors when publishing.
```bash
python scripts/download_mafaulda.py                          # 12.9 GB zipped, ~35 GB unzipped
python scripts/download_mafaulda.py --classes normal imbalance   # a subset
```
How it is split (`configs/mafaulda.yaml`): 70% of normal files for training, 30% for testing, plus
`files_per_fault` (default 20) faulty files per fault type. Files are cut into 1 s clips; the split is by file, so
no recording appears in both train and test. With only ~15 normal test files, AUCs are noisy: compare models on
several seeds (`--set seed=1`) before drawing conclusions.

### Adding a dataset without code

Put the files in this layout (sub-folders deeper down are fine), copy `configs/folder.yaml` to
`configs/<name>.yaml` and set `data_root`:

```
<data_root>/<machine>/normal/...              normal recordings; split into train/test by file (test_fraction)
<data_root>/<machine>/anomaly/<fault>/...     faulty recordings, test only; folder name = fault type
```
If the dataset has an official split, use `train/normal/`, `test/normal/` and `test/anomaly/<fault>/` instead.
With one machine type, `normal/` (or `train/`) can sit directly in `<data_root>`. Symlinked files work, so you do
not have to copy large datasets.

Instead of moving files, you can list them in a manifest CSV (`folder.manifest: path/to/manifest.csv`):
```
path,label,split,fault,machine
bearing1/run_003.wav,normal,,,
bearing1/run_104.wav,anomaly,test,outer_race,
```
`path` is relative to `data_root`; `label` is `normal`/`anomaly` (or 0/1); empty `split` = automatic; `fault` and
`machine` are optional.

Files: `.wav`, `.flac`, `.ogg` (sample rate read from the file), or `.csv` / `.npy` with one column per channel
(set `folder.sample_rate`; `folder.csv_header: true` if there is a header row). Choose the channels per sensor with
`folder.sound_channels` and `folder.vibration_channels` (e.g. `[0, 1, 2]` for a 3-axis accelerometer); set both for
`--modality fusion`. `folder.segment_seconds` cuts long recordings into clips, and `folder.files_per_fault` caps
the faulty test files per fault type. Vibration keeps its own sample rate (`features.vibration.sample_rate: null`).

```bash
python -m msad.run --config configs/<name>.yaml --model ae --set seed=0
```

## Train and evaluate

```bash
# DCASE, sound
python -m msad.run --machine fan --model gmm knn ae
python -m msad.run --model ae                                # all machine types

# MaFaulDa: each sensor alone and combined
python -m msad.run --config configs/mafaulda.yaml --modality sound vibration fusion --model gmm knn ae
python -m msad.run --config configs/mafaulda.yaml --modality fusion --model ae --set fusion.method=max
```
Outputs go to `outputs/<machine>/<modality>/<model>/` (model, threshold + config in `meta.json`, per-clip scores,
training history). Fusion runs also save each sensor's standardised score (`z_sound`, `z_vibration`) per clip, to
see which sensor caught a fault. Every run appends a row to `outputs/results.csv`.

Score new recordings with a trained model (same file format as its training data):
```bash
python -m msad.predict --model-dir outputs/fan/sound/ae path/to/*.wav
python -m msad.predict --model-dir outputs/mafaulda/fusion/ae path/to/*.csv
```

No data yet? Synthetic datasets test the whole pipeline in under a minute:
```bash
python scripts/make_synthetic.py dcase
python -m msad.run --machine synth --model gmm knn ae --set data_root=data/synthetic ae.epochs=5
python scripts/make_synthetic.py mafaulda
python -m msad.run --config configs/mafaulda.yaml --modality sound vibration fusion --model knn ae --set data_root=data/synthetic_mafaulda ae.epochs=5
python scripts/make_synthetic.py folder
python -m msad.run --config configs/folder.yaml --model gmm knn ae --set data_root=data/synthetic_folder ae.epochs=5
```

Unit tests: `python -m pytest` (after `uv sync`, which installs pytest).

## Features

Both sensors use log-mel spectrograms, so every model works on either one.

| | Sound | Vibration |
|---|---|---|
| Sample rate | 16 kHz | 50 kHz |
| Window / hop | 1024 / 512 | 8192 (~6 Hz bins) / 2048 |
| Bands | 128 mel | 64 mel per axis, up to 10 kHz, 6 axes stacked |

Mel bands are nearly linear below ~1 kHz, which keeps shaft harmonics (imbalance at 1x rotation, misalignment at 2x)
apart, and wider above, where bearing resonances sit.

## Models

| Name | Input | Anomaly score |
|---|---|---|
| `gmm` | mean + std of each band over the clip | negative log-likelihood under a Gaussian mixture |
| `knn` | same clip statistics | distance to the nearest normal training clip |
| `ae`  | 5 stacked spectrogram frames | autoencoder reconstruction error (DCASE baseline) |

`--modality fusion` trains one detector per sensor, standardises each sensor's scores on normal training clips and
combines them: `mean` (default, both sensors must agree) or `max` (either sensor is enough).

## Metrics

- **AUC (source / target)**, DCASE only: how well scores separate normal from anomalous clips. Each domain's normal
  clips are compared against all anomalies. Target AUC shows robustness when operating conditions change.
- **AUC (all)**: same over all test clips.
- **pAUC**: AUC limited to false-positive rate ≤ 0.1, the low-false-alarm range that matters in a factory.
- **hmean**: harmonic mean of AUC(source), AUC(target), pAUC; for datasets without domains (MaFaulDa) of AUC(all) and
  pAUC. Main number for comparing models.
- **auc_fault/<type>** (in `results.json`): normal vs. one fault type, e.g. which faults vibration catches but sound misses.
- **precision / recall / F1**: at the decision threshold, the 90th percentile of scores on normal training clips
  (`threshold_percentile`). Raise it for fewer false alarms.

## Layout

```
configs/default.yaml           DCASE / shared settings
configs/mafaulda.yaml          MaFaulDa settings (inherits default.yaml)
configs/folder.yaml            template for any other dataset (folder layout or manifest)
scripts/download_dcase2024.py
scripts/download_mafaulda.py
scripts/make_synthetic.py
src/msad/data.py               datasets (DCASE, MaFaulDa, generic folder): clips, labels, fault types, signals
src/msad/features.py           log-mel features (cached in data/cache), frame stacking, clip stats
src/msad/models.py             gmm / knn / autoencoder detectors, score-level fusion
src/msad/metrics.py            AUC, pAUC, hmean, per-fault AUC, F1
src/msad/run.py                train + evaluate CLI
src/msad/predict.py            score new files
tests/                         unit tests (python -m pytest)
```

## Working together

Project management lives in Notion (Tasks board, Experiments leaderboard, Meetings & Decisions).

1. Pick a task in Notion, set it to *In progress*.
2. Branch from `main`: `git switch -c <ID>-short-name` (e.g. `MSAD-6-parallel-features`).
3. Commit, push, open a pull request. Fill in the PR template, paste the PR link into the task's *Branch / PR* field and
   set the task to *In review*.
4. The other person reviews; merge with squash, then set the task to *Done*.
5. Log training runs worth keeping in the Notion *Experiments* database (metrics from `outputs/results.csv`).

Never commit `data/` or `outputs/` (already in `.gitignore`); datasets are re-downloaded with the scripts.

## Next steps

1. Vibration indicators used in industry: envelope spectrum, kurtosis, crest factor; bearing fault frequencies from
   the tachometer (BPFO = 2.998x, BPFI = 5.002x rotation on MaFaulDa).
2. Stronger models: self-supervised CNN (MobileFaceNet/ResNet + ArcFace), pretrained audio embeddings (BEATs, PANNs,
   EAT) + kNN, a 1D CNN fault-type classifier for vibration.
3. Learned fusion (weights per sensor, or one model on both spectrograms) instead of fixed mean / max.
4. Your own machines: record normal sound and vibration together, put them in the MaFaulDa-style layout, train, and set
   the threshold.
