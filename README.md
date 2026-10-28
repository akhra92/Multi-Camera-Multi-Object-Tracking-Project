# Machine Fault Detection from Sound and Vibration

Unsupervised fault detection from **sound** (microphones) and **vibration** (accelerometers), alone or combined.
An autoencoder is trained on normal operation only and flags recordings that sound or vibrate differently.

## Setup

```bash
uv sync
source .venv/bin/activate
```
GPU: needs an NVIDIA driver with CUDA 12.8+ (`nvidia-smi`); for older drivers set the PyTorch index in
`pyproject.toml` to `cu126` or `cu124`. On Windows with a long project path, put the venv at a short path first:
`set UV_PROJECT_ENVIRONMENT=C:\venvs\msad`.

## Data

To be added after dataset selection.

### Adding a dataset

Copy `configs/folder.yaml` to `configs/<name>.yaml`, set `data_root` and the `folder:` section, and arrange the files:
```
<data_root>/<machine>/normal/...            normal recordings, split into train/test by file
<data_root>/<machine>/anomaly/<fault>/...   faulty recordings (test only); folder name = fault type
```
With an official split use `train/normal/`, `test/normal/`, `test/anomaly/<fault>/`. Or list the files in a manifest
CSV (`folder.manifest`) with columns `path,label[,split,fault,machine,domain]`.

Files: `.wav` / `.flac` / `.ogg`, or `.csv` / `.npy` with one column per channel (set `folder.sample_rate`).
`folder.sound_channels` and `folder.vibration_channels` pick the channels per sensor (set both for fusion).

## Train and evaluate

```bash
python -m msad.run --config configs/<name>.yaml --set seed=0
python -m msad.run --config configs/<name>.yaml --modality sound vibration fusion
```
`--machine` picks machine types (default: all), `--set key=value` overrides any config value. Results go to
`outputs/<machine>/<modality>/` (model, threshold, per-clip scores) and one row per run to `outputs/results.csv`.

Score new recordings:
```bash
python -m msad.predict --model-dir outputs/<machine>/<modality> path/to/*.wav
```

Smoke test without data: `python scripts/make_synthetic.py folder`, then
`python -m msad.run --config configs/folder.yaml --set data_root=data/synthetic_folder ae.epochs=5`.
Unit tests: `python -m pytest`.

## Model

Dense autoencoder (DCASE baseline) on 5 stacked log-mel frames; anomaly score = reconstruction error. Settings in
the `ae:` section of `configs/default.yaml`. Vibration uses log-mel spectrograms at its own sample rate.
`--modality fusion` trains one autoencoder per sensor and combines their standardised scores (`fusion.method`:
`mean` or `max`).

## Metrics

| Metric | Meaning |
|---|---|
| AUC source / target | normal vs. anomalous clips per domain, if the dataset has domains |
| AUC all | all test clips |
| pAUC | AUC at false-positive rate ≤ 0.1 |
| hmean | harmonic mean of the AUCs and pAUC; main comparison number |
| auc_fault/\<type\> | normal vs. one fault type (in `results.json`) |
| precision / recall / F1 | at the 90th percentile of normal training scores (`threshold_percentile`) |

## Layout

```
configs/default.yaml     shared settings (features, autoencoder, metrics)
configs/folder.yaml      template for a new dataset
scripts/make_synthetic.py
src/msad/data.py         dataset loaders
src/msad/features.py     log-mel features (cached in data/cache)
src/msad/models.py       autoencoder, score-level fusion
src/msad/metrics.py      AUC, pAUC, hmean, per-fault AUC, F1
src/msad/run.py          train + evaluate
src/msad/predict.py      score new files
tests/
```

## Working together

Tasks, datasets and experiments are tracked in Notion. Branch from `main` (`git switch -c <ID>-short-name`), open
a pull request, and the other person reviews before merging. Log runs worth keeping in Notion *Experiments*.
Never commit `data/` or `outputs/`.
