"""Log-mel spectrogram features for sound and vibration, with an on-disk cache so each clip is processed once.

Vibration uses the same mel filterbank as sound but at the accelerometer sample rate and a lower fmax: mel bands are
nearly linear below ~1 kHz (fine resolution around shaft harmonics) and wider above (bearing resonances).
Multi-channel signals (e.g. 3-axis accelerometers) are stacked along the band axis.
"""
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def spectrogram(signal: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """(channels, samples) -> (channels * n_mels, n_time) log-mel spectrogram in dB, as in the DCASE baseline."""
    target_sr = cfg.get("sample_rate") or sr  # null sample_rate = keep the signal's own rate
    fmax = min(cfg.get("fmax") or target_sr / 2, target_sr / 2)
    specs = []
    for y in np.atleast_2d(signal):
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=target_sr,
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
            power=cfg["power"],
            fmin=cfg.get("fmin", 0.0),
            fmax=fmax,
        )
        specs.append(20.0 / cfg["power"] * np.log10(mel + np.finfo(np.float32).eps))
    return np.concatenate(specs).astype(np.float32)


class FeatureCache:
    def __init__(self, cache_dir: str | Path, dataset, modality: str, cfg: dict):
        """`cfg` is the full config; the cache key covers everything that changes the features."""
        self.dataset, self.modality = dataset, modality
        self.fcfg = cfg["features"][modality]
        key = {"modality": modality, "features": self.fcfg, "dataset": cfg.get("dataset"),
               "dataset_cfg": cfg.get(cfg.get("dataset", ""), {})}
        tag = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
        self.dir = Path(cache_dir) / f"{modality}_{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, clip) -> np.ndarray:
        ident = f"{Path(clip.path).resolve()}|{clip.segment}"
        cached = self.dir / (hashlib.md5(ident.encode()).hexdigest() + ".npy")
        if cached.exists():
            return np.load(cached)
        spec = spectrogram(*self.dataset.load(clip, self.modality), self.fcfg)
        tmp = cached.with_suffix(".tmp.npy")
        np.save(tmp, spec)
        tmp.replace(cached)
        return spec

    def get_many(self, clips, desc: str = "features") -> list[np.ndarray]:
        return [self.get(c) for c in tqdm(clips, desc=desc, leave=False)]


def stack_frames(spec: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_bands, T) -> (T - n_frames + 1, n_bands * n_frames) context windows for the autoencoder."""
    n_bands, t = spec.shape
    n_vectors = t - n_frames + 1
    if n_vectors < 1:
        raise ValueError(f"clip too short: {t} frames < n_frames={n_frames}")
    out = np.empty((n_vectors, n_bands * n_frames), dtype=np.float32)
    for i in range(n_frames):
        out[:, n_bands * i : n_bands * (i + 1)] = spec[:, i : i + n_vectors].T
    return out

