"""Anomaly detectors, trained on normal clips only; one score per clip (higher = more anomalous).
They work on any (n_bands, T) feature matrix, so the same model serves sound and vibration.

- AutoencoderDetector: DCASE-style dense autoencoder on stacked log-mel frames; score = reconstruction error
- FusionDetector: one autoencoder per modality (sound, vibration); scores are standardised on normal training
  clips and combined (score-level fusion)

After fit(), `train_scores_` holds scores of the normal training clips, used to set the decision threshold.
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .features import stack_frames


class Detector:
    def fit(self, specs: list[np.ndarray]) -> "Detector":
        raise NotImplementedError

    def score(self, specs: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Detector":
        return joblib.load(path)


class _DenseAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_layers, bottleneck_dim):
        super().__init__()

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]

        enc, dec = [], []
        dims = [input_dim] + [hidden_dim] * n_hidden_layers
        for i, o in zip(dims[:-1], dims[1:]):
            enc += block(i, o)
        enc += block(hidden_dim, bottleneck_dim)
        dec += block(bottleneck_dim, hidden_dim)
        for _ in range(n_hidden_layers - 1):
            dec += block(hidden_dim, hidden_dim)
        dec.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*enc, *dec)

    def forward(self, x):
        return self.net(x)


class AutoencoderDetector(Detector):
    def __init__(self, n_frames=5, hidden_dim=128, n_hidden_layers=4, bottleneck_dim=8,
                 epochs=100, batch_size=512, lr=1e-3, val_fraction=0.1, seed=0, device=None):
        self.n_frames = n_frames
        self.arch = dict(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, bottleneck_dim=bottleneck_dim)
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.val_fraction, self.seed = val_fraction, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: list[dict] = []

    def _vectors(self, spec):
        return (stack_frames(spec, self.n_frames) - self.mean) / self.std

    def fit(self, specs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(specs))
        n_val = int(len(specs) * self.val_fraction)
        val_idx, train_idx = order[:n_val], order[n_val:]

        raw = np.concatenate([stack_frames(specs[i], self.n_frames) for i in train_idx])
        self.mean, self.std = raw.mean(axis=0), raw.std(axis=0) + 1e-6
        x_train = torch.from_numpy((raw - self.mean) / self.std).to(self.device)
        del raw
        x_val = (torch.from_numpy(np.concatenate([self._vectors(specs[i]) for i in val_idx])).to(self.device)
                 if n_val else None)

        self.model = _DenseAE(x_train.shape[1], **self.arch).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_val, best_state = float("inf"), None

        bar = tqdm(range(self.epochs), desc=f"AE on {self.device}", leave=False)
        for epoch in bar:
            self.model.train()
            perm = torch.randperm(len(x_train), device=self.device)
            total = 0.0
            for start in range(0, len(x_train), self.batch_size):
                batch = x_train[perm[start : start + self.batch_size]]
                if len(batch) < 2:  # BatchNorm needs >1 sample
                    continue
                loss = loss_fn(self.model(batch), batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            row = {"epoch": epoch + 1, "train_loss": total / len(x_train)}
            if x_val is not None:
                row["val_loss"] = self._mse(x_val).mean().item()
                if row["val_loss"] < best_val:
                    best_val = row["val_loss"]
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.history.append(row)
            bar.set_postfix({k: f"{v:.4f}" for k, v in row.items() if k != "epoch"})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.train_scores_ = self.score(specs)
        return self

    @torch.no_grad()
    def _mse(self, x):
        self.model.eval()
        return torch.cat([((self.model(x[i : i + 8192]) - x[i : i + 8192]) ** 2).mean(dim=1)
                          for i in range(0, len(x), 8192)])

    def score(self, specs):
        return np.array([
            self._mse(torch.from_numpy(self._vectors(s)).to(self.device)).mean().item() for s in specs
        ])

    def save(self, path):
        self.model.cpu()
        device, self.device = self.device, "cpu"
        super().save(path)
        self.device = device
        self.model.to(device)


class FusionDetector(Detector):
    """Score-level fusion of per-modality detectors. fit/score take {modality: list of feature matrices}."""

    def __init__(self, detectors: dict[str, Detector], method: str = "mean"):
        if method not in ("mean", "max"):
            raise ValueError(f"fusion method must be 'mean' or 'max', got {method!r}")
        self.detectors, self.method = detectors, method

    def fit(self, feats):
        self.norm = {}
        for m, det in self.detectors.items():
            det.fit(feats[m])
            s = det.train_scores_
            self.norm[m] = (float(s.mean()), float(s.std()) + 1e-9)
        self.train_scores_ = self._combine({m: d.train_scores_ for m, d in self.detectors.items()})
        return self

    def score(self, feats):
        return self._combine({m: d.score(feats[m]) for m, d in self.detectors.items()})

    def modality_scores(self, feats) -> dict[str, np.ndarray]:
        """Standardised score of each modality, to see which sensor drove a decision."""
        return {m: (d.score(feats[m]) - self.norm[m][0]) / self.norm[m][1] for m, d in self.detectors.items()}

    def _combine(self, scores):
        z = [(scores[m] - mu) / sd for m, (mu, sd) in self.norm.items()]
        # mean: both sensors must agree; max: an anomaly seen by either sensor is enough
        return np.max(z, axis=0) if self.method == "max" else np.mean(z, axis=0)

    def save(self, path):
        for det in self.detectors.values():
            if isinstance(det, AutoencoderDetector):
                det.model.cpu()
                det.device = "cpu"
        super().save(path)


def build(cfg: dict, modality: str = "sound") -> Detector:
    if modality == "fusion":
        return FusionDetector({m: build(cfg, m) for m in ("sound", "vibration")},
                              method=cfg.get("fusion", {}).get("method", "mean"))
    return AutoencoderDetector(n_frames=cfg["features"][modality]["n_frames"], **cfg["ae"], seed=cfg.get("seed", 0))
