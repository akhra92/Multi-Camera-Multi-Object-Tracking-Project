"""Train the autoencoder on normal clips and evaluate it on the labelled test set.

    python -m msad.run --machine fan                                         # DCASE 2024, sound
    python -m msad.run --config configs/mafaulda.yaml --modality sound vibration fusion
    python -m msad.run --machine fan --set ae.epochs=20 seed=1
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .data import MODALITIES, get_dataset
from .features import FeatureCache
from .metrics import evaluate
from .models import build


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: list[str] = ()) -> dict:
    """Load YAML config; a config may name an `inherit:` file (relative to itself) that it overrides."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("inherit")), cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def modalities_of(modality: str) -> list[str]:
    return list(MODALITIES) if modality == "fusion" else [modality]


def load_features(dataset, clips, modality: str, cfg: dict, desc: str):
    """{modality: [feature matrix per clip]} for fusion, otherwise a plain list."""
    feats = {m: FeatureCache(cfg["cache_dir"], dataset, m, cfg).get_many(clips, desc=f"{desc} {m}")
             for m in modalities_of(modality)}
    return feats if modality == "fusion" else feats[modality]


def run_one(dataset, machine: str, modality: str, cfg: dict) -> dict:
    out_dir = Path(cfg["output_dir"]) / machine / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    train = [c for c in dataset.clips(machine, "train") if c.label in (0, None)]
    test = dataset.clips(machine, "test")
    x_train = load_features(dataset, train, modality, cfg, f"{machine} train")
    x_test = load_features(dataset, test, modality, cfg, f"{machine} test")

    t0 = time.time()
    model = build(cfg, modality).fit(x_train)
    fit_seconds = time.time() - t0

    threshold = float(np.percentile(model.train_scores_, cfg["threshold_percentile"]))
    test_scores = model.score(x_test)
    per_modality = model.modality_scores(x_test) if modality == "fusion" else {}

    model.save(out_dir / "model.joblib")
    meta = {"threshold": threshold, "modality": modality, "machine": machine, "config": cfg}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    with open(out_dir / "test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "segment", "domain", "fault", "label", "score", "predicted_anomaly",
                    *[f"z_{m}" for m in per_modality]])
        for i, (c, s) in enumerate(zip(test, test_scores)):
            w.writerow([c.path.name, c.segment, c.domain, c.fault, c.label, f"{s:.6f}", int(s > threshold),
                        *[f"{v[i]:.4f}" for v in per_modality.values()]])
    histories = {m: d.history for m, d in getattr(model, "detectors", {"": model}).items() if getattr(d, "history", None)}
    if histories:
        (out_dir / "history.json").write_text(json.dumps(histories, indent=1))

    result = {"dataset": cfg.get("dataset", "dcase2024"), "machine": machine, "modality": modality,
              "model": "ae", "n_train": len(train), "n_test": len(test), "fit_seconds": round(fit_seconds, 1)}
    if all(c.label is not None for c in test):
        result |= evaluate(test, test_scores, threshold, cfg["max_fpr"])
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--machine", nargs="+", default=["all"], help="machine type(s) or 'all' (default)")
    parser.add_argument("--modality", nargs="+", choices=[*MODALITIES, "fusion"],
                        help="default: the config's `modality`")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    np.random.seed(cfg["seed"])
    dataset = get_dataset(cfg)
    modalities = args.modality or [cfg.get("modality", "sound")]
    for modality in modalities:
        missing = set(modalities_of(modality)) - set(dataset.modalities)
        if missing:
            parser.error(f"dataset {cfg.get('dataset')!r} has no {sorted(missing)} data (has {dataset.modalities})")
    machines = dataset.machines() if args.machine == ["all"] else args.machine
    if not machines:
        parser.error(f"no machine folders found under {cfg['data_root']}")

    results = []
    for machine in machines:
        for modality in modalities:
            r = run_one(dataset, machine, modality, cfg)
            results.append(r)
            keys = ("auc_source", "auc_target", "auc_all", "pauc", "hmean", "f1")
            metrics = "  ".join(f"{k}={r[k]:.3f}" for k in keys if k in r)
            print(f"[{machine:>9} | {modality:>9}] {metrics}  ({r['fit_seconds']}s)")

    summary = Path(cfg["output_dir"]) / "results.csv"
    fields = ["dataset", "machine", "modality", "model", "auc_source", "auc_target", "auc_all", "pauc", "hmean",
              "precision", "recall", "f1", "n_train", "n_test", "fit_seconds"]
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(results)
    print(f"\nAppended {len(results)} result row(s) to {summary}")


if __name__ == "__main__":
    main()
