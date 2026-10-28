"""Score new recordings with a trained detector. Uses the dataset format and settings saved with the model
(.wav for DCASE-style sound models, MaFaulDa-style .csv for vibration / fusion models). Files that are split into
segments during training are scored per segment; the file score is the mean segment score.

    python -m msad.predict --model-dir outputs/fan/sound recordings/*.wav
    python -m msad.predict --model-dir outputs/mafaulda/fusion recordings/*.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import get_dataset
from .models import Detector
from .run import load_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    cfg, modality, threshold = meta["config"], meta["modality"], meta["threshold"]
    model = Detector.load(model_dir / "model.joblib")
    dataset = get_dataset(cfg)

    for f in args.files:
        clips = dataset.file_clips(f)
        score = float(np.mean(model.score(load_features(dataset, clips, modality, cfg, Path(f).name))))
        print(f"{'ANOMALY' if score > threshold else 'normal ':7}  score={score:.4f}  {f}")


if __name__ == "__main__":
    main()
