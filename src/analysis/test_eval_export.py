"""Phase 12 step 1 (Windows .venv, needs pandas): export per-tier TEST features for the frozen evaluation.

    python -m src.analysis.test_eval_export --config configs/phase12/test_eval.yaml

Reads `data/processed/test.pkl` (the Phase 2 matrix, transformed with the TRAIN-fitted preprocessor), verifies the pinned hashes,
writes `X_test` (float64, the same dtype and tier column order as the validation arrays in the FHE handoffs) and the
TransactionIDs per tier, and the labels into a SEPARATE file that no other step opens before the frozen run. No model, threshold or
statistic is computed here. See docs/test_evaluation_protocol.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned_inputs(config: dict) -> dict[str, str]:
    """Every pinned input must match its recorded hash, else stop before reading any data."""
    seen = {}
    inputs = config["inputs"]
    for key in ("test_pkl", "tiers_json", "split_boundaries"):
        actual = sha256_path(PROJECT_ROOT / inputs[key])
        if actual != inputs[f"{key}_sha256"]:
            raise SystemExit(f"{key}: sha256 {actual} does not match the pinned {inputs[key + '_sha256']} -- refusing to export")
        seen[key] = actual
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase12/test_eval.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    pinned = verify_pinned_inputs(config)

    import pandas as pd

    out_dir = PROJECT_ROOT / config["export_dir"]
    if (out_dir / "export.json").exists():
        raise SystemExit(f"{out_dir / 'export.json'} already exists: the export is written once; delete it deliberately to redo")
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    tiers = json.loads((PROJECT_ROOT / config["inputs"]["tiers_json"]).read_text())["tiers"]
    frame = pd.read_pickle(PROJECT_ROOT / config["inputs"]["test_pkl"])
    if not frame.index.is_monotonic_increasing:
        raise SystemExit("test.pkl is not ordered by TransactionID/time; the block bootstrap relies on time order")
    ids = frame.index.to_numpy()
    files = {}
    for tier, columns in tiers.items():
        path = out_dir / f"X_test_{tier}.npz"
        np.savez(path, X_test=frame[columns].to_numpy(dtype=np.float64), transaction_ids=ids)
        files[tier] = {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_path(path), "shape": [int(len(frame)), len(columns)]}
    y_path = out_dir / "labels" / "y_test.npy"
    np.save(y_path, frame["isFraud"].to_numpy(dtype=np.int64))
    del frame
    manifest = {"pinned_inputs": pinned, "tier_files": files, "labels_file": {"path": y_path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_path(y_path)}, "n_rows": int(len(ids)),
                "note": "labels are written here and first parsed by the frozen run (src.analysis.test_evaluation --run)"}
    (out_dir / "export.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_rows", "tier_files")}, indent=2))


if __name__ == "__main__":
    main()
