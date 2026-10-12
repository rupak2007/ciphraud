"""Phase 12 frozen test-partition evaluation (WSL/Linux FHE venv). DRAFT: prepared, not executed.

    python -m src.analysis.test_evaluation --config configs/phase12/test_eval.yaml --freeze   # stage A: labels stay closed
    python -m src.analysis.test_evaluation --config configs/phase12/test_eval.yaml --run      # stage B: ONE run, opens the labels

Stage A rebuilds each of the 18 accepted models with the existing loaders (no retraining), proves it reproduces the committed
VALIDATION numbers exactly, and writes `freeze.json` (hashes, thresholds, fingerprints). Stage B refuses to start without an intact
`freeze.json` or after a previous run, re-verifies every model against it, computes all test probabilities (labels still closed),
then writes `run_started.json`, opens the labels and computes the metrics and block-bootstrap intervals. Scoring is done in clear
mode (`fhe="disable"`): no keys, no ciphertexts, no encrypted execution. Nothing is tuned; thresholds come from validation labels.
See docs/test_evaluation_protocol.md.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXACT = 1e-12  # reproduced validation PR-AUC must equal the committed value to this absolute tolerance
THRESHOLD_TOL = 1e-6  # stored (Phase 4/6 committed) threshold vs the one recomputed from validation: LR differs by ~1e-7 (float noise); XGBoost is identical


CODE_FILES = (
    "src/analysis/test_evaluation.py", "src/analysis/test_eval_export.py", "src/benchmark/run.py", "src/benchmark/phase8_grid.py", "src/benchmark/phase9_grid.py",
    "src/fhe/handoff.py", "src/fhe/xgb_poc.py", "src/fhe/compile/linear.py", "src/fhe/compile/tree.py", "src/fhe/compile/mlp.py", "src/fhe/validate/correctness.py", "src/train/metrics.py",
)


def code_hashes() -> dict[str, str]:
    """sha256 of every source file that builds, scores or measures a model here; frozen, and re-checked before the run."""
    return {f: sha256_path(PROJECT_ROOT / f) for f in CODE_FILES}


class FrozenProtocolError(Exception):
    """Raised whenever a protocol safeguard is violated; the run stops before any label is opened or any result is written."""


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- guards (pure, tested)

def verify_pinned_inputs(config: dict[str, Any]) -> dict[str, str]:
    inputs = config["inputs"]
    seen: dict[str, str] = {}
    for key in ("test_pkl", "tiers_json", "split_boundaries"):
        actual = sha256_path(PROJECT_ROOT / inputs[key])
        if actual != inputs[f"{key}_sha256"]:
            raise FrozenProtocolError(f"{key}: sha256 {actual} != pinned {inputs[key + '_sha256']}")
        seen[key] = actual
    return seen


def verify_export(config: dict[str, Any]) -> dict[str, Any]:
    """The exported arrays must be exactly the ones recorded at export time. Reads bytes for hashing only; parses no labels."""
    export_dir = PROJECT_ROOT / config["export_dir"]
    manifest_path = export_dir / "export.json"
    if not manifest_path.exists():
        raise FrozenProtocolError(f"{manifest_path} missing: run src.analysis.test_eval_export first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for tier, rec in manifest["tier_files"].items():
        if sha256_path(PROJECT_ROOT / rec["path"]) != rec["sha256"]:
            raise FrozenProtocolError(f"exported array for {tier} was modified after export")
    if sha256_path(PROJECT_ROOT / manifest["labels_file"]["path"]) != manifest["labels_file"]["sha256"]:
        raise FrozenProtocolError("the exported label file was modified after export")
    return manifest


def require_freeze(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "freeze.json"
    if not path.exists():
        raise FrozenProtocolError(f"{path} missing: stage A (--freeze) must complete before the run")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_no_previous_run(results_dir: Path) -> None:
    for name in ("run_started.json", "run_complete.json"):
        if (results_dir / name).exists():
            raise FrozenProtocolError(f"{results_dir / name} exists: the test evaluation runs exactly once and is never repeated or resumed after labels were opened")


def compare_with_freeze(frozen: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    """Every field that identifies the model and its validation behavior must be identical to the frozen record."""
    problems = []
    for key in ("fingerprints", "thresholds", "validation_reproduction"):
        if frozen.get(key) != rebuilt.get(key):
            problems.append(f"{frozen.get('label')}: {key} differs from freeze.json")
    return problems


# ---------------------------------------------------------------------------------------------------- bootstrap (pure, tested)

def block_edges(n_rows: int, n_blocks: int) -> np.ndarray:
    return np.linspace(0, n_rows, n_blocks + 1).astype(np.int64)


def resample_indices(edges: np.ndarray, seed: int, r: int) -> np.ndarray:
    """One time-block resample: draw len(blocks) contiguous blocks with replacement (deterministic in (seed, r))."""
    rng = np.random.default_rng([seed, r])
    n_blocks = len(edges) - 1
    chosen = rng.integers(0, n_blocks, size=n_blocks)
    return np.concatenate([np.arange(edges[b], edges[b + 1]) for b in chosen])


def _chunk(y: np.ndarray, matrix: np.ndarray, edges: np.ndarray, seed: int, rs: list[int]) -> np.ndarray:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = np.full((len(rs), matrix.shape[0], 2), np.nan)
    for i, r in enumerate(rs):
        idx = resample_indices(edges, seed, r)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        for j in range(matrix.shape[0]):
            s = matrix[j, idx]
            out[i, j, 0] = average_precision_score(yy, s)
            out[i, j, 1] = roc_auc_score(yy, s)
    return out


def bootstrap_distribution(y: np.ndarray, scores: dict[str, np.ndarray], n_blocks: int, n_resamples: int, seed: int, n_jobs: int = 1) -> dict[str, dict[str, np.ndarray]]:
    """PR-AUC and ROC-AUC of every score vector on the SAME resamples (paired), so differences between vectors are valid."""
    names = list(scores)
    matrix = np.vstack([np.asarray(scores[n], dtype=np.float64) for n in names])
    edges = block_edges(len(y), n_blocks)
    chunks = [list(range(s, min(s + 25, n_resamples))) for s in range(0, n_resamples, 25)]
    if n_jobs > 1:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=n_jobs)(delayed(_chunk)(y, matrix, edges, seed, c) for c in chunks)
    else:
        parts = [_chunk(y, matrix, edges, seed, c) for c in chunks]
    stacked = np.concatenate(parts, axis=0)  # [R, vectors, 2]
    return {n: {"pr_auc": stacked[:, j, 0], "roc_auc": stacked[:, j, 1]} for j, n in enumerate(names)}


def percentile_ci(samples: np.ndarray, confidence: float) -> list[float]:
    lo, hi = np.nanpercentile(samples, [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
    return [float(lo), float(hi)]


def contrast_ci(dist: dict[str, dict[str, np.ndarray]], a: str, b: str, metric: str, confidence: float) -> dict[str, Any]:
    diff = dist[a][metric] - dist[b][metric]
    return {"a": a, "b": b, "metric": metric, "mean_resampled_diff": float(np.nanmean(diff)), "ci": percentile_ci(diff, confidence), "fraction_a_greater": float(np.nanmean(diff > 0))}


# ---------------------------------------------------------------------------------------------------- configuration bookkeeping

def tier_of(label: str) -> str:
    m = re.search(r"top_?(\d+)", label)
    if not m:
        raise FrozenProtocolError(f"cannot read the tier from label {label!r}")
    return f"top_{m.group(1)}"


def list_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    import yaml

    out = []
    for key in ("phase8_config", "phase9_config"):
        grid = yaml.safe_load((PROJECT_ROOT / config["inputs"][key]).read_text(encoding="utf-8"))
        for e in grid["configurations"]:
            mtype = e.get("model_type", "mlp")
            out.append({"label": e["label"], "model_type": mtype, "tier": e.get("tier") or tier_of(e["label"]), "n_bits": int(e["n_bits"]), "grid": key, "entry": e, "grid_config": grid})
    return out


def _recorded_val(entry: dict[str, Any]) -> dict[str, float]:
    sub = "phase8_research" if entry["model_type"] in ("lr", "xgboost") else "phase9_mlp"
    m = json.loads((PROJECT_ROOT / "results" / sub / entry["label"] / "metrics.json").read_text(encoding="utf-8"))
    acc = m["accuracy"]
    return {"quantized_pr_auc": acc["quantized_full_metrics"]["pr_auc"], "float_pr_auc": acc["float_full_metrics"]["pr_auc"]}


# ---------------------------------------------------------------------------------------------------- model adapters (heavy imports)

class Adapter:
    """Builds one accepted model with the existing loader and exposes: the frozen record (validation only) and test scoring."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self.entry = entry
        self.record: dict[str, Any] = {}

    def score_test(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:  # (quantized prob, float prob or None)
        raise NotImplementedError


def _val_pr_auc(y: np.ndarray, prob: np.ndarray) -> float:
    from src.train.metrics import threshold_independent_metrics

    return float(threshold_independent_metrics(y, prob)["pr_auc"])


class LrXgbAdapter(Adapter):
    def __init__(self, entry: dict[str, Any], config: dict[str, Any]) -> None:
        super().__init__(entry)
        from src.benchmark.phase8_grid import _load_configuration
        from src.train.metrics import select_threshold

        loaded = _load_configuration(entry["entry"])
        self.loaded = loaded
        src = loaded["source_config"]
        y_val = loaded["y_val"]
        if entry["model_type"] == "lr":
            manifest = json.loads((PROJECT_ROOT / src["output"]["handoff_manifest"]).read_text())
            self.params_path = PROJECT_ROOT / manifest["params_path"]
            from src.fhe.handoff import standardize_features

            self.params = json.loads(self.params_path.read_text())
            self._standardize = standardize_features
            fingerprints = {"params_sha256": sha256_path(self.params_path), "handoff_npz_sha256": manifest["npz_sha256"]}
            q_val = loaded["cml_model"].predict_proba(loaded["X_val_model"], fhe="disable")[:, 1]
        else:
            manifest = json.loads((PROJECT_ROOT / src["output"]["dir"] / entry["tier"] / "handoff_manifest.json").read_text())
            fingerprints = {"booster_sha256": sha256_path(PROJECT_ROOT / manifest["booster_path"]), "handoff_npz_sha256": manifest["npz_sha256"]}
            # Phase 8's own path (src/benchmark/phase8_grid.py::_accuracy_xgboost): chunked integer evaluation. A single predict_proba on
            # all 105,088 rows allocates a (n_trees*leaves x rows) int64 matrix of ~18 GiB (found by the first freeze attempt).
            self.chunk = int(entry["grid_config"]["disable_chunk_size"])
            q_val = self._xgb_probabilities(loaded["X_val_model"])
        recorded = _recorded_val(entry)
        reproduced = {"quantized_pr_auc": _val_pr_auc(y_val, q_val), "float_pr_auc": _val_pr_auc(y_val, loaded["reference_val_prob"])}
        thr = float(loaded["threshold"])
        own = float(select_threshold(y_val, loaded["reference_val_prob"])["selected_threshold"])
        self.record = {
            "label": entry["label"], "model_type": entry["model_type"], "tier": entry["tier"], "n_bits": entry["n_bits"],
            "fingerprints": fingerprints,
            "thresholds": {"decision_threshold": thr, "recomputed_from_validation_float_prob": own, "matches_recomputed": bool(abs(thr - own) < THRESHOLD_TOL), "abs_difference": abs(thr - own)},
            "validation_reproduction": {"recorded": recorded, "reproduced": reproduced,
                                        "equal": bool(all(abs(recorded[k] - reproduced[k]) <= EXACT for k in recorded))},
        }

    def _xgb_probabilities(self, X: np.ndarray) -> np.ndarray:
        from src.fhe.xgb_poc import _probabilities, disable_integers

        cml = self.loaded["cml_model"]
        return np.asarray(_probabilities(cml, disable_integers(cml, cml.quantize_input(X), self.chunk)))

    def score_test(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cml = self.loaded["cml_model"]
        if self.entry["model_type"] == "lr":
            q = cml.predict_proba(self._standardize(self.params, X_test), fhe="disable")[:, 1]
            f = self.loaded["plaintext_predict"](X_test)
        else:
            q = self._xgb_probabilities(X_test)
            f = self.loaded["clf"].predict_proba(X_test)[:, 1]
        return np.asarray(q), np.asarray(f)


class MlpAdapter(Adapter):
    def __init__(self, entry: dict[str, Any], config: dict[str, Any]) -> None:
        super().__init__(entry)
        from src.benchmark import phase9_grid as g
        from src.fhe.compile import mlp
        from src.train.metrics import select_threshold

        if config.get("mlp_float_twin", "off") != "off":
            raise FrozenProtocolError("mlp_float_twin != 'off' needs separate approval and is not implemented in this draft (the twin's weights were never saved)")
        grid, tier, bits, seed = entry["grid_config"], entry["tier"], entry["n_bits"], int(entry["grid_config"]["seed"])
        data = g.load_tier_data(grid, tier)
        data.pop("X_train", None)
        qat_dir = g._checkpoint_root(grid) / "qat" / f"{tier}_bits{bits}_seed{seed}"
        twin_dir = g._checkpoint_root(grid) / "float_twin" / f"{tier}_seed{seed}"
        self.model = mlp.load_model((qat_dir / "model.json").read_text())
        self.mlp = mlp
        self.scaler = data["scaler"]
        from src.fhe.handoff import standardize_features

        self._standardize = standardize_features
        y_val = data["y_val"]
        logits = mlp.clear_quantized_logits(self.model, data["X_val"])
        saved = np.load(qat_dir / "val_logits.npy")
        q_prob, f_prob = mlp.fraud_probability(logits), mlp.fraud_probability(np.load(twin_dir / "val_logits.npy"))
        recorded = _recorded_val(entry)
        reproduced = {"quantized_pr_auc": _val_pr_auc(y_val, q_prob), "float_pr_auc": _val_pr_auc(y_val, f_prob)}
        self.record = {
            "label": entry["label"], "model_type": "mlp", "tier": tier, "n_bits": bits,
            "fingerprints": {"qat_model_json_sha256": sha256_path(qat_dir / "model.json"), "scaler_sha256": data["manifest"]["scaler_sha256"],
                             "float_twin_val_logits_sha256": sha256_path(twin_dir / "val_logits.npy")},
            "thresholds": {"decision_threshold": float(select_threshold(y_val, f_prob)["selected_threshold"]), "qat_own_threshold": float(select_threshold(y_val, q_prob)["selected_threshold"])},
            "validation_reproduction": {"recorded": recorded, "reproduced": reproduced, "saved_logits_reproduced_exactly": bool(np.array_equal(logits, saved)),
                                        "equal": bool(all(abs(recorded[k] - reproduced[k]) <= EXACT for k in recorded) and np.array_equal(logits, saved))},
        }

    def score_test(self, X_test: np.ndarray) -> tuple[np.ndarray, None]:
        X = self._standardize(self.scaler, X_test).astype(np.float32)
        return self.mlp.fraud_probability(self.mlp.clear_quantized_logits(self.model, X)), None  # float twin: weights were never saved


def build_adapter(entry: dict[str, Any], config: dict[str, Any]) -> Adapter:
    return MlpAdapter(entry, config) if entry["model_type"] == "mlp" else LrXgbAdapter(entry, config)


# ---------------------------------------------------------------------------------------------------- stages

def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict[str, str]:
    from importlib import metadata

    return {n: metadata.version(n) for n in ("numpy", "scikit-learn", "xgboost", "concrete-ml", "concrete-python", "torch", "pandas") if _has(n)}


def _has(name: str) -> bool:
    from importlib import metadata

    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def stage_freeze(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    results_dir = PROJECT_ROOT / config["results_dir"]
    if (results_dir / "freeze.json").exists():
        raise FrozenProtocolError("freeze.json already exists; the freeze is written once")
    pinned = verify_pinned_inputs(config)
    export = verify_export(config)
    records, problems = [], []
    partial_dir = results_dir / "freeze_partial"  # per-model checkpoints so a killed VM does not cost the finished models; labels are not involved
    partial_dir.mkdir(parents=True, exist_ok=True)
    current_code = code_hashes()
    for entry in list_entries(config):
        cache = partial_dir / f"{entry['label']}.json"
        if cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved["code_sha256"] == current_code and saved["config_sha256"] == sha256_path(PROJECT_ROOT / config_path):
                rec = saved["record"]
                print(json.dumps({"reused_checkpoint": rec["label"]}), flush=True)
                records.append(rec)
                if not rec["validation_reproduction"]["equal"]:
                    problems.append(f"{rec['label']}: validation results were not reproduced exactly")
                continue
            cache.unlink()  # code or config changed since: rebuild
        t0 = time.perf_counter()
        adapter = build_adapter(entry, config)
        rec = adapter.record
        rec["build_and_validate_seconds"] = round(time.perf_counter() - t0, 1)
        cache.write_text(json.dumps({"record": rec, "code_sha256": current_code, "config_sha256": sha256_path(PROJECT_ROOT / config_path)}, sort_keys=True, default=str), encoding="utf-8")
        records.append(rec)
        if not rec["validation_reproduction"]["equal"]:
            problems.append(f"{rec['label']}: validation results were not reproduced exactly")
        if rec["model_type"] != "mlp" and not rec["thresholds"]["matches_recomputed"]:
            problems.append(f"{rec['label']}: stored threshold differs from the validation F1-optimal threshold by more than the tolerance")
        del adapter
        gc.collect()
        print(json.dumps({"frozen": rec["label"], "val_equal": rec["validation_reproduction"]["equal"]}), flush=True)
    if problems:
        raise FrozenProtocolError("freeze failed, nothing written: " + "; ".join(problems))
    freeze = {
        "protocol": "docs/test_evaluation_protocol.md", "pinned_inputs": pinned, "export": {k: export[k] for k in ("tier_files", "labels_file", "n_rows")},
        "config_sha256": sha256_path(PROJECT_ROOT / config_path), "grid_config_sha256": {k: sha256_path(PROJECT_ROOT / config["inputs"][k]) for k in ("phase8_config", "phase9_config")},
        "code_sha256": code_hashes(), "records": records, "git_head": _git_head(), "library_versions": _library_versions(), "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels_opened": False,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return freeze


def stage_run(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    results_dir = PROJECT_ROOT / config["results_dir"]
    assert_no_previous_run(results_dir)
    freeze = require_freeze(results_dir)
    verify_pinned_inputs(config)
    export = verify_export(config)
    if freeze["config_sha256"] != sha256_path(PROJECT_ROOT / config_path):
        raise FrozenProtocolError("the evaluation config changed after the freeze")
    for k, h in freeze["grid_config_sha256"].items():
        if sha256_path(PROJECT_ROOT / config["inputs"][k]) != h:
            raise FrozenProtocolError(f"{k} changed after the freeze")
    if freeze["code_sha256"] != code_hashes():
        raise FrozenProtocolError("source code changed after the freeze: " + ", ".join(f for f, h in code_hashes().items() if freeze["code_sha256"].get(f) != h))
    if freeze["export"]["tier_files"] != export["tier_files"] or freeze["export"]["labels_file"] != export["labels_file"]:
        raise FrozenProtocolError("the exported arrays differ from the ones frozen")

    # -- part 1: re-verify each model against the freeze and compute TEST PROBABILITIES. No label is opened here.
    frozen_by_label = {r["label"]: r for r in freeze["records"]}
    X_cache: dict[str, np.ndarray] = {}
    probs: dict[str, dict[str, Any]] = {}
    for entry in list_entries(config):
        adapter = build_adapter(entry, config)
        problems = compare_with_freeze(frozen_by_label[entry["label"]], adapter.record)
        if problems:
            raise FrozenProtocolError("; ".join(problems))
        tier = entry["tier"]
        if tier not in X_cache:
            X_cache[tier] = np.load(PROJECT_ROOT / export["tier_files"][tier]["path"])["X_test"]
        q, f = adapter.score_test(X_cache[tier])
        probs[entry["label"]] = {"quantized": q, "float": f, "record": adapter.record, "entry": entry}
        del adapter
        gc.collect()
        print(json.dumps({"scored_without_labels": entry["label"]}), flush=True)

    # -- part 2: point of no return. The marker is written BEFORE the labels are parsed; a crash after this is never resumed.
    (results_dir / "run_started.json").write_text(json.dumps({"started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), encoding="utf-8")
    y = np.load(PROJECT_ROOT / export["labels_file"]["path"])
    result = evaluate(config, freeze, probs, y)
    (results_dir / "test_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (results_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    done = {"completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "outputs_sha256": {n: sha256_path(results_dir / n) for n in ("freeze.json", "test_results.json", "summary.md")}}
    (results_dir / "run_complete.json").write_text(json.dumps(done, indent=2, sort_keys=True), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------------------------------- metrics (labels are used only below)

def evaluate(config: dict[str, Any], freeze: dict[str, Any], probs: dict[str, dict[str, Any]], y: np.ndarray) -> dict[str, Any]:
    from src.fhe.validate.correctness import t3_quantization_check
    from src.train.metrics import metrics_at_threshold, threshold_independent_metrics

    boot = config["bootstrap"]
    conf = float(boot["confidence"])
    vectors: dict[str, np.ndarray] = {}
    for label, p in probs.items():
        vectors[label] = p["quantized"]
        if p["float"] is not None:
            vectors[f"float:{p['entry']['model_type']}_{p['entry']['tier']}"] = p["float"]
    dist = bootstrap_distribution(y, vectors, int(boot["n_blocks"]), int(boot["n_resamples"]), int(boot["seed"]), int(boot.get("n_jobs", 1)))

    def block(prob: np.ndarray, name: str, thresholds: dict[str, float]) -> dict[str, Any]:
        out = {**threshold_independent_metrics(y, prob), "at_threshold": {k: metrics_at_threshold(y, prob, t) for k, t in thresholds.items()}}
        out["pr_auc_ci"] = percentile_ci(dist[name]["pr_auc"], conf)
        out["roc_auc_ci"] = percentile_ci(dist[name]["roc_auc"], conf)
        return out

    per_config: dict[str, Any] = {}
    for label, p in probs.items():
        rec, e = p["record"], p["entry"]
        thr = {"decision_threshold": rec["thresholds"]["decision_threshold"]}
        if e["model_type"] == "mlp":
            thr["qat_own_threshold"] = rec["thresholds"]["qat_own_threshold"]
        item: dict[str, Any] = {"model_type": e["model_type"], "tier": e["tier"], "n_bits": e["n_bits"], "thresholds_from_validation": thr,
                                "validation": rec["validation_reproduction"]["recorded"], "quantized": block(p["quantized"], label, thr)}
        if p["float"] is not None:
            fname = f"float:{e['model_type']}_{e['tier']}"
            item["float"] = block(p["float"], fname, thr)
            tol = e["grid_config"]["tolerances"]  # the very bars the validation gate used
            t3 = t3_quantization_check(p["quantized"], p["float"], y, thr["decision_threshold"], tol["t3_min_decision_agreement"], tol["t3_max_pr_auc_drop"])
            item["t3_analogue_on_test"] = {**t3, "note": "same bars as the validation gate; reported, not used for any decision"}
            item["quantized_minus_float_pr_auc"] = contrast_ci(dist, label, fname, "pr_auc", conf)
        else:
            item["float"] = None
            item["note"] = "MLP float twin: weights never saved, no test-set float comparison (mlp_float_twin=off)"
        per_config[label] = item

    contrasts: dict[str, Any] = {"bit_width": [], "model_gap": [], "tier_steps": []}
    by = {(p["entry"]["model_type"], p["entry"]["tier"], p["entry"]["n_bits"]): label for label, p in probs.items()}
    for (m, t, b), label in sorted(by.items()):
        higher = [bb for (mm, tt, bb) in by if mm == m and tt == t and bb > b]
        if higher:
            contrasts["bit_width"].append(contrast_ci(dist, by[(m, t, max(higher))], label, "pr_auc", conf))
    for t in ("top_20", "top_50", "top_100"):
        lr, xg = [k for k in by if k[0] == "lr" and k[1] == t], [k for k in by if k[0] == "xgboost" and k[1] == t]
        if lr and xg:
            contrasts["model_gap"].append(contrast_ci(dist, by[max(xg, key=lambda k: k[2])], by[max(lr, key=lambda k: k[2])], "pr_auc", conf))
    for m in ("lr", "xgboost", "mlp"):
        for b in sorted({k[2] for k in by if k[0] == m}):
            for small, big in (("top_20", "top_50"), ("top_50", "top_100")):
                if (m, small, b) in by and (m, big, b) in by:
                    contrasts["tier_steps"].append(contrast_ci(dist, by[(m, big, b)], by[(m, small, b)], "pr_auc", conf))
    return {
        "status": "complete", "n_test_rows": int(len(y)), "n_positive": int(y.sum()), "positive_rate": float(y.mean()),
        "bootstrap": {**boot, "method": "time-block percentile bootstrap, paired resamples", "n_resamples_without_both_classes": int(np.isnan(next(iter(dist.values()))["pr_auc"]).sum())},
        "labels": "TEST. Post hoc. Nothing here selects, tunes or changes any model, tier, bit-width or threshold.", "configurations": per_config, "contrasts": contrasts,
        "freeze_sha256": hashlib.sha256(json.dumps(freeze, sort_keys=True, default=str).encode()).hexdigest(),
    }


def render_summary(result: dict[str, Any]) -> str:
    lines = ["# Frozen test-partition results (post hoc; not used for selection)", "", f"Rows {result['n_test_rows']:,}, positives {result['n_positive']:,} ({100 * result['positive_rate']:.2f}%). Intervals: {result['bootstrap']['method']}, {result['bootstrap']['n_resamples']} resamples.", "",
             "| Config | Validation PR-AUC (quantized) | Test PR-AUC (quantized) [95% CI] | Test ROC-AUC [95% CI] | Test PR-AUC (float) |", "|---|---|---|---|---|"]
    for label, c in result["configurations"].items():
        q = c["quantized"]
        f = f"{c['float']['pr_auc']:.4f}" if c.get("float") else "n/a"
        lines.append(f"| {label} | {c['validation']['quantized_pr_auc']:.4f} | {q['pr_auc']:.4f} [{q['pr_auc_ci'][0]:.4f}, {q['pr_auc_ci'][1]:.4f}] | {q['roc_auc']:.4f} [{q['roc_auc_ci'][0]:.4f}, {q['roc_auc_ci'][1]:.4f}] | {f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase12/test_eval.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    import warnings

    warnings.filterwarnings("ignore")
    from src.config import load_config

    config = load_config(args.config)
    try:
        stage_freeze(config, args.config) if args.freeze else stage_run(config, args.config)
    except FrozenProtocolError as exc:
        print(f"PROTOCOL STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
