"""Safeguards of the frozen test-partition evaluation (docs/test_evaluation_protocol.md). Synthetic data only: no real test label or
real test feature is read by these tests, and no model is built."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import test_evaluation as te

ROOT = Path(__file__).resolve().parents[1]


def _src(obj) -> str:
    """Source text of a function/class/method. `inspect.getsource` fails when a cached .pyc written on another OS (WSL vs Windows) records a
    foreign file path, so fall back to parsing the module file by qualified name."""
    try:
        return inspect.getsource(obj)
    except OSError:
        import ast

        text = Path(te.__file__).read_text(encoding="utf-8")
        node = ast.parse(text)
        for part in obj.__qualname__.split("."):
            node = next(n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part and n is not node)
        return ast.get_source_segment(text, node)


def _config(tmp_path):
    return {"results_dir": str(tmp_path / "results"), "export_dir": str(tmp_path / "export"), "bootstrap": {"n_resamples": 60, "n_blocks": 10, "seed": 7, "confidence": 0.95, "n_jobs": 1},
            "inputs": {}, "mlp_float_twin": "off"}


# ---- guards ---------------------------------------------------------------------------------------------------------

def test_run_refuses_without_a_freeze_and_opens_no_label_file(tmp_path, monkeypatch):
    opened = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda p, *a, **k: (opened.append(str(p)), real_load(p, *a, **k))[1])
    with pytest.raises(te.FrozenProtocolError, match="freeze.json missing"):
        te.stage_run(_config(tmp_path), "configs/phase12/test_eval.yaml")
    assert not any("y_test" in p for p in opened)


@pytest.mark.parametrize("marker", ["run_started.json", "run_complete.json"])
def test_run_refuses_a_second_or_resumed_run(tmp_path, marker):
    results = tmp_path / "results"
    results.mkdir()
    (results / marker).write_text("{}")
    with pytest.raises(te.FrozenProtocolError, match="exactly once"):
        te.stage_run(_config(tmp_path), "configs/phase12/test_eval.yaml")


def test_modified_pinned_input_stops_before_anything_is_read(tmp_path):
    f = tmp_path / "test.pkl"
    f.write_bytes(b"not the pinned file")
    cfg = {"inputs": {"test_pkl": str(f), "test_pkl_sha256": "0" * 64, "tiers_json": str(f), "tiers_json_sha256": "0" * 64, "split_boundaries": str(f), "split_boundaries_sha256": "0" * 64}}
    with pytest.raises(te.FrozenProtocolError, match="!= pinned"):
        te.verify_pinned_inputs(cfg)


def test_tampered_export_is_detected(tmp_path):
    export = tmp_path / "export"
    (export / "labels").mkdir(parents=True)
    arr = export / "X_test_top_20.npz"
    arr.write_bytes(b"x")
    y = export / "labels" / "y_test.npy"
    y.write_bytes(b"y")
    manifest = {"tier_files": {"top_20": {"path": str(arr), "sha256": te.sha256_path(arr)}}, "labels_file": {"path": str(y), "sha256": te.sha256_path(y)}, "n_rows": 1}
    (export / "export.json").write_text(json.dumps(manifest))
    cfg = {"export_dir": str(export)}
    assert te.verify_export(cfg)["n_rows"] == 1
    y.write_bytes(b"changed")
    with pytest.raises(te.FrozenProtocolError, match="label file was modified"):
        te.verify_export(cfg)


def test_compare_with_freeze_flags_any_model_or_threshold_change():
    base = {"label": "m", "fingerprints": {"a": "1"}, "thresholds": {"decision_threshold": 0.5}, "validation_reproduction": {"equal": True, "recorded": {"quantized_pr_auc": 0.4}}}
    assert te.compare_with_freeze(base, json.loads(json.dumps(base))) == []
    changed = json.loads(json.dumps(base))
    changed["thresholds"]["decision_threshold"] = 0.51
    assert len(te.compare_with_freeze(base, changed)) == 1


def test_only_the_run_stage_ever_parses_the_label_file():
    for fn in (te.stage_freeze, te.verify_export, te.build_adapter, te.LrXgbAdapter.__init__, te.MlpAdapter.__init__):
        src = _src(fn)
        assert "y_test" not in src and 'np.load(PROJECT_ROOT / export["labels_file"]' not in src, fn.__name__
    assert "np.load" not in _src(te.verify_export)  # hashes the label file's bytes only
    assert _src(te.stage_run).count('export["labels_file"]["path"]') == 1


def test_labels_are_opened_only_after_the_run_marker_is_written():
    src = _src(te.stage_run)
    assert src.index("run_started.json") < src.index('np.load(PROJECT_ROOT / export["labels_file"]["path"])')
    assert src.index("score_test") < src.index("run_started.json")  # all test probabilities exist before any label is parsed


def test_mlp_float_twin_retraining_is_not_available_without_separate_approval():
    pytest.importorskip("concrete.ml", reason="MlpAdapter imports the Concrete-ML stack (WSL venv only)")
    cfg = {"mlp_float_twin": "retrain_verified"}
    with pytest.raises(te.FrozenProtocolError, match="separate approval"):
        te.MlpAdapter({"grid_config": {"seed": 42}, "tier": "top_20", "n_bits": 3, "label": "x"}, cfg)


# ---- the pinned values and the live inputs ---------------------------------------------------------------------------

def test_pinned_hashes_in_the_config_match_the_protocol_document():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "phase12" / "test_eval.yaml").read_text(encoding="utf-8"))
    doc = (ROOT / "docs" / "test_evaluation_protocol.md").read_text(encoding="utf-8")
    for key in ("test_pkl_sha256", "tiers_json_sha256", "split_boundaries_sha256"):
        assert cfg["inputs"][key] in doc
    assert cfg["bootstrap"]["confidence"] == 0.95 and cfg["mlp_float_twin"] == "off"


def test_live_small_inputs_still_match_their_pinned_hashes():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "phase12" / "test_eval.yaml").read_text(encoding="utf-8"))
    for key in ("tiers_json", "split_boundaries"):
        assert te.sha256_path(ROOT / cfg["inputs"][key]) == cfg["inputs"][f"{key}_sha256"], key  # hashes JSON files only, never the test data


# ---- bootstrap and evaluation on synthetic data ------------------------------------------------------------------------

def _synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    good = np.clip(0.2 * y + rng.normal(0.3, 0.15, n), 0, 1)
    weak = np.clip(0.05 * y + rng.normal(0.3, 0.15, n), 0, 1)
    return y, good, weak


def test_resamples_are_deterministic_and_made_of_contiguous_blocks():
    edges = te.block_edges(1000, 10)
    a, b = te.resample_indices(edges, 5, 3), te.resample_indices(edges, 5, 3)
    assert np.array_equal(a, b) and len(a) == 1000
    runs = a.reshape(10, 100)
    assert all(np.array_equal(r, np.arange(r[0], r[0] + 100)) for r in runs)  # each block stays whole and time-ordered


def test_bootstrap_is_paired_reproducible_and_identical_vectors_have_zero_difference():
    from sklearn.metrics import average_precision_score

    y, good, weak = _synthetic()
    vec = {"good": good, "good2": good.copy(), "weak": weak}
    d1 = te.bootstrap_distribution(y, vec, 10, 60, 7, n_jobs=1)
    d2 = te.bootstrap_distribution(y, vec, 10, 60, 7, n_jobs=1)
    assert np.array_equal(d1["good"]["pr_auc"], d2["good"]["pr_auc"])
    assert np.array_equal(d1["good"]["pr_auc"], d1["good2"]["pr_auc"])
    zero = te.contrast_ci(d1, "good", "good2", "pr_auc", 0.95)
    assert zero["ci"] == [0.0, 0.0] and zero["mean_resampled_diff"] == 0.0
    lo, hi = te.percentile_ci(d1["good"]["pr_auc"], 0.95)
    assert lo <= average_precision_score(y, good) * 1.15 and hi >= average_precision_score(y, good) * 0.85
    assert te.contrast_ci(d1, "good", "weak", "pr_auc", 0.95)["fraction_a_greater"] > 0.9


def test_evaluate_runs_end_to_end_on_synthetic_scores_and_keeps_the_validation_numbers_beside_the_test_ones():
    y, good, weak = _synthetic()
    grid = {"tolerances": {"t3_min_decision_agreement": 0.99, "t3_max_pr_auc_drop": 0.01}}

    def entry(model, tier, bits):
        return {"model_type": model, "tier": tier, "n_bits": bits, "grid_config": grid}

    def rec(thr=0.5):
        return {"thresholds": {"decision_threshold": thr, "qat_own_threshold": thr}, "validation_reproduction": {"recorded": {"quantized_pr_auc": 0.4, "float_pr_auc": 0.41}}}

    probs = {
        "lr_top20_bits8": {"quantized": weak, "float": good, "record": rec(), "entry": entry("lr", "top_20", 8)},
        "lr_top20_bits16": {"quantized": good, "float": good, "record": rec(), "entry": entry("lr", "top_20", 16)},
        "xgboost_top20_bits14": {"quantized": good, "float": good, "record": rec(), "entry": entry("xgboost", "top_20", 14)},
        "mlp_top20_bits3": {"quantized": weak, "float": None, "record": rec(), "entry": entry("mlp", "top_20", 3)},
    }
    result = te.evaluate(_config(Path(".")), {"x": 1}, probs, y)
    c = result["configurations"]
    assert c["lr_top20_bits8"]["validation"]["quantized_pr_auc"] == 0.4  # validation value stays next to the test value
    assert c["lr_top20_bits8"]["t3_analogue_on_test"]["passed"] is False and c["lr_top20_bits16"]["t3_analogue_on_test"]["passed"] is True
    assert c["mlp_top20_bits3"]["float"] is None and "weights never saved" in c["mlp_top20_bits3"]["note"]
    assert len(result["contrasts"]["bit_width"]) == 1 and len(result["contrasts"]["model_gap"]) == 1
    assert "Post hoc" in result["labels"] and result["n_positive"] == int(y.sum())
    assert "| lr_top20_bits8 |" in te.render_summary(result)


def test_code_hashes_cover_the_loaders_and_scoring_code_and_exist():
    h = te.code_hashes()
    assert set(h) == set(te.CODE_FILES) and all(len(v) == 64 for v in h.values())
    assert "src/benchmark/run.py" in h and "src/fhe/compile/mlp.py" in h and "src/analysis/test_evaluation.py" in h
    assert 'freeze["code_sha256"] != code_hashes()' in _src(te.stage_run)


def test_freeze_checkpoints_each_model_and_only_reuses_them_for_identical_code_and_config():
    src = _src(te.stage_freeze)
    assert "freeze_partial" in src and 'saved["code_sha256"] == current_code' in src and 'saved["config_sha256"]' in src
    assert "y_test" not in src


def test_xgboost_clear_scoring_uses_the_chunked_phase8_integer_path_never_one_giant_predict_proba():
    src = _src(te.LrXgbAdapter)
    assert "disable_integers" in src and "self.chunk" in src and "_xgb_probabilities(X_test)" in src
    assert 'cml.predict_proba(X_test, fhe="disable")' not in src.split("def score_test")[1].split("else:")[1]  # XGBoost branch only


def test_stage_freeze_assembles_end_to_end_with_stored_thresholds_that_differ_by_float_noise(tmp_path, monkeypatch):
    """Regression for the failed assembly: LR thresholds differ from the recomputed ones by ~1e-7 and must be accepted; 1e-3 must not."""
    cfg = {"results_dir": str(tmp_path / "results"), "export_dir": str(tmp_path / "export"), "inputs": {"phase8_config": "configs/phase8/research_grid.yaml", "phase9_config": "configs/phase9/mlp_grid.yaml"}}

    def make(diff):
        class Fake:
            def __init__(self, entry, config):
                thr = {"decision_threshold": 0.5, "recomputed_from_validation_float_prob": 0.5 + diff, "matches_recomputed": abs(diff) < te.THRESHOLD_TOL, "abs_difference": abs(diff)}
                if entry["model_type"] == "mlp":
                    thr = {"decision_threshold": 0.5, "qat_own_threshold": 0.4}
                self.record = {"label": entry["label"], "model_type": entry["model_type"], "tier": entry["tier"], "n_bits": entry["n_bits"], "fingerprints": {"x": "1"}, "thresholds": thr,
                               "validation_reproduction": {"recorded": {"quantized_pr_auc": 0.4, "float_pr_auc": 0.4}, "reproduced": {"quantized_pr_auc": 0.4, "float_pr_auc": 0.4}, "equal": True}}
        return Fake

    export = {"tier_files": {"top_20": {"path": "a", "sha256": "b"}}, "labels_file": {"path": "c", "sha256": "d"}, "n_rows": 5}
    monkeypatch.setattr(te, "verify_pinned_inputs", lambda c: {"k": "v"})
    monkeypatch.setattr(te, "verify_export", lambda c: export)
    monkeypatch.setattr(te, "build_adapter", make(3.4e-8))
    freeze = te.stage_freeze(cfg, "configs/phase12/test_eval.yaml")
    assert len(freeze["records"]) == 18 and freeze["labels_opened"] is False and set(freeze["code_sha256"]) == set(te.CODE_FILES)
    assert (tmp_path / "results" / "freeze.json").exists()
    # a real threshold disagreement must still stop the freeze and write nothing
    (tmp_path / "results" / "freeze.json").unlink()
    import shutil

    shutil.rmtree(tmp_path / "results" / "freeze_partial")
    monkeypatch.setattr(te, "build_adapter", make(1e-3))
    with pytest.raises(te.FrozenProtocolError, match="tolerance"):
        te.stage_freeze(cfg, "configs/phase12/test_eval.yaml")
    assert not (tmp_path / "results" / "freeze.json").exists()
