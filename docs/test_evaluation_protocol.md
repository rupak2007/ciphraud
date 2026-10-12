# Frozen test-partition evaluation — PROTOCOL (stage A freeze COMPLETE; stage B `--run` NOT EXECUTED, awaiting explicit approval)

Nothing in this document has been run. No test label has been read for this work. It is a proposal; the owner approves before anything executes.
Purpose: state absolute held-out performance for the already-selected models and check, on unseen later-in-time rows, the relative findings in `report.md`.
It must not change any model, tier, threshold, bit-width or earlier result.

## 1. What was inspected (read-only, 2026-10-12)

**Split and preprocessing.** Time-based split on `TransactionDT` quantiles 0.6 / 0.8 (`results/phase2_pipeline/split_boundaries.json`: train <= 9,521,239 < val <= 12,666,185 < test).
Train 380,815 / val 105,088 / test 104,637 rows. `src/data/pipeline.py` fits the preprocessor (`fit_preprocessor`) on the train partition only, then only `transform`s val and test
(frequency encoder and median imputer record their train `fit_index_`; the leakage audit checks this). Cached matrices: `data/processed/{train,val,test}.pkl`.
Verified today that tier columns of the cached `val.pkl` equal, bit for bit, the validation arrays in the Phase 6/8/9 handoffs (feature values, labels and TransactionIDs, tiers 20 and 50; same code path for 100),
so applying the same tier columns to `test.pkl` is the identical preprocessing that the models were built on.

**Were test labels ever used?** Evidence: (a) `"test_partition_touched": false` in all 51 result files that carry the field, none true; (b) in `src/`, `X_test`/`y_test` appear only in the Phase 2 pipeline and loader
(construction, caching, row counts) and in comments stating they are never used; (c) sentinel tests in `tests/test_train_pipeline_*`, `tests/test_features_pipeline_*` and `tests/test_fhe_export*` replace the test objects with
objects that raise on access and still pass; (d) Phase 3-9 thresholds are chosen by `select_threshold(y_val, ...)` on validation labels, and tiers, tree counts and hyperparameters on train/validation only.
**Disclosed exceptions (descriptive, not used for any decision):** Phase 2 computed and stored the test partition's positive rate (0.03404; train 0.03417, val 0.03892) and per-column out-of-vocabulary rates
in `results/phase2_pipeline/preprocessing_summary.json`, and wrote the test matrix with its labels into the local `data/processed/test.pkl`. I did read the same positive rate and the shape of `test.pkl` for this review.
I did not compute any model score on test.

## 2. Frozen inputs (hashes recorded before any scoring)

| Item | Value |
|---|---|
| `data/processed/test.pkl` | sha256 `0775f916203aff0123ad004a211d93409b6808cf21efff771e1ae5e033a43bb8` |
| `data/processed/val.pkl` | sha256 `b077c8807b5e2a62133936ac886ec75817bdef43db0e1a0c7d0160a499e21255` |
| `results/phase4_features/tiers.json` | sha256 `c62a9f9068bb7b2a13148f16371ba69dc187cec17c2ca29caba8fe8ab4002f6b` |
| `results/phase2_pipeline/split_boundaries.json` | sha256 `d7239c67bf30c7244fe82a666f0e8aa4c06faa7a970f96b54a6b227556cd87c6` |
| `models/phase3_baselines/xgboost.json` | sha256 `51b39a15cc25c743b0fe62448528728d34e9922e13652f917b41fc59371df679` |
| Models | the 18 grid configurations of Phases 8-9 (6 LR, 6 XGBoost, 6 MLP) rebuilt by the existing loaders, plus the two 832-feature Phase 3 baselines (float only) |
| Thresholds | recomputed from validation labels with `select_threshold` exactly as the grids did (LR/XGBoost: the float reference's F1-optimal threshold; MLP: float twin's, plus the QAT model's own, both from validation), written to the freeze file before test labels are opened |
| Calibration sets, seeds | as in `configs/phase8/research_grid.yaml` and `configs/phase9/mlp_grid.yaml` (seed 42), unchanged |

A freeze manifest (`results/phase12_test_eval/freeze.json`) will hold these hashes, the thresholds and the rebuilt-model fingerprints. The evaluation script refuses to run if the manifest is missing, any hash differs, or a completed-run marker already exists.

## 3. Procedure

1. **Windows step (needs pandas pickle):** `python -m src.analysis.test_eval_export` reads `test.pkl`, writes per-tier feature arrays `X_test` and TransactionIDs to `data/phase12/test_arrays/{tier}.npz` (gitignored), and labels to a separate `y_test.npy`; no labels are loaded into memory by the feature export. Records array hashes.
2. **WSL step, stage A (labels still closed):** rebuild each of the 18 models with the existing loaders; **reproduce the recorded validation results** (quantized/float PR-AUC and T3 agreement equal the committed `metrics.json` values exactly, which also proves the rebuilt model is the accepted one); write `freeze.json`. Any mismatch stops the run.
3. **WSL step, stage B (single run):** load `y_test`; score each config on test with the float reference and the clear-quantized model (`fhe="disable"`; T2/T1 already showed the circuit equals it; **no encryption, no keys, no FHE runs**); compute metrics at the frozen threshold; write results and a run-complete marker.
4. Exactly one execution. No re-runs after seeing results; a crash before stage B opens labels may be resumed, after that never.

## 4. Metrics (only what the research claims need)

PR-AUC (primary), ROC-AUC, precision / recall / F1 / F2 and the confusion matrix at the frozen threshold, for float and clear-quantized models; T3-style decision agreement and PR-AUC drop (quantized minus float) on test;
for the two LR/XGBoost families, the comparison at each family's highest passing bit-width. Optional (needs approval): 1,000-resample bootstrap 95% intervals (seed 42) for PR-AUC and for paired differences (quantized vs float; adjacent tiers), to show whether the single-split ordering holds.
No new thresholds, no threshold sweeps, no model comparison for selection.

## 5. Decision rules (pre-registered)

- Results are reported as found, whatever they show, next to the validation numbers in a new `results/phase12_test_eval/` folder and a labelled addition to `report.md` (a separate section, "held-out test results, post hoc, not used for selection").
- No model, tier, bit-width, threshold, gate or earlier result is changed because of them. A test result worse than validation is expected possible (later time period, drift) and is not a reason to retune.
- Claims in `report.md` are marked confirmed / not confirmed on test: bit-width effect (8 vs 14/16), LR vs XGBoost gap, tier ordering, MLP T3 failure.

## 6. Commands (to be created after approval; names are proposals)

```
python -m src.analysis.test_eval_export --config configs/phase12/test_eval.yaml         # Windows
python -m src.analysis.test_evaluation --config configs/phase12/test_eval.yaml --freeze # WSL, stage A
python -m src.analysis.test_evaluation --config configs/phase12/test_eval.yaml --run    # WSL, stage B, once
```

## 7. Expected cost (estimates from recorded timings; not measured for this task)

- **Runtime:** compilation to rebuild the quantized models: LR 6 x about 95 s, XGBoost 6 x 34-111 s (about 6 min total), MLP 6 x about 5-9 s, plus validation reproduction and test scoring in clear mode.
  Overall about 20-40 minutes, one WSL session. With bootstrap intervals add a few minutes.
- **Memory:** compile peaks recorded at up to about 3.5 GB (XGBoost); the WSL limit is currently 5 GB. Nothing else should be running.
- **Storage:** test tier arrays about 40-150 MB depending on dtype (gitignored), a few small JSON files in `results/` (under 1 MB). No keys, no ciphertexts.
- **Artifact changes (all new files; no existing file modified except `report.md` afterwards):** `configs/phase12/test_eval.yaml`, `src/analysis/test_eval_export.py`, `src/analysis/test_evaluation.py`, `tests/test_phase12_test_eval.py` (refuses without freeze, refuses a second run, hash mismatch stops, rebuilt val results must match), `results/phase12_test_eval/`, `docs/` write-up.

## 8. What it will not establish

It is one later-in-time partition, point estimates (unless the optional intervals are approved), and the dataset's fraud labels and features are anonymized; it says nothing about other datasets, other periods, or real deployment.
The fraud-detection quality is a property of the plaintext models; the test run adds no FHE-cost information.

---

## 9. Preparation status (2026-10-12) — scripts written, NOT run; changes to the draft above

Approved by the owner: preparation for execution under these safeguards, with pre-specified 95% bootstrap intervals. **No stage has been executed**; no test label has been parsed.

**Changes versus sections 1-8, and why**
1. **Bootstrap (approved, pre-specified):** time-block percentile bootstrap, 50 contiguous blocks of the time-ordered test rows, 1,000 resamples, seed 42, 95% intervals, paired resamples across all models, for PR-AUC and ROC-AUC and for paired differences (quantized minus float; high minus low bit-width; XGBoost minus LR per tier; tier steps). A block bootstrap is used because the rows are a time series with repeated cards; an i.i.d. row bootstrap would give intervals that are too narrow. It captures test-sample variability only, not training or model-selection variability, and 50 blocks give a coarse resampling unit. Resamples without both classes are skipped and counted (none expected).
2. **MLP float twin (finding):** the twin's *weights were never saved* (`data/fhe_handoff/phase9/checkpoints/float_twin/*` holds only `training.json` and validation logits). A test-set QAT-versus-float comparison for the MLP therefore needs retraining the twin. Default in the config is `mlp_float_twin: "off"`: only the saved QAT models are scored, so the MLP rows have no test-set float counterpart or T3 analogue. Retraining (with a bit-exact check against the saved validation logits) is NOT implemented and needs separate approval.
3. **Phase 3 832-feature baselines dropped** from the executable protocol (they would need a 350 MB full-matrix export); report text keeps citing them as validation-only.
4. **Threshold rule made precise:** LR/XGBoost use the stored handoff-manifest threshold (verified equal to the validation F1-optimal threshold of the float reference); the MLP uses the float twin's validation F1-optimal threshold (as its T3 gate did) and also its own QAT validation threshold. Both are written into `freeze.json` before labels are opened.
5. **MLP scoring needs no compilation or keys** (saved QAT checkpoints, clear mode), so the three key-infeasible cells are scorable like the others.

**Files created (all new; none of the existing evidence is modified)**
`configs/phase12/test_eval.yaml`, `src/analysis/test_eval_export.py`, `src/analysis/test_evaluation.py`, `tests/test_phase12_test_eval.py` (14 tests, passing on synthetic data only: refuses without a freeze, refuses a second or resumed run, stops on any hash mismatch, label file parsed only in the run stage after the marker, bootstrap determinism/pairing, end-to-end evaluate on synthetic scores).

**Final commands (to run only after approval)**
```
# 1. Windows .venv (pandas): writes data/phase12/test_arrays/ (tier features + ids, labels in a separate file); ~1-2 min
python -m src.analysis.test_eval_export --config configs/phase12/test_eval.yaml

# 2. WSL FHE venv, stage A (labels stay closed): rebuild 18 models, reproduce validation numbers exactly, write freeze.json; ~10-20 min
python -m src.analysis.test_evaluation --config configs/phase12/test_eval.yaml --freeze

# 3. review results/phase12_test_eval/freeze.json together; then, once:
python -m src.analysis.test_evaluation --config configs/phase12/test_eval.yaml --run   # re-verifies, scores, opens labels, bootstrap; ~20-40 min
```
**Expected outputs:** `data/phase12/test_arrays/{X_test_top_20,X_test_top_50,X_test_top_100}.npz`, `labels/y_test.npy`, `export.json` (gitignored); `results/phase12_test_eval/freeze.json`, `run_started.json`, `test_results.json`, `summary.md`, `run_complete.json` (committed after review).

**Disk and memory:** test arrays about 142 MB (float64, 3 tiers: 104,637 x 170 x 8 bytes) plus 0.8 MB labels; results about 1 MB; WSL pip/temp usage small. Free space on C: was 47 GB. Peak memory: XGBoost compile about 3.5 GB (recorded), within the 5 GB WSL limit; run with nothing else heavy open. The `--run` stage rebuilds all models again (compile about 10 min), because compiled models are not persisted between stages.

**Remaining risks**
- Stage A could fail to reproduce a recorded validation number exactly (e.g. different BLAS threading or a library difference). Then the run stops by design, and I report it rather than loosen the tolerance (1e-12).
- A crash after the label file is opened is final: `run_started.json` blocks any rerun.
- Test performance may be lower than validation (later period, drift). It is reported as found, and nothing is changed because of it.
- Block-bootstrap intervals are approximate and do not include training-time or selection variability; the validation set already selected trees, thresholds and tiers.
- The Phase 2 matrices were produced by the cached pipeline; the validation-equality check against the handoffs (today, tiers 20/50 explicit) supports that the test matrix uses the identical preprocessing, but the test rows cannot be checked against a second source.
- Exposure already disclosed in section 1 (test positive rate and out-of-vocabulary rates seen earlier; I loaded `test.pkl` once to read its shape and fraud rate).

---

## 10. Freeze result (2026-10-12; stage A complete, labels NOT opened)

`results/phase12_test_eval/freeze.json` (sha256 `3a192a945f62e937f7230191f7a2d86d609465414a156f1ebc7846c2d24a2ced`, frozen 2026-10-12T05:23:10Z, git HEAD `7ae3b82`) records all 18 configurations. For every one the rebuilt model reproduced the committed validation quantized and float PR-AUC to within 1e-12 (all `equal: true`; for the MLP the saved validation logits were also reproduced exactly). `labels_opened: false`; no `run_started.json`, `run_complete.json` or `test_results.json` exists.

**Corrections to this document's earlier statements (found during the freeze):**
1. The stored LR thresholds equal the validation F1-optimal threshold only to about 1e-7 (differences 3.4e-8, 1.0e-7, 4.5e-8 for top-20/50/100); the six XGBoost thresholds match exactly. The freeze accepts a difference below 1e-6 and records it. The **stored** thresholds are the frozen decision thresholds.
2. XGBoost scoring (validation reproduction and test) uses Phase 8's chunked integer path (`disable_integers`, 1,000 rows), not a single `predict_proba` call, which needs about 17.7 GiB.

**Attempts before the successful freeze (all logs kept):** attempt 1 killed by a WSL VM restart (3 of 18 checkpoints); attempt 2 failed at the first XGBoost model (the 17.7 GiB allocation); attempt 3 completed all 18 checkpoints but failed assembly on the too-strict 1e-9 threshold check; attempt 4 succeeded. Superseded checkpoint folders: `freeze_partial_superseded_code_v1/` and `_v2/`. Logs: `data/phase12/logs/freeze*.log|err`, `guard_v{2,3,4}_state.log`, `guard_v1_*`.

**Frozen thresholds** (decision threshold, from `freeze.json`): LR 0.70864117 / 0.82168913 / 0.83436090 (top-20/50/100); XGBoost 0.80312127 / 0.81969267 / 0.77844280; MLP float-twin validation F1-optimal 0.90609349 / 0.88008759 / 0.91681667, plus each QAT model's own validation threshold. The 12 code-file hashes in `freeze.json` match the live files.

The `--run` stage still requires explicit approval.


---

## 11. Pre-commit audit note (2026-10-12 14:45)

Verified without reading the sealed files' bytes: `freeze.json` sha256 `3a192a945f62e937f7230191f7a2d86d609465414a156f1ebc7846c2d24a2ced`; all 12 code hashes, the evaluation and grid config hashes, the three feature-array hashes and the two small pinned JSON hashes match; the 18 `freeze_partial` records equal the freeze records; superseded folders preserved; no `run_started.json`, `run_complete.json`, `test_results.json` or `summary.md`; no evaluation process running; the only code that parses the label file is `stage_run`, after the `run_started.json` marker.
**Open observation (unexplained):** the label file `data/phase12/test_arrays/labels/y_test.npy` has a last-access time of 2026-10-12 12:02:37 (modified 00:18:57 at export; size unchanged 837,224 bytes), and `logs/guard_v2_state.log` was accessed in the same second. No Ciphraud code path that parses labels ran (no marker, no results). The reader is unidentified: an antivirus scan or another session's file listing is possible but not shown. A read of the bytes does not by itself reveal that anything was scored or tuned, but this is disclosed because "labels unopened" cannot be asserted without qualification. `--run` verifies the label file's sha256 before use, so a modified file would stop the run.

**Update (2026-10-12 15:5x, pre-commit verification):** the label file's last-access time now reads 14:51:20 and later 15:5x, i.e. it moved again. Cause unknown: possibly my own metadata calls (`stat`/`ls`) opening a handle, an indexer or antivirus; NTFS access-time updates are lazy, so this is not proof of a content read. No marker, result or scoring process exists, the file's mtime (00:18:57) and size (837,224) are unchanged, and no Ciphraud code that parses labels has run. Further metadata probing of that file has been stopped.

**Investigation result (2026-10-12 ~16:20, before the sealed run; labels not opened, hashed or copied):** a metadata probe at 15:53:46 (`ls --time=atime` run from Git Bash) showed the *same* access time, to the millisecond, on `y_test.npy`, all three `X_test_top_*.npz` files and `export.json`, equal to the wall-clock time printed by `date` in that very command. Git Bash's `stat`/`ls` on Windows evidently opens file handles and NTFS records an access time, so the later change was caused by my own inspection and is not evidence of a label read. An earlier probe round produced the same uniform pattern (14:51:20 on the feature files and labels alike), and the 12:02:37 value coincides with the previous audit's directory listing (the Claude session transcripts show no other tool call at that time). A targeted label read would not touch the feature arrays at the same instant. This is a strong but not certain explanation; the residual uncertainty is that no OS-level audit log exists to prove it. No code that parses labels has run (no marker, no results), the label file's size and mtime are unchanged, and `--run` verifies its sha256 against the pinned value before use. Decision: the timestamp is not credible evidence of contamination; the run proceeds under the existing guard.

## 12. Run record (2026-10-12)
- Command: `python -m src.analysis.test_evaluation --run` in the WSL FHE venv (`~/.venvs/fhe-fraud-detection`), started 16:38:47 IST, `run_started.json` 12:48:23Z (labels parsed after all 18 models were re-verified and scored without labels), `run_complete.json` 12:59:44Z, **exit 0, run exactly once**, full stdout in `data/phase12/logs/sealed_run.log` (stderr empty).
- Outputs and sha256: `freeze.json` 3a192a94...2ced (unchanged), `test_results.json` 63c5ac2e...10e, `summary.md` 7eb40f1e...c8 (also in `run_complete.json`).
- Result: all 18 configurations lower on test than validation; see `summary.md`. No threshold, model, code or config was changed before, during or after the run; the run will not be repeated to replace this result.
- Caveats: the label-file access-time question (section 11) remains explained only circumstantially; the validation set had selected tiers/thresholds, so a validation-to-test drop is expected, and part of the drop reflects temporal drift in fraud rates and patterns.
