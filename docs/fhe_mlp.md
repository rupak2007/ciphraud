# Phase 9 — Quantized MLP under FHE (living document)

**Status: COMPLETE (2026-09-28).** All 6 configured `model × tier × bit-width` cells (`configs/phase9/mlp_grid.yaml`) have been attempted and recorded — every one, either to a plaintext-and-FHE outcome or to a fully-evidenced infeasibility. The FR5/D5 plaintext-only seed-stability evaluation (`--plaintext`, 3 seeds × 3 tiers, primary bit-width 4) has also been run — see §7.7. Code, tests, the export, the API probe, the feasibility probe and the combined Pareto analysis (§7) are done. Every number below traces to a file under `results/phase9_mlp/` or to `configs/phase9/mlp_grid.yaml`; nothing is estimated.

## 1. Objective

`docs/plan.md` Phase 9: extend the LR/XGBoost benchmark to a **quantized MLP** (Brevitas quantization-aware training, per `docs/architecture.md` §5), compile it with Concrete-ML, validate its correctness, log **TLU (programmable-bootstrap) count and compile time** per configuration, and integrate it into the same Pareto analysis. It also closes **PRD FR5** ("train and evaluate a quantized MLP as a plaintext baseline"), which Phase 3 never did.

## 2. Decisions (approved 2026-09-26)

| | Decision | Why it was needed |
|---|---|---|
| D1 | One bit-width knob `b = n_w_bits = n_a_bits`; accumulator capped at `n_accum_bits = 16`. Grid bit-widths = the largest `b` feasible for all three tiers, plus the next lower one. 8 is probed and logged | The plan's "the same bit-width grid as Phases 6–8" has no single meaning for an MLP, which has weight, activation and accumulator precision |
| D2 | **T3 = QAT clear-quantized model vs a float twin** (same architecture, data, seed, class weights, optimizer, epoch budget; no quantization), full validation, Phase 8's bars (decision agreement ≥ 0.99, PR-AUC drop ≤ 0.01) | QAT trains the network already quantized, so it has no float version of itself; the comparison therefore includes training-run variance, which the seed-stability spread (D5) sizes |
| D3 | **T0 = the checkpoint written after training reloads to exactly the same clear-quantized outputs** | Phase 8's T0 checked a Windows→WSL transfer; the MLP is trained inside WSL, so nothing crosses that boundary |
| D4 | One fixed small architecture (Concrete-ML `NeuralNetClassifier` defaults: `n_layers=3`, hidden width = 4 × input features, ReLU), full train split, class-weighted loss, early stopping and threshold on **validation**, **no hyperparameter search** | Deviates from Phase 4's per-tier CV re-tuning of LR/XGBoost; validation-based early stopping carries the same caveat as XGBoost's (`docs/research.md` §7.7) |
| D5 | Plaintext seed stability over 3 seeds × 3 tiers at the primary bit-width | CLAUDE.md §11 |
| D6 | Local commits before the grid runs, so `provenance.json` records the real code commit (Phase 8's stale `e85d16a` problem) | Pushing still only on the user's instruction |
| D7 | The QAT trainer's FINAL re-export/re-quantize (`restore_weights`, called once after early stopping) calibrates on the same ~3,000-row `compile_calibration` subset already used to compile the FHE circuit, instead of the full 380,815-row training set. Every epoch of actual gradient training is unchanged | Every `mlp_top50_bits4` attempt was killed by the WSL VM's Linux OOM killer (confirmed via `journalctl -k` on the failed boots) a few minutes into training; before changing anything, a calibration diagnostic on the completed `top_20_bits3`/`bits4` checkpoints found this re-export calibration-set-size **invariant**: full-data vs. this subset gave bit-identical quantizer parameters and all 105,088 validation-row integer outputs (Brevitas QAT's quantizer scale/zero-point are fixed learned power-of-two values, not statistics derived from the calibration range). A memory/time fix, not a methodology change |
| D8 | Epoch 1 is now built via a manual skorch-level `initialize()` + `fit()` (`src/fhe/compile/mlp.py::_init_sklearn_model`), never calling Concrete-ML's own `.fit()` for epoch 1. Every epoch of actual gradient training (weights, data, order, seed) is bit-for-bit unchanged | D7 fixed the redundant FINAL export, but a memory probe (isolated `train_only` vs. `export_only` experiments, capped and safety-killed before risking the VM) found epoch 1's ORIGINAL `model.fit()` call **also** triggers the same unconditional full-data export/quantize as a side effect (confirmed: `top_50`/`top_100`'s isolated export step alone exceeded 3,000 MB and was still climbing when safety-killed at ~12–24 s, while the real gradient-training pass added ~0 MB and finished in 11–20 s). Traced through the Concrete-ML/skorch source: skorch's OWN `NeuralNet.fit()` is *exactly* `initialize()` (no data touched) + `partial_fit()` (the real epoch) with no other side effect; the expensive export lives one layer above, in Concrete-ML's `QuantizedTorchEstimatorMixin.fit()`, which D8 simply never enters for epoch 1. Verified bit-identical (`base_module.state_dict()`, `qat_forward_logits`) on real `top_20` data before implementation, and by the regression below. A memory/time fix, not a methodology change — see the caveat below |

## 3. Concrete-ML facts, measured (`src/fhe/probes/mlp_api_probe.py` → `results/phase9_mlp/api_probe.json`, synthetic data)

- `module__n_outputs` and `module__input_dim` are **forbidden** as kwargs (set in `.fit` from the data); `random_state` is **rejected** — training is seeded with `torch.manual_seed` (verified reproducible).
- `n_layers=3` gives `input → 4·input → 4·input → 2`; e.g. 20 features → weights (80×20), (80×80), (2×80).
- **The accumulator ceiling is enforced by the compiler, not by refusing to compile:** with `n_accum_bits=16`, widths 2–8 all compiled on synthetic data (max integer bit-width 6 → 16). The binding constraint turned out to be **key size** (§5), not the accumulator.
- Integer level: `quantized_module_.quantized_forward(int_input, fhe="disable")` **equals** `circuit.simulate` exactly, and `post_processing(dequantize_output(ints))` reproduces `predict_proba`; so T1/T2 can be exact integer checks like XGBoost's. The input quantizer has `n_bits = n_a_bits`; the output quantizer 5 bits at `b=4`.
- **Concrete-ML's two-class `predict_proba` applies an element-wise sigmoid to both logits** (rows do not sum to 1). All Phase 9 probabilities are `sigmoid(logit_1 − logit_0)` from the dequantized logits (`src/fhe/compile/mlp.py::fraud_probability`), for the QAT model and the float twin alike.
- Serialization: `pickle` fails after compile (`ctypes objects containing pointers`); Concrete-ML's `dumps/loads` works **only with `callbacks="disable"`**, and reloads to exactly the same clear-quantized outputs. A model trained with `criterion__weight` serializes but `loads` then fails (`Unexpected key(s) in state_dict: "weight"`), so the loss weight is dropped for the dump only (`dump_model`); tested.
- `set_params(max_epochs=...)` changes only the Concrete-ML wrapper, **not the inner skorch net that trains**; the epoch count must be set on `model.sklearn_model` (found because a "zero-epoch refit" silently trained another epoch; tested in `test_restore_weights_...`).
- **Training cost:** every Concrete-ML `fit` call re-exports the network to ONNX and re-quantizes on the whole training set — a fixed ≈ 60–70 s per call on `top_20` — while one epoch itself is ≈ 20 s. The trainer therefore fits once, runs later epochs on the inner net (`partial_fit`), chooses the stopping epoch from the validation PR-AUC of the QAT (fake-quantized) forward, and exports the best weights once. The reported score is the **clear-quantized** validation PR-AUC of the final model (recorded beside the selection curve).
- Convergence check that fixed the pinned hyperparameters (one check on `top_20`, 4 bits, not a search): lr 0.003 with batch 2048 did **not** learn (validation PR-AUC 0.13); lr 0.001 with batch 512 did (0.31 → 0.34 in three epochs).

## 4. Data handoff (`src/fhe/export_mlp.py`, `configs/phase9/mlp_export.yaml`)

Every earlier handoff lacks the training labels (their models were trained on Windows), so `save_handoff` gained an optional `y_train` and an optional `reference_val_prob` (additive; existing handoffs keep their exact layout, tested). The three tier matrices are identical to the Phase 8 LR handoffs (asserted against the real files), NaN-free, with train fraud rate 3.42% and validation 3.89%. The MLP standardizes with a **train-only** scaler (mean/scale in `results/phase9_mlp/export/{tier}/mlp_{tier}_scaler.json`, hash in the manifest; a leak check is tested). The test partition is never referenced.

## 5. Feasibility (R5) — `results/phase9_mlp/feasibility.json`

Method: a short training run (2 epochs, 20,000 seeded train rows) on the real standardized tier data, compile against 3,000 train rows + each feature's min/max row, and — when the predicted key material is ≤ 2.4 GB — real key generation and one timed encrypted run. One process per candidate. A short-trained network only **indicates** the final circuit; the grid re-verifies every chosen configuration.

| Tier | b | PBS (TLUs) | Max integer bits | Bootstrap key | Key material | Compile | Real run | Outcome |
|---|---|---|---|---|---|---|---|---|
| top_20 | 2 / 3 / 4 | 400 / 560 / 640 | 6 / 7 / 10 | 687 / 747 / 888 MB | 0.83 / 1.01 / 1.26 GB | 8.7 / 8.6 / 9.5 s | 7.3 / 19.8 / 34.7 s | feasible; decrypted == simulated |
| top_50 | 2 / 3 / 4 | 1,200 / 1,600 / 2,000 | 7 / 9 / 11 | 489 / 451 / 908 MB | 0.61 / 0.65 / 1.32 GB | 8.0 / 8.6 / 8.4 s | 14.8 / 32.0 / 95.0 s | feasible; decrypted == simulated |
| top_100 | 2 / 3 / 4 | 2,800 / 3,200 / 4,400 | 7 / 9 / 11 | 656 / 451 / 1,114 MB | 0.77 / 0.65 / 1.63 GB | 11.1 / 19.9 / 19.6 s | 41.5 / 73.1 / 191.0 s | feasible; decrypted == simulated |
| all | 5 | 800 / 2,400 / 5,200 | 11 / 13 / 13 | 2.3 / 2.3 / 2.9 GB | 3.08 / 3.12 / 3.88 GB | ≈ 8–21 s | — | compiles; **infeasible here**: keys exceed the 3.8 GiB WSL memory (keygen not attempted) |
| all | 6 | 960 / 2,800 / 5,600 | 13 / 15 / 16 | 7.0 / 8.9 / 10.8 GB | 8.65 / 10.53 / 12.16 GB | ≈ 9–15 s | — | compiles; **infeasible here** (keys 8.7–12.2 GB) |
| all | 8 | — | — | — | — | — | — | **compile fails: `NoParametersFound`** at every tier |

(PBS/keys per tier are listed top_20 / top_50 / top_100 in the 5- and 6-bit rows.) The plan's expectation that the accumulator ceiling would block wider MLPs was **not** what limits this machine: 5-bit and 6-bit circuits compile; their bootstrap keys simply do not fit in 3.8 GiB, and 8 bits has no valid Concrete parameter set. Peak process RSS in the probe reached 3.43 GB for `top_100` at 4 bits (1.63 GB of keys), which is why the grid runner frees the training matrix before holding keys.

**Resulting grid (decision D1):** `b = 4` (largest feasible for all three tiers) and `b = 3` (next lower) × `top_20 / top_50 / top_100` = **6 configurations** (`configs/phase9/mlp_grid.yaml`). `b = 5, 6, 8` are recorded findings, not silently dropped. Measured probe latency of the 4-bit circuits (26–191 s per request) is 1–2 orders of magnitude below XGBoost's 14-bit runs (2,651–4,099 s) and 3–4 orders above LR's (8.6–31 ms) — an indication only; the grid measures it properly.

## 5a. Memory fix (D7): calibration-set-size invariance, and a remaining caveat

Every `mlp_top50_bits4` attempt (5 in total) was killed by the WSL VM's own Linux OOM killer (`journalctl -k` on the failed boots: `pt_main_thread` killed holding 3.18–3.64 GB, against a ~3.8 GiB VM cap with a full 1 GiB swap), not by anything on the Windows host. Concrete-ML's `NeuralNetClassifier.fit()` re-exports the network to ONNX and re-quantizes it (`quantize_module`) on whatever `X` it is given, in one unbatched forward pass that keeps every layer's output for every row; the trainer's original code passed the **full 380,815-row training set** to this call both at epoch 1 and again — purely to rebuild the quantized module from the best epoch's weights, with no actual training — in `restore_weights` at the end. That second call was redundant: the training set was never needed there, only enough rows to exercise the network's shapes.

**Before changing any code**, a read-only diagnostic loaded the completed `top_20_bits3`/`bits4` checkpoints and rebuilt `restore_weights`'s quantized module from the same trained weights, once with the full 380,815-row set (reproducing the saved checkpoint exactly) and once with the ~3,000-row `compile_calibration` subset already used to compile the FHE circuit. Result, both bit-widths: every quantizer parameter (input/output scale, zero-point, `n_bits`, offset) and all 105,088 validation-row clear-quantized integer outputs were **bit-for-bit identical**. This is expected for Brevitas QAT specifically: the quantizer scale/zero-point are fixed, learned power-of-two values, not statistics derived from the calibration data's range — unlike post-training quantization, where this would NOT be a safe substitution.

**Change made** (`src/fhe/compile/mlp.py`, `src/benchmark/phase9_grid.py`, `TRAINER_VERSION` bumped to `phase9-v2` to invalidate stale checkpoints): `train_qat_mlp`/`restore_weights` take an optional calibration subset, used only for the final re-export; every epoch of actual training is untouched, and omitting the argument reproduces the original full-data behavior exactly (tested). The grid runner now passes the same `compile_calibration` subset used for `compile_mlp`, so the quantized module and the FHE circuit are calibrated on identical rows.

**Regression proof** (`mlp_top20_bits3`, re-run through the real pipeline, `TRAINER_VERSION=phase9-v2`): status (`failed_accuracy_gates`), T0/T2/T1 exact-match gates, `quantized_full_metrics.pr_auc` (0.3924097013252142), `float_full_metrics.pr_auc` (0.4891828305207049), T3's decision agreement, confusion matrices, and every `circuit_stats` field (bootstrap count, key/keyswitch/secret key sizes, max integer bit-width) matched the values already on record from the original run exactly. `final_export_seconds` fell from a full-data cost of ≈ 37–70 s (still visible, unchanged, in the untouched `top_20_bits4` checkpoint) to **0.21 s**. `peak_rss_mb_after_compile` fell modestly, 3308 → 3240 MB (≈ 2%). Only wall-clock FHE latency (machine-load noise, not a correctness signal) differed. A process-level mishap limited this to a proof-by-reproduction rather than a byte-exact file diff: the pre-change checkpoint/results backup was written to the WSL VM's `/tmp` and lost to an automatic VM restart between the backup and the re-run (the same idle-shutdown behavior implicated in the original OOM investigation) — future before/after backups for this kind of check should go to the Windows-side filesystem or a git commit, not WSL `/tmp`.

**Caveat found by the D7 regression, resolved by D8 below:** epoch 1's `model.fit()` call (which both initializes the network and trains the first epoch) was untouched by D7 and still re-exported/re-quantized on the full training set as a side effect — confirmed directly in the untouched `top_20_bits4` checkpoint: epoch 1 took 102.4 s against 39.8 s and 44.8 s for epochs 2–3, an excess consistent with the documented ≈ 60–70 s full-data export cost.

## 5b. Memory fix (D8): epoch 1 no longer enters Concrete-ML's `.fit()` at all

**Measurement before changing anything:** a safety-watchdog-capped memory probe (kills the child process the instant its RSS crosses a conservative threshold, well under the VM's ~3.8 GiB cap, so nothing could OOM the VM again) isolated the two things epoch 1 does — a real gradient-training pass over the full data (`train_only`) and Concrete-ML's full-data export/quantize (`export_only`) — for `top_50` and `top_100` at 4 bits:

| Tier | `train_only` (real training, full data) | `export_only` (Concrete-ML's export/quantize, full data) |
|---|---|---|
| top_50 | 11.1 s, peak RSS 1,013 MB (≈0 MB above baseline) | killed by the safety cap at 3,051 MB after 24 s, still climbing (~330 MB/s), never finished |
| top_100 | 19.7 s, peak RSS 1,556 MB (≈0 MB above baseline) | killed by the safety cap at 3,083 MB after ~12 s of the export step, still climbing (~700 MB/s), never finished |

The real gradient training (already mini-batched, `batch_size=512`) is not the problem and never was. The spike is caused exclusively by Concrete-ML's full-data export/quantize — the same operation D7 already removed from the final restore step, but which epoch 1's `model.fit()` still triggers as an unconditional side effect.

**Why a clean fix exists:** tracing the installed Concrete-ML/skorch source (`concrete/ml/sklearn/qnn.py::NeuralNetClassifier.fit`, `concrete/ml/sklearn/base.py::QuantizedTorchEstimatorMixin.fit`/`BaseEstimator._fit_sklearn_model`, `skorch/net.py::NeuralNet.fit`) shows the expensive export lives entirely in Concrete-ML's own `.fit()`, one layer above skorch. Skorch's OWN `NeuralNet.fit()` is *exactly* `if not initialized_: self.initialize()` (builds the module/criterion/optimizer from already-set `module__*` attributes; consumes the RNG for weight init; touches no data) followed by `self.partial_fit(X, y)` (the real epoch) — nothing else. So epoch 1 can be built by replicating the three setup steps Concrete-ML's preamble does (`module__n_outputs`, `module__input_dim`, `criterion__weight`→tensor) and calling `model.sklearn_model.fit(X_train, y_train)` — skorch's own `.fit()` — directly, never entering Concrete-ML's `.fit()` (and therefore never triggering its export) at all.

**⚠️ Private API dependency** (`src/fhe/compile/mlp.py::_init_sklearn_model`, documented in full in its own docstring): this relies on three Concrete-ML/skorch internals that have no leading underscore (reachable) but are **not documented public API** — `model.sklearn_model_class`, `model.get_sklearn_params()`, and the `model.sklearn_model = model.sklearn_model_class(**params)` construction line copied from `BaseEstimator._fit_sklearn_model`. A future `concrete-ml`/`skorch` upgrade that changes this plumbing will make the D8 tests below **fail loudly** (bit-identity assertions), not silently diverge — that is the explicit purpose of testing it this thoroughly.

**Verification before implementation:** a scratch check on real `top_20` data built one model via the original `model.fit(X_train, y_train)` and one via the proposed manual construction + `model.sklearn_model.fit(X_train, y_train)`, same seed. Result: `base_module.state_dict()` bit-identical, `qat_forward_logits` bit-identical, and the proposed path's `_is_fitted`/`quantized_module_` confirmed Concrete-ML's `.fit()` never ran (10.8 s vs. 22.6 s).

**Tests added** (`tests/test_fhe_compile_mlp.py`): `_init_sklearn_model`+`initialize()` is deterministic given the same seed (no training at all); epoch 1 built via D8 matches the original epoch-1 path bit-for-bit in both weights and validation logits; D8's epoch 1 provably never triggers Concrete-ML's export (`_is_fitted` stays `False`, `quantized_module_` stays empty); and a full multi-epoch trajectory test against a preserved pre-D8 reference implementation (kept deliberately independent of the production code) proving identical epoch-by-epoch history, identical best epoch, and — after `restore_weights` — identical quantizer parameters and clear-quantized outputs.

**Regression proof** (`mlp_top20_bits3`, re-run through the real pipeline, `TRAINER_VERSION=phase9-v3`, backed up to the Windows filesystem this time): **every** accuracy/correctness/quantization value matched the D7 (pre-D8) reference bit-for-bit — `val_logits.npy` identical across all 105,088 rows, the full 22-epoch training history identical, `best_epoch`/`epochs_run`/`stopped_early` identical, the float twin identical, T0/T2/T1 exact-match gates all identical, T3's decision agreement/PR-AUC-drop/quantized-and-float PR-AUC identical, all 8 plaintext metric fields identical for both quantized and float, and every `circuit_stats` field (bootstrap count, key sizes, bit widths, layer shapes) identical. Only timing and memory changed, exactly as intended:

| | D7 (before D8) | D8 (after) |
|---|---|---|
| epoch-1 seconds | 20.62 s | **9.42 s** (now matches a normal epoch — epoch 2 was 9.26–10.06 s in both runs) |
| `peak_rss_mb_after_compile` | 3,240 MB | **971 MB** (−70%) |
| `peak_rss_mb` (end of run) | 3,240 MB | **1,671 MB** (−48%) |

For `top_20` this closes the loop D7 left open: D7 alone only reduced peak RSS by ≈2% because epoch 1's un-addressed export was already the dominant driver; D8 removes that driver and the measured peak drops by roughly the same ~2.3 GB the row-scaling estimate in §1 attributed to it. Since the memory probe above showed `top_50`/`top_100`'s `export_only` step is what exceeds the VM's cap while `train_only` costs ≈0 MB regardless of tier width, D7+D8 together are expected to bring `top_50`/`top_100` to roughly the same low-hundreds-of-MB epoch-1 footprint measured for `top_20` here — **this is the expectation the actual `top_50_bits4` run will confirm or refute**, not yet attempted.

## 5c. T2 correctness finding: accumulator overflow on one out-of-calibration-range row (`mlp_top50_bits3`)

`mlp_top50_bits3`'s T2 gate (`circuit.simulate` vs. the clear integer computation, 5,000 seeded validation rows, exact match required) recorded exactly **1 mismatching row out of 5,000** (T2-subset index 4667, validation row `X_val[97498]`), zero decision flips. This is preserved as a genuine T2 **FAIL** — the gate and its tolerance were not changed to accommodate it.

**Root cause, fully localized and reproduced deterministically from saved artifacts (no retraining):**

- The clear/disable path (`[-13, 19]`) and `circuit.simulate()` (`[-4, 10]`) disagree by 9 in each output integer — not a rounding blip. Ten repeated `simulate()` calls after a fresh recompile all returned the identical `[-4, 10]`, matching the cached value from the original run exactly: the divergence is **fully deterministic**, not simulation noise.
- Neighboring T2 rows all match exactly; row `97498` alone has 7 of its 50 quantized inputs saturated at the input quantizer's clip boundary, with raw standardized feature values up to **12.5 standard deviations** from the mean.
- Using Concrete-ML's `debug=True` forward pass to inspect every intermediate integer value: the second hidden layer's accumulator (`fc1/Gemm`) reaches **139** for this row, against **37** for a normal neighboring row. `max_integer_bit_width` for this compiled circuit is **8** (representable signed range ≈ ±127) — the true value exceeds what the compiled circuit can represent. The exact/disable path uses unbounded Python integers and computes 139 correctly; the compiled circuit (sized from the ~3,063-row `compile_calibration` set, not from this validation row) cannot, and both the real circuit and its faithful `simulate()` emulation produce a different, wrapped result instead.
- **Why calibration missed it**: `compile_calibration`'s `include_extremes` guarantees each *individual* feature's train min/max row is included, not *combinations* of several simultaneously-extreme features — exactly what drives this row's unusually large accumulator.

**Verified unaffected**: T1's 2 real-hardware sample rows (`32148`, `13560`) do not include row `97498`, and T1's own independent exact-match check already passed. T3 already fails on accuracy grounds regardless. The completed `mlp_top50_bits3` result is accurate and requires no correction — T2's `FAIL` and its exact mismatch count are the correct, honest record.

**Methodology implication, not a code bug**: T2's check is working as designed. This surfaces a calibration-coverage limitation (individual-feature extremes don't guarantee combined-extreme coverage) worth documenting, not an error in the T2 gate itself. The same mechanism is expected to recur, plausibly more often, in wider architectures (more accumulator terms) — consistent with `top_100`'s circuits needing higher `max_integer_bit_width` (10–11) than `top_50`'s (8) at the same bit-width.

## 6. Gates

T0 exact checkpoint reload (D3); T3 QAT-vs-float-twin at Phase 8's bars (D2); T2 `circuit.simulate` vs the clear integer outputs, **exact**, on 5,000 seeded validation rows (simulate costs 0.02–0.10 s per row, so all 105,088 rows would take up to ≈ 3 h per configuration); T1 real encrypt→run→decrypt on the 2-row stratified sample, **exact**; latency = 5 repeated executions of validation row 32,148 (the row Phase 8 used), mean ± sample std, keygen and compile separate. A configuration whose keys exceed `max_key_material_gb` (2.4) is recorded as `infeasible_key_memory` with its plaintext gates instead of being attempted.

## 7. Results

All 6 grid cells attempted; every number below is read from `results/phase9_mlp/*/metrics.json`, cross-checked by `src.analysis.phase9_results.validate_all()` (schema, gate consistency, and reported means/std against the raw per-trial arrays — 6/6 configurations found, zero problems) and combined with Phase 8 in `src.analysis.phase9_pareto` (`results/phase9_mlp/pareto/`).

**Provenance note**: `mlp_top20_bits4` was trained before decisions D7/D8 existed and was never re-run under them (only `mlp_top20_bits3` served as the regression-proof configuration for both fixes). Its accuracy/correctness numbers are unaffected — D7/D8 were proven bit-identical to the pre-fix training path — but its `peak_rss_mb` (3,530 MB) reflects the *old*, pre-D7/D8 full-data-export memory profile, not the fixed path every other cell in this grid used.

### 7.1 Six-cell outcome

| Config | Status | T0 | T3 | T2 | T1 |
|---|---|---|---|---|---|
| `mlp_top20_bits3` | `failed_accuracy_gates` | PASS | FAIL | PASS (0/5000) | PASS |
| `mlp_top20_bits4`* | `failed_accuracy_gates` | PASS | FAIL | PASS (0/5000) | PASS |
| `mlp_top50_bits3` | `failed_accuracy_gates` | PASS | FAIL | **FAIL (1/5000, §5c)** | PASS |
| `mlp_top50_bits4` | `infeasible_key_memory` | PASS | FAIL | not run | not run |
| `mlp_top100_bits3` | `infeasible_key_memory` | PASS | FAIL | not run | not run |
| `mlp_top100_bits4` | `infeasible_key_memory` | PASS | FAIL | not run | not run |

\*pre-D7/D8 checkpoint, see the provenance note above.

**T0 passes in all 6.** **T3 fails in all 6** — no configuration in this grid reaches the ≥0.99 decision-agreement / ≤0.01 PR-AUC-drop bar; decision agreement stays in a narrow 0.966–0.979 band regardless of tier or bit-width. **T2 fails in exactly one cell** (§5c, a fully diagnosed, deterministic, isolated finding — preserved as FAIL). **T1 passes in every cell it ran.**

### 7.2 Accuracy

| Config | Quantized PR-AUC | Float-twin PR-AUC | Decision agreement | PR-AUC drop |
|---|---|---|---|---|
| `mlp_top20_bits3` | 0.3924 | 0.4892 | 0.9788 | 0.0968 |
| `mlp_top20_bits4` | 0.4070 | 0.4892 | 0.9789 | 0.0822 |
| `mlp_top50_bits3` | 0.4173 | 0.4696 | 0.9703 | 0.0523 |
| `mlp_top50_bits4` | 0.4324 | 0.4696 | 0.9656 | 0.0372 |
| `mlp_top100_bits3` | 0.4494 | 0.4908 | 0.9764 | 0.0414 |
| `mlp_top100_bits4` | 0.4392 | 0.4908 | 0.9706 | 0.0516 |

Both quantized and float PR-AUC rise with feature count (as in Phase 8), but the QAT-vs-float gap that drives T3's failure never closes.

### 7.3 Circuit statistics and key material

| Config | Max int. bit-width | PBS | Key-switch | Key material | Compile |
|---|---|---|---|---|---|
| `mlp_top20_bits3` | 8 | 560 | 640 | 0.65 GB | 5.2 s |
| `mlp_top20_bits4`* | 10 | 640 | 720 | 2.03 GB | 7.4 s |
| `mlp_top50_bits3` | 8 | 1,200 | 1,400 | 1.19 GB | 5.7 s |
| `mlp_top50_bits4` | 11 | 1,800 | 2,000 | **4.44 GB** | 6.0 s |
| `mlp_top100_bits3` | 10 | 2,400 | 2,800 | **3.24 GB** | 9.1 s |
| `mlp_top100_bits4` | 11 | 3,200 | 3,600 | **4.80 GB** | 8.1 s |

**Only `mlp_top20_bits3`, `mlp_top20_bits4`, and `mlp_top50_bits3` fit under the 2.4 GB `max_key_material_gb` gate**, unchanged throughout Phase 9. Every 4-bit configuration at `top_50`/`top_100`, and even 3-bit at `top_100`, exceeds it — recorded as `infeasible_key_memory` with the exact key size and circuit statistics that explain it, per `docs/instructions.md`'s "never silently drop a configuration" rule. Key material scales faster than linearly with tier width at a fixed bit-width (0.65 → 1.19 → 3.24 GB for bits3 across `top_20/50/100`).

### 7.4 Memory and stability (decisions D7+D8)

| Config | Peak RSS | Max swap | Same WSL boot throughout? |
|---|---|---|---|
| `mlp_top20_bits3` | 1,671 MB | 0 | yes |
| `mlp_top20_bits4`* | 3,530 MB | — | yes |
| `mlp_top50_bits3` | 2,856 MB | 2 MB (trivial) | yes |
| `mlp_top50_bits4` | 1,312 MB (compile only) | 0 | yes |
| `mlp_top100_bits3` | 1,934 MB (compile only) | 0 | yes |
| `mlp_top100_bits4` | 1,889 MB (compile only) | 0 | yes |

Every real run in this grid (bar the pre-D7/D8 `top_20_bits4`) completed on one continuous WSL boot with no kernel OOM kill and no meaningful swap use — the repeated, direct confirmation that D7+D8 resolved the OOM that originally blocked `mlp_top50_bits4` (§5a, §5b), across the whole remaining grid, not only the configuration the fix was first proven on.

### 7.5 Latency (measured only where FHE ran)

| Config | Mean total | Std |
|---|---|---|
| `mlp_top20_bits3` | 5.17 s | 0.59 s |
| `mlp_top20_bits4`* | 72.61 s | 5.67 s |
| `mlp_top50_bits3` | 17.13 s | 1.14 s |

The 3 `infeasible_key_memory` configurations were never attempted under FHE — no latency exists for them, by design.

### 7.6 Combined Pareto (LR + XGBoost + quantized MLP; `results/phase9_mlp/pareto/`)

Full tables: `combined_tables.md`; machine-readable: `combined_table.csv`/`.json`, `combined_frontiers.json`; figures: `fig_combined_accuracy_vs_latency.png`, `fig_combined_accuracy_vs_memory_ciphertext.png`. Built from `src.analysis.phase9_pareto`, which validates both Phase 8 and Phase 9 results before analyzing and uses only completed (`passed`/`failed_accuracy_gates`) rows for the frontiers — the 3 infeasible MLP configurations are listed in their own "without an FHE latency" table, never dropped.

- **No MLP configuration passes T3**, so none appears on the "T3-passing only" frontier — every T3-passing point on every cost axis (latency, memory, ciphertext) is LR-16-bit or XGBoost-14-bit. The "all configurations" latency frontier does admit `mlp_top20_bits3` on peak-RSS terms only (it has lower memory than XGBoost while beating no LR configuration on accuracy at that memory level).
- **MLP latency sits between LR and XGBoost**, by 1–2 orders of magnitude in both directions: `mlp_top20_bits3` is ~604× slower than `lr_top20_bits16` (8.6 ms) but ~792× faster than `xgboost_top20_bits14` (4,099 s); `mlp_top50_bits3` is ~1,112× slower than `lr_top50_bits16` but ~155× faster than `xgboost_top50_bits14`.
- **Bit-width effect, `top_20`, 3→4 bits (the only pair with FHE data on both sides)**: PR-AUC 0.392 → 0.407, latency 14.0×, key material 3.1×, peak RSS 2.1× — accuracy gain is small; cost grows fast.
- **Prior-art context** (`docs/prd.md` §11's ~296 ms neural-network figure, comparison only — that work's network/hardware/definition of latency are unverified here): measured MLP latency is 17–245× that figure across the three cells with FHE data.

### 7.7 FR5/D5: plaintext seed-stability evaluation (closes PRD FR5)

`phase9_grid.py --plaintext` (decision D5): each of the 3 feature tiers, QAT model + float twin, trained independently at seeds 42/43/44 at the primary bit-width (4), no FHE. `results/phase9_mlp/plaintext/{tier}/metrics.json` and `summary.json`. Seed 42 shares its checkpoint fingerprint with the `mlp_top{20,50,100}_bits4` grid cells: for `top_50` and `top_100` this meant the grid's own already-D7/D8-fixed checkpoint was reused directly; for `top_20`, whose grid checkpoint predates D7/D8 (§7.4's caveat, unchanged), this instead **retrained it from scratch under the current D7/D8 code**. That retrain reproduced the original pre-D7/D8 `mlp_top20_bits4` PR-AUC exactly — quantized `0.40697029219832703` and float `0.4891828305207049`, bit-for-bit identical to both runs — a live, unplanned confirmation of D7/D8's bit-identical claim on the one grid cell it had not previously been checked against directly (only proxied through the `top_20_bits3` regression).

| Tier | QAT PR-AUC (mean ± std, 3 seeds) | Float-twin PR-AUC (mean ± std, 3 seeds) | QAT − float, per seed |
|---|---|---|---|
| `top_20` | 0.4049 ± 0.0045 | 0.4852 ± 0.0045 | −0.082, −0.072, −0.086 |
| `top_50` | 0.4318 ± 0.0038 | 0.4761 ± 0.0057 | −0.037, −0.052, −0.043 |
| `top_100` | 0.4455 ± 0.0111 | 0.4868 ± 0.0083 | −0.052, −0.034, −0.038 |

**Finding: the QAT-vs-float gap is systematic, not seed noise.** In every tier, the seed-to-seed standard deviation (0.004–0.011 PR-AUC) is 4–20× smaller than the QAT-minus-float gap itself (0.034–0.086 PR-AUC), and the gap has the same sign in all 9 runs. This corroborates §0/§7.2's T3 finding (decision agreement 0.966–0.979, never reaching the ≥0.99 bar) from an independent angle: 4-bit QAT's accuracy shortfall relative to its float twin is a property of the quantization itself at this architecture/bit-width, not a training-run artifact that a different seed would erase. Training-run variance is real but small (±0.004–0.011 PR-AUC) and does not, by itself, explain why T3 fails.
