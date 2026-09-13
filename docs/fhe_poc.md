# Phase 5 — FHE Proof of Concept (Logistic Regression · `top_20` · `n_bits=8`)

Implements `docs/plan.md` Phase 5: get one model (Logistic Regression, the
simplest) compiling and running correctly under Concrete-ML end-to-end,
before the full grid exists. Run as:

```bash
# Windows .venv (plaintext side)
python -m src.fhe.export --config configs/phase5/lr_poc.yaml

# WSL2 FHE venv (Concrete-ML side) -- from the project root
~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.poc --config configs/phase5/lr_poc.yaml
```

**Status: PARTIALLY COMPLETE.** `docs/plan.md`'s literal exit criterion --
"at least one full encrypt→infer→decrypt round trip completes correctly" --
**is met** (T1, below), and independently re-verified after a config fix
(Sec.9.4). This session's own additional acceptance gate on quantization
accuracy (T3) **failed at `n_bits=8`**, with a measured, well-characterized
cause (Sec.6), **and failed again at `n_bits=16`** (the empirically
confirmed maximum bit-width Concrete-ML can compile for this circuit
shape) in a single controlled follow-up experiment (Sec.9) -- ruling out
"insufficient quantization resolution" as a fix. Per this session's
explicit instruction, both failures are reported, not silently worked
around, retrained, or loosened past. Phase 5 is not declared fully
complete; Sec.8/Sec.9 lay out the options for how to proceed.

## 1. Objective and scope

Quantize the committed Phase 4 `top_20` Logistic Regression at `n_bits=8`,
compile it with Concrete-ML, and run **real encrypted inference**: keygen
→ encrypt → run on ciphertext → decrypt. Prove, with layered checks, that
the FHE result matches the quantized clear model exactly, and that the
quantized model stays within an approved tolerance of the float plaintext
model. LR only, `top_20` only, `n_bits=8` only -- no XGBoost, no other
tiers, no bit-width sweep, no client/server, no benchmarking (all
explicitly out of scope for this phase, per `docs/plan.md` and this
session's instructions).

## 2. Environment split and the cross-environment handoff

Concrete-ML 1.9.0 only installs under WSL2/Linux (`docs/environment.md`),
with its own pinned scikit-learn (1.5.0) / numpy (1.26.4) -- materially
older than the Windows `.venv`'s (1.9.0 / 2.5.3), which is where the Phase
4 plaintext model was fit and pickled. `docs/environment.md` explicitly
deferred resolving this to Phase 5.

**Resolution: never unpickle across the boundary.** `src/fhe/export.py`
(Windows) extracts the fitted `StandardScaler`+`LogisticRegression`'s exact
numeric parameters (mean, scale, coef, intercept, classes, `C`) into plain
JSON, and the calibration (train) / correctness-check (val) feature
matrices into a pickle-free `.npz` (`allow_pickle=False` enforced on load).
`src/fhe/handoff.py` is the only module that reads/writes/verifies this
handoff, depends on nothing but numpy + stdlib, and is imported unchanged
by both `src/fhe/export.py` (Windows) and `src/fhe/poc.py` (WSL).
`src/fhe/handoff.py::rebuild_pipeline_predict_proba` reimplements the
fitted pipeline's `predict_proba` from those parameters using only numpy
arithmetic -- no scikit-learn estimator object crosses the boundary at all.

Every handoff is integrity-checked, layered exactly like Phase 3→4's
gates: the npz's SHA-256 is verified against the manifest before it is
read (`load_handoff`), and the Phase 4 tier LR is re-verified against its
own committed `metrics.json` (config-hash match + a reload-no-refit
PR-AUC check, tolerance `1e-6`) before export (`src/fhe/export.py::_verify_phase4_tier_integrity`)
-- the same "reload the saved artifact" pattern
`src/features/pipeline.py::_verify_phase3_integrity` established for
Phase 3→4.

## 3. Why Logistic Regression, and why `top_20`

- **Simplest encrypted computation**: the compiled circuit only needs to
  compute a linear score (dot product + bias) -- additions and
  plaintext-constant multiplications are cheap in TFHE.
- **Sigmoid and thresholding stay client-side**, running in the clear
  after decryption (`post_processing`) -- confirmed empirically (Sec.7)
  that the compiled circuit needs **zero** programmable bootstraps (PBS),
  the dominant FHE cost driver for nonlinear operations.
- **A deterministic, well-understood plaintext reference** (`lbfgs`,
  converged, exactly reproducible per Phase 3/4) means any divergence is
  attributable to quantization or FHE, not model instability.
- **`top_20`** is the smallest committed tier -- cheapest to compile,
  least likely to hit the WSL2 3.8 GiB memory ceiling
  (`docs/environment.md`).

## 4. End-to-end flow

```
WINDOWS .venv (sklearn 1.9 / numpy 2.5)              WSL2 FHE venv (concrete-ml 1.9 / sklearn 1.5 / numpy 1.26)
────────────────────────────────────                 ─────────────────────────────────────────────────────────
src/fhe/export.py                                    src/fhe/poc.py
 load_phase2_features, load_tiers (top_20,             load_handoff (npz SHA-256 + manifest verified)
   HASH-VERIFIED, exact tiers.json order)              T0: rebuild_pipeline_predict_proba(params, X_val)
 _verify_phase4_tier_integrity (config hash              == reference_val_prob  (max diff 1.11e-16)
   + reload-no-refit PR-AUC match)                     build_concrete_lr(params, X_train, n_bits=8)
 extract_lr_pipeline_params -> params JSON               (from_sklearn_model, calibrated on X_train)
 X_train/X_val UPCAST TO FLOAT64 (Sec.6.1) ──►         compile(X_train) -> circuit; circuit_stats
   reference_val_prob = model.predict_proba(X_val)     T3: predict_proba(X_val, fhe="disable") vs. float
 save_handoff -> lr_top20.npz + handoff_manifest.json    reference, FULL val (105,088 rows)  -- FAILED (Sec.5)
                                                        T2: predict_proba(X_val, fhe="simulate") vs. "disable",
                                                          FULL val -- PASSED, exact (105,088/105,088)
                                                        E5/T1: stratified 100-row sample (25 fraud/75 legit):
                                                          quantize_input -> keygen -> encrypt -> run ->
                                                          decrypt -> dequantize_output -> post_processing
                                                          vs. fhe="simulate" -- PASSED, exact (100/100)
```

Calibration uses the train partition only (380,815 rows). All correctness
checks use val only (105,088 rows for T2/T3; a 100-row stratified sample
for T1). **The test partition is never read by either side** -- `src/fhe/export.py`
never even slices `features.X_test`/`features.y_test` beyond `len()`, and
`src/fhe/poc.py` never receives them at all (the handoff npz only ever
contains train/val arrays). `metrics.json` records
`"test_partition_touched": false`.

## 5. Results for every acceptance gate

Real run: `git commit f8e8bfe` (Windows export) / handoff manifest
`npz_sha256 c6f5c222...`, `phase4_tiers_membership_hash ffe76f3b2c4af9ca...`
(the committed Phase 4 hash, unchanged). Full detail:
`results/phase5_fhe_poc/{correctness_report,circuit_stats,metrics,provenance}.json`.

| Gate | Compares | Result | Tolerance | Status |
|---|---|---|---|---|
| **T0** (transfer) | WSL-reconstructed float model vs. Windows reference, full val | max abs diff **1.11e-16** | ≤1e-9 | **PASSED** |
| **T2** (simulation) | `fhe="simulate"` vs. `fhe="disable"`, full val (105,088 rows) | **105,088 / 105,088 exact** | 100% exact | **PASSED** |
| **T1** (execution) | real encrypt→run→decrypt vs. `fhe="simulate"`, 100-row stratified sample | **100 / 100 exact** | 100% exact | **PASSED** — this is `docs/plan.md`'s literal exit criterion |
| **T3** (quantization) | `fhe="disable"` (quantized) vs. float reference, full val | decision agreement **0.9576**; PR-AUC 0.3484 → **0.0554** (drop 0.2930) | ≥0.99 agreement, ≤0.01 PR-AUC drop | **FAILED** |

T3's full detail:

| Metric | Float (reference) | Quantized (`n_bits=8`) |
|---|---|---|
| PR-AUC | 0.3484 | 0.0554 |
| ROC-AUC | 0.8219 | 0.5701 |
| Confusion matrix @ threshold 0.7086 | TN=98,739 FP=2,259 / FN=2,608 TP=1,482 | TN=98,263 FP=2,735 / FN=3,754 TP=336 |

The quantized model's ROC-AUC (0.570) is barely above random (0.5) and its
PR-AUC (0.055) is close to the ~0.035 random/base-rate floor this project
uses elsewhere as a sanity floor -- 8-bit quantization at this
calibration essentially destroyed the model's discriminative power, not
merely degraded it slightly.

## 6. Two real behaviors verified empirically

### 6.1 sklearn's `StandardScaler.transform` preserves input dtype -- a real, measured precision boundary

While building `src/fhe/export.py`'s test suite
(`tests/test_fhe_export.py::test_export_params_reproduce_reference_probabilities`),
comparing `rebuild_pipeline_predict_proba`'s pure-float64 arithmetic
against `model.predict_proba(X_val)` on the real (float32, per Phase 2's
`downcast_floats: true`) feature matrix failed at the originally-planned
`1e-9` tolerance, off by up to `2.85e-7`. Investigated empirically rather
than assumed:

```
scaler.mean_.dtype, scaler.scale_.dtype  -> float64, float64
scaler.transform(X_float32).dtype        -> float32   <-- confirmed directly
```

`StandardScaler.transform` returns output in the **same dtype as its
input**, silently downcasting its float64-fitted `mean_`/`scale_` to
float32 before the subtract/divide when given float32 data. This is not a
transfer bug -- it is how scikit-learn actually evaluates this exact
pipeline shape on the real (float32) Phase 2 data, and it would blow
through a strict T0 tolerance for a reason having nothing to do with the
cross-environment handoff T0 exists to check.

**Fix (approved)**: `src/fhe/export.py` upcasts `X_train`/`X_val` to
float64 immediately after loading, before computing `reference_val_prob`
-- making "the reference" unambiguously float64-precise by construction,
matching `rebuild_pipeline_predict_proba`'s arithmetic. This changes
`reference_val_prob`'s exact numeric value from what Phase 4 originally
committed (float32-precision) by ~1e-7 -- immaterial to any reported
PR-AUC digit, and recorded here as a deviation (Sec.10), not silently
absorbed. With this fix, T0's real measured diff is `1.11e-16` -- close
to true float64 machine epsilon, not merely "under tolerance."

### 6.2 The compiled circuit needs zero programmable bootstraps -- confirmed, not assumed

`src/fhe/compile/linear.py`'s design assumed a linear model's affine
arithmetic (dot product + bias) needs no PBS, since the sigmoid and
thresholding run client-side in `post_processing` after decryption. This
was verified directly on both the synthetic PoC model
(`tests/test_fhe_lr_poc.py::test_compile_produces_a_circuit_with_stats`)
and the real `top_20` circuit: `programmable_bootstrap_count == 0` in both
cases, `size_of_bootstrap_keys_bytes == 0`, `p_error`/`global_p_error`
≈5.3e-13 (real run) -- consistent with a circuit that has no probabilistic
bootstrapping to introduce execution noise. This is exactly why T1 (real
FHE execution) and T2 (simulation) were expected to match exactly, not
merely closely, and both did (Sec.5) -- a zero-PBS circuit has nothing
that could make a real encrypted execution diverge from its noiseless
simulation.

## 7. The FHE PoC's security-claim limits (vs. the eventual Phase 10 client/server)

The PoC runs in **one process, one trust domain**: the secret key,
plaintext, and ciphertext all share memory. It proves *computational
correctness* -- the circuit computes the right answer on ciphertext -- and
explicitly does **not** demonstrate this project's security claim ("the
server never sees plaintext or the secret key", `docs/architecture.md`
Sec.8). The explicit step-by-step API used for T1 maps onto the eventual
client/server roles:

- **Client-side operations** (`src/fhe/poc.py::_explicit_round_trip`):
  `quantize_input`, `circuit.keygen`, `circuit.encrypt`, `circuit.decrypt`,
  `dequantize_output`, `post_processing` (sigmoid + threshold).
- **Server-side operation**: `circuit.run` only.

`FHEModelDev`/`FHEModelClient`/`FHEModelServer` (Concrete-ML's deployment
API, confirmed present in the installed 1.9.0), evaluation-key
serialization, a separate FastAPI process, and measurement of
network/serialization overhead are all deferred to Phase 10, per
`docs/plan.md` ordering and this session's explicit constraint.

## 8. What T3's failure means, and the options

`docs/eda.md` Sec.5 anticipated exactly this risk in Phase 1: *"quantization
calibration should not naively use raw min/max for these heavy-tailed
columns -- expect this to force either clipping/winsorization or a
non-uniform quantization scheme."* Measured directly on the real `top_20`
calibration data (train partition, scaled): several columns have their
1st-99th percentile occupying a tiny fraction of their full min-max range
-- e.g. `C8` 0.42%, `V187` 0.46%, `V294` 1.01%, `V308` 1.65%,
`TransactionAmt` 3.41% (`docs/eda.md`'s skew-14.37 finding, now directly
implicated in an accuracy failure, not just a documented risk).
`Concrete-ML.from_sklearn_model`'s uniform `n_bits=8` quantizer is
calibrated from each feature's raw min/max, so these few extreme-outlier
columns each consume nearly all 256 quantization levels on their sparse
tail, crushing the dense bulk of real values into a handful of
indistinguishable buckets -- destroying most of their signal before the
circuit ever runs.

This is a genuine, well-characterized research finding (`CLAUDE.md` §18:
*"an infeasible configuration is a valid research finding"*), not a code
defect -- T0/T1/T2 all passing confirms the handoff, compilation, and FHE
execution machinery are all correct; only the *accuracy* of `n_bits=8`
uniform quantization on this specific feature set is the problem.

**Options, none applied without approval:**
1. Raise `n_bits` (Concrete-ML supports higher bit-widths for linear
   models) and re-measure T3 -- more resolution, same calibration scheme.
   **Tried (Sec.9): failed.** `n_bits=16`, the empirically-confirmed
   maximum this circuit shape can compile (18+ raises `NoParametersFound`),
   left T3 essentially unchanged (decision agreement 0.9576→0.9614,
   PR-AUC drop 0.2930→0.2901). This option is exhausted for this
   circuit/tier: there is no higher compilable `n_bits` left to try, and
   the two measured points show resolution alone was never going to close
   a ~0.29 PR-AUC-drop gap.
2. Clip/winsorize the calibration data before compiling (a model-input
   definition change, would need to be documented as a deviation and
   would also need to be applied consistently to any later inference
   input).
3. Investigate a per-feature or non-uniform calibration approach, if
   Concrete-ML's API supports one for linear models.
4. Report `n_bits=8` uniform quantization as infeasible for this feature
   mix at this tier, and let Phase 6/8's bit-width sweep discover the
   actually-required precision empirically, rather than hand-picking a
   fix now.

## 9. Follow-up: raising `n_bits` to 16, and a config bug found along the way

This session approved testing Sec.8's Option 1 (raise `n_bits`) as a single
controlled experiment: same committed Phase 4 `top_20` LR, same handoff,
same threshold, same tolerances, same everything except the bit-width.

### 9.1 Probe: the highest `n_bits` this circuit shape can compile

Before spending a real ~10-20 minute run, a synthetic toy model shaped like
`top_20` (heavy-tailed first column) was compiled at `n_bits` in
`{8, 10, 12, 14, 16, 18, 20, 24}` in WSL -- read-only, no real artifacts
touched. Result: 8/10/12/14/16 all compiled, `programmable_bootstrap_count
== 0` at every one of them; 18/20/24 all failed with
`RuntimeError: NoParametersFound`. `n_bits=16` was selected as the single
real follow-up run: the empirically-confirmed ceiling for this circuit
shape, and since PBS stayed 0 at every tested width there was no PBS-cost
reason to test an intermediate value first.

### 9.2 Real `n_bits=16` run: T3 fails again, essentially unchanged

`configs/phase5/lr_poc.yaml`'s `n_bits` was changed to 16 (nothing else)
and the real PoC was rerun in WSL. Results, from the run's own structured
log output (the on-disk `results/phase5_fhe_poc/*.json` files were **not**
overwritten by this run -- Sec.9.3):

| Gate | `n_bits=8` (Sec.5) | `n_bits=16` | Status |
|---|---|---|---|
| T0 (transfer) | max abs diff 1.11e-16 | max abs diff 1.11e-16 | **PASSED**, both |
| Compile | 0 PBS, 129.7s | 0 PBS, 97.0s | zero-PBS, both |
| T3 (quantization) | agreement 0.9576, drop 0.2930 | agreement 0.9614, drop 0.2901 | **FAILED**, both |
| T2 (simulation) | 105,088/105,088 exact | 105,088/105,088 exact | **PASSED**, both |
| T1 (execution) | 100/100 exact | **did not execute** (Sec.9.3) | crashed, not a gate failure |

T3 full detail at `n_bits=16`: quantized PR-AUC **0.0583** (float 0.3484),
quantized ROC-AUC **0.5836** (float 0.8220), max \|Δprob\| 1.0, mean
\|Δprob\| 0.3541.

**Finding: raising `n_bits` to the compilable ceiling does not resolve
T3.** Decision agreement improved by only 0.0038 (0.9576→0.9614) and the
PR-AUC drop improved by only 0.0029 (0.2930→0.2901) -- both still far
outside tolerance, after *doubling* the available quantization levels.
This directly confirms Sec.8's diagnosis: the problem is not that 256
levels is too coarse in an absolute sense, it is that a *uniform
min/max-calibrated* quantizer wastes nearly all of its levels -- 256 or
65,536, it makes almost no difference -- on the same few heavy-tailed
columns' sparse extremes. Doubling total levels does nothing to fix how
badly those levels are *allocated* across each feature's real value
distribution.

### 9.3 Config bug: T1 crashed with `KeyError: 'seed'`

The `n_bits=16` run crashed at `src/fhe/poc.py:180`
(`_stratified_sample_positions(..., config["seed"])`) before T1 could
execute, because `configs/phase5/lr_poc.yaml`'s top-level `seed: 42` key
was missing at the time -- inadvertently dropped while editing the file
for the `n_bits=16` comment block. (The file is untracked/uncommitted, so
there is no git history that pinpoints the exact edit that dropped it.)
Because the crash happened before `src/fhe/poc.py`'s write block -- which
only runs after T1 completes -- **no file under `results/phase5_fhe_poc/`
was overwritten**: `metrics.json` still read `"n_bits": 8` immediately
after the `n_bits=16` run finished. The `n_bits=16` T0/T3/T2 numbers in
Sec.9.2 come only from that run's structured log, never from a results
file, and no Phase 5 artifact was corrupted or lost.

This is a code/config defect, not a correctness finding, and per this
session's explicit instruction it was not silently patched and rerun in
the same breath -- it was reported, and the fix + re-verification (9.4)
were done as an explicitly separate, approved step.

### 9.4 Fix and independent T1-only re-verification (`n_bits=8`, seed restored)

`configs/phase5/lr_poc.yaml` was reverted to `n_bits: 8` -- the `n_bits=16`
experiment having failed to resolve T3, and 16 being the confirmed ceiling
with no higher value left to try -- and `seed: 42` was restored. A
standalone script re-used `src/fhe/poc.py`'s own
`_stratified_sample_positions`, `_explicit_round_trip`, and
`t1_execution_check` functions unchanged, with the fixed config: it
compiled the `n_bits=8` circuit fresh and ran **only** T1's own
sample-level steps (`predict_proba(fhe="simulate")` on the 100-row sample,
then the explicit `quantize_input → keygen → encrypt → run → decrypt →
dequantize_output → post_processing` round trip). It never called
`fhe="disable"` or `fhe="simulate"` on the full val partition, so **T3 and
T2 were not recomputed** by this check -- the T3/T2 values reported for
`n_bits=8` throughout this document (Sec.5) remain the original,
untouched measurement.

Result: exact match, 100/100 rows, `max_abs_diff: 0.0` -- **byte-identical**
to the T1 result already recorded in
`results/phase5_fhe_poc/correctness_report.json`. This independently
confirms the originally-recorded T1 result is genuinely reproducible with
the config bug fixed, rather than an artifact of whatever seed value
happened to be in place during the original successful run. `metrics.json`
and `correctness_report.json` were **not** regenerated or overwritten by
this re-verification -- they still hold exactly the same `n_bits=8`
T0/T1/T2/T3 report as before this follow-up.

## 10. Deviations from `docs/plan.md`

1. **The T0 reference is computed on float64-upcast data**, not the raw
   float32 Phase 2 matrix (Sec.6.1) -- changes `reference_val_prob`'s
   exact value from Phase 4's own committed (float32-precision) number by
   ~1e-7, immaterial to any reported metric digit.
2. **T1's real round trip is a stratified 100-row sample**, not the full
   val partition -- per-row encrypt/run/decrypt is meaningfully slower
   than the vectorized `fhe="disable"`/`"simulate"` paths (Sec.11);
   `docs/plan.md`'s exit criterion only requires "at least one" round
   trip, which this satisfies with margin (100, not 1).
3. **This session added T0/T2/T3 as additional acceptance gates**,
   beyond `docs/plan.md`'s literal "validate against plaintext output" --
   justified by wanting each potential error source (transfer,
   quantization, circuit/simulation, execution) to be independently
   attributable, per this project's general correctness-layering
   discipline (`CLAUDE.md` §7/§11).
4. **T3 failed and Phase 5 is not being silently marked complete despite
   `docs/plan.md`'s literal exit criterion (T1) being met** -- an explicit
   deviation from "phase passes when its exit criterion is met", per this
   session's stronger instruction to treat T0-T3 as hard gates, all
   required, none silently loosened.
5. **A single controlled `n_bits=16` follow-up experiment was run**
   (Sec.9.1/9.2), beyond the plan's original `n_bits=8`-only scope, to test
   whether raising bit-width alone resolves T3. It did not; `n_bits` is
   restored to 8 (Sec.9.4) and no further bit-width increase is planned
   inside Phase 5.
6. **A config bug (`configs/phase5/lr_poc.yaml` missing its `seed` key)
   caused the `n_bits=16` run's T1 step to crash** before writing anything
   to disk (Sec.9.3). Fixed, and T1 was independently re-verified at
   `n_bits=8` (Sec.9.4) without recomputing T2/T3, reproducing the
   originally-recorded T1 result exactly.

## 11. What Phase 5 deliberately did not do

- No XGBoost or MLP FHE compilation (Phase 6/9).
- No `top_50`/`top_100` tiers, and no `n_bits` other than 8 (Phase 8's
  benchmark grid).
- No repeated-trial benchmarking -- `compile_seconds` (129.7s at the
  original `n_bits=8` run, 97.0s at `n_bits=16`, 105.5s at the `n_bits=8`
  T1 re-verification) and the per-row round-trip timings (`n_bits=8`
  original: `keygen_seconds` 0.0022s, `mean_row_seconds` 0.0093s;
  `n_bits=8` re-verification: `keygen_seconds` 0.0001s, `mean_row_seconds`
  0.0114s -- `n_bits=16` never reached T1, Sec.9.3) are single
  observational measurements, not a benchmark (Phase 7 owns ≥5-trial
  measurement).
- No client/server implementation, no `FHEModelDev`/`Client`/`Server`
  usage, no network/serialization overhead measurement (Phase 10).
- No use of the test partition anywhere (Sec.4), including during the
  `n_bits=16` follow-up and the T1 re-verification (Sec.9).
- No modification of any Phase 1-4 committed artifact -- verified via
  `git status`/`git diff` showing zero changes to those paths, both at
  the original PoC and after the Sec.9 follow-up.
- No clipping, winsorizing, retraining, or further `n_bits` change in
  response to either T3 failure (`n_bits=8` or `n_bits=16`) -- both
  reported as findings (Sec.8/Sec.9), not worked around.
- No further `n_bits` increase beyond 16 attempted -- confirmed
  uncompilable for this circuit shape (`NoParametersFound` at 18+,
  Sec.9.1), so there is nothing higher left to try inside Phase 5's
  linear-model scope.
- No T2/T3 recomputation during the Sec.9.4 config-bug fix -- only T1's
  own sample-level steps were rerun; the originally-recorded `n_bits=8`
  T0/T2/T3 report was left untouched.
