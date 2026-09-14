# Phase 6 — FHE Fraud Inference (XGBoost · `top_20` / `top_50` / `top_100` · `n_bits=8`)

Implements `docs/plan.md` Phase 6: extend the working FHE path from Logistic
Regression (Phase 5) to XGBoost, across all three committed Phase 4 feature
tiers, at one initial bit-width. Run as:

```bash
# Windows .venv (plaintext side) -- exports all three tiers
python -m src.fhe.export_xgboost --config configs/phase6/xgb_poc.yaml

# WSL2 FHE venv (Concrete-ML side) -- from the project root, one tier per process
~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.xgb_poc --config configs/phase6/xgb_poc.yaml --tier top_20
~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.xgb_poc --config configs/phase6/xgb_poc.yaml --tier top_50
~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.xgb_poc --config configs/phase6/xgb_poc.yaml --tier top_100
~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.xgb_poc --config configs/phase6/xgb_poc.yaml --summary
```

**Status: CLOSED, exit criterion met (`docs/plan.md`), same shape as Phase
5.** `docs/plan.md`'s Phase 6 exit criterion -- "all feature tiers compile
and validate correctly... at the initial bit-width" -- uses "validate" as
`architecture.md` §5 / `prd.md` FR7 / `instructions.md` define it: circuit
output vs. plaintext (quantized) output, not quantized-vs-float accuracy.
By that definition the criterion **is met**: for every one of the three
tiers, T0 (cross-version transfer), T2 (simulated integer outputs), and T1
(**real** encrypt → run → decrypt) all pass exactly. This is the literal
proof that the FHE path works: a real ciphertext round trip through a
compiled XGBoost circuit, decrypted, matches the circuit's own simulation and
its clear-quantized computation bit-for-bit.

T3 (quantized-vs-float accuracy) -- this project's own additional accuracy
gate, not part of the roadmap's "validate" definition above -- fails for
every tier at `n_bits=8`, with decision agreement between 0.649 and 0.894
(need ≥0.99) and PR-AUC dropping by 0.19-0.29 (need ≤0.01) -- the same
quantization-accuracy problem Phase 5 found for Logistic Regression, now
independently reproduced for a structurally different model type. Per this
session's explicit instruction, no remediation was attempted (no bit-width
sweep, no retrained model, no loosened T3 tolerance): the failure is
measured and reported, not worked around, and is forwarded as a documented
research finding to Phase 8's systematic bit-width sweep -- the roadmap's
designated venue for the feature-count × bit-width × model-type
investigation (`docs/plan.md` Phase 8), not Phase 6's.

## 1. Objective and scope

Compile the committed Phase 4 XGBoost model for each of `top_20`, `top_50`,
`top_100` with Concrete-ML at `n_bits=8`, and run real encrypted inference on
each: keygen → encrypt → run on ciphertext → decrypt. Validate with the same
layered gates Phase 5 used (T0/T1/T2/T3), using the Phase 4 tier model
exactly, the Phase 2 temporal split and leakage boundaries, train-only
calibration, and validation-only (never test-partition) evaluation. No
benchmark harness (Phase 7), no bit-width sweep or full research grid (Phase
8) -- both explicitly out of scope here.

## 2. Environment split and the handoff -- simpler than Phase 5's

Concrete-ML only installs under WSL2/Linux (`docs/environment.md`), same
constraint as Phase 5. Unlike Logistic Regression, **XGBoost's native JSON
booster format is already cross-version-portable**: no numpy
reimplementation of `predict_proba` is needed, because the WSL side's older
`xgboost==1.6.2` can load and predict from a booster fit by the Windows
`.venv`'s `xgboost==3.4.1` directly. `src/fhe/export_xgboost.py` (Windows)
therefore only has to:

1. Reload the committed Phase 4 tier XGBoost model and verify it against its
   own committed `metrics.json` (config-hash match, reload-no-refit val
   PR-AUC within `1e-6` -- the same "reload the saved artifact" pattern
   `src/features/pipeline.py` and `src/fhe/export.py` already established).
2. Slice the booster to exactly the trees `predict_proba` uses (Sec. 6.1) and
   write it as native XGBoost JSON.
3. Reuse `src/fhe/handoff.py::save_handoff`/`sha256_file` **unchanged** --
   already model-agnostic -- to write the train/val arrays + a manifest.

Every handoff is integrity-checked exactly like Phase 5's: the npz's SHA-256
is verified against the manifest before it is read, the exported booster's
SHA-256 is recorded and re-verified on the WSL side, and the booster is
asserted to reproduce the Windows reference's predictions bit-for-bit before
it is trusted.

## 3. Why these three tiers, and why `n_bits=8`

`docs/plan.md`'s Phase 6 scope is explicit: all three committed Phase 4
tiers, one initial bit-width. Neither `docs/plan.md` nor `docs/prd.md` names
a specific number for "initial bit-width" for XGBoost; `n_bits=8` is kept for
consistency with Phase 5's own initial value, stated here as an
interpretation, not a hidden default.

## 4. End-to-end flow (per tier)

1. Load + verify the Windows-side handoff (npz SHA-256, exported booster
   SHA-256, tree count).
2. **T0**: WSL xgboost (`1.6.2`) predicting the exported booster vs. the
   Windows xgboost (`3.4.1`) reference, on the full val partition.
3. Build + compile the Concrete-ML `XGBClassifier` on train-only calibration
   rows (Sec. 6.3).
4. **T3**: `fhe="disable"` (clear-quantized) vs. the float reference, on the
   full val partition.
5. **T2**: official `Circuit.simulate` vs. `disable`, compared as exact
   integer circuit outputs, on a seeded 5000-row val sample (Sec. 9.2).
6. **T1**: real `keygen → encrypt → run → decrypt` on a seeded, class-
   stratified 2-row val sample (1 fraud, 1 legitimate; Sec. 9.3), compared as
   exact integer outputs against that circuit's own simulation.
7. Write `results/phase6_fhe_xgboost/{tier}/{correctness_report,circuit_stats,metrics,provenance}.json`
   and `execute_sample.csv`. Every measurement is written to disk **before**
   any gate failure is raised, and one tier failing does not stop the others
   from running (`src/fhe/xgb_poc.py::main` collects failures and continues).

The test partition is never received by the WSL side: the handoff npz only
holds train/val arrays.

## 5. Results for every acceptance gate

All three tiers, real handoff, real compile, real execution -- no toy data.

| Tier | Features | Trees (inference / stored) | T0 | T1 (real) | T2 (5000 rows) | T3 |
|---|---|---|---|---|---|---|
| `top_20` | 20 | 358 / 378 | PASS | PASS | PASS | **FAIL** |
| `top_50` | 50 | 219 / 239 | PASS | PASS | PASS | **FAIL** |
| `top_100` | 100 | 271 / 291 | PASS | PASS | PASS | **FAIL** |

**T0** (cross-version transfer, tolerance `1e-7` -- Sec. 6.4): max|diff| is
exactly `5.960464e-08` for every tier (45-50 of 105,088 val rows affected).

**T1** (real execution, `n=2`, stratified 1 fraud / 1 legitimate): decrypted
integer circuit outputs match that row's own simulation **exactly**, for
every row, every tier -- `n_decision_flips=0`, `n_integer_outputs_mismatched=0`.
Also cross-checked against the clear-quantized (`disable`) computation for
the same rows: exact match there too
(`decrypted_vs_disable_exact_integer_match=True`).

**T2** (simulated, `n=5000` seeded val rows): exact integer match against
`disable` for every row, every tier -- zero mismatches, zero decision flips.

**T3** (quantized `disable` vs. float reference, full val, `n=105,088`):

| Tier | Float PR-AUC | Quantized PR-AUC | PR-AUC drop (max 0.01) | Decision agreement (min 0.99) |
|---|---|---|---|---|
| `top_20` | 0.52267 | 0.33093 | 0.19175 | 0.64901 |
| `top_50` | 0.54422 | 0.30636 | 0.23786 | 0.67531 |
| `top_100` | 0.56780 | 0.28243 | 0.28537 | 0.89371 |

All three fail both the decision-agreement and PR-AUC-drop bars by a wide
margin -- not a borderline result.

### Compile and real-execution observations (not benchmarks -- Phase 7 owns repeated-trial measurement)

| Tier | PBS count | Compile time | Real `run` (1 row, mean) | Simulate (1 row, mean of 5000) | Peak RSS |
|---|---|---|---|---|---|
| `top_20` | 248,452 | 60.5s | 1892.8s (31.5 min) | 0.294s | 2981 MB |
| `top_50` | 151,986 | 51.7s | 1238.4s (20.6 min) | 0.224s | 2558 MB |
| `top_100` | 188,074 | 90.5s | 1525.2s (25.4 min) | 0.259s | 3019 MB |

All three circuits: `p_error = global_p_error = 8.6926e-13`, max integer
bit-width `10`, bootstrap keys ≈802 MB, keyswitch keys ≈111 MB, secret key
59,136 bytes. Encrypt (~0.01-0.02s) and decrypt (~0.15-0.26s) are negligible
next to `run`; keygen is ~2.1-2.2s for every tier. Real `run` time scales
almost linearly with PBS count (≈0.0076-0.0081s per PBS across all three
tiers) -- consistent, not noisy, and a useful planning number for Phase 7.

## 6. Real behaviors verified empirically

### 6.1 Concrete-ML converts every tree stored in a booster, ignoring `best_iteration`

Every committed Phase 4 tier model is early-stopped, so its booster stores
`best_iteration + 1 + early_stopping_rounds` (20) trees, but `predict_proba`
only uses the first `best_iteration + 1` at inference. Concrete-ML's
`XGBClassifier.from_sklearn_model` compiles **every tree the booster has
stored**, with no `best_iteration` awareness -- confirmed by direct test
(`tests/test_fhe_xgb_poc.py::test_concrete_compiles_every_stored_tree_ignoring_best_iteration`):
compiling the full booster produces an inference-time output with one entry
per *stored* tree, not per *used* tree. Exporting the full booster unsliced
would silently compile a **different model** than the reference. The fix
(`src/fhe/export_xgboost.py::inference_booster`) slices to
`booster[0:best_iteration+1]` before export, and `export_tier` asserts the
sliced booster reproduces the reference's predictions bit-for-bit and that
its tree count equals Phase 4's committed `selected_n_estimators`, so this
can never regress silently.

### 6.2 `xgboost==1.6.2`'s `load_model` doesn't set `n_classes_`

Loading a native-JSON booster into a bare `xgb.XGBClassifier()` under the
WSL venv's `xgboost==1.6.2` leaves `n_classes_` unset, and Concrete-ML's
Hummingbird-based conversion (via `from_sklearn_model`) raises
`AttributeError: 'XGBClassifier' object has no attribute 'n_classes_'`.
`src/fhe/compile/tree.py::load_inference_classifier` sets
`clf.n_classes_ = 2` explicitly after load (this project's binary
classification setting, verified against the loaded booster's own
`binary:logistic` objective first).

### 6.3 Calibration memory scales with tree count x row count, not just row count

Concrete-ML's `from_sklearn_model` computes each tree's output bit-width from
a `(trees x calibration_rows x nodes)` matmul over the *whole* calibration
set in one call (`_compute_lsb_to_remove_for_trees`) -- unlike Logistic
Regression, whose per-feature linear quantizer made "calibrate on the whole
train partition" (Phase 5's convention) cheap. Measured directly on the real
`top_20` booster (358 trees -- the most of the three tiers, so the worst
case) in the WSL venv (3.8 GiB RAM, no swap headroom to spare):

| Calibration rows | Wall time | Peak RSS |
|---|---|---|
| 2,000 | 8.9s | 1.87 GB |
| 3,000 | 22.5s | 2.41 GB |
| 5,000 | 119.8s | 3.33 GB (87% of the ceiling) |

The jump from 3,000 to 5,000 rows is not linear (2.5x the rows, ~5.3x the
time) -- consistent with swap/memory pressure starting to dominate near the
ceiling, not a safe operating point. `calibration.rows: 3000` with
`include_extremes: true` (each feature's train min/max row added on top) was
chosen for a comfortable safety margin while keeping every per-feature
quantizer's range anchored to the true train-set extremes regardless of the
random subsample.

### 6.4 T0's cross-version difference is a float32 ULP floor, not a bug

Measured on all three real tiers before any tolerance change: `max|diff|` between
the WSL `xgboost==1.6.2` prediction and the Windows `xgboost==3.4.1` reference
is **exactly** `5.960464477539063e-08` (`== 2**-24`, the IEEE-754 float32
machine epsilon) for every tier, affecting only 45-50 of 105,088 val rows
(≈0.05%); the 99th and 99.9th percentile of the diff distribution are both
exactly `0.0` -- over 99.9% of rows are bit-identical across the two xgboost
builds. This is qualitatively different from Phase 5's T0, which compared a
numpy-reimplemented, deterministic `predict_proba` against the reference (no
cross-binary variability possible); Phase 6's T0 compares two independently
compiled xgboost releases' own C++ prediction routines evaluating the
identical booster JSON. A single-ULP float32 divergence on a handful of rows
between library versions is expected floating-point behavior, not a modeling
error. `t0_max_abs_diff` was changed from Phase 5's copied `1e-9` to `1e-7` --
comfortably above the observed floor, still far tighter than any
decision-relevant probability difference.

### 6.5 `disable` vs `simulate`/real execution: exact at the integer level, for every tier

Phase 5's approved plan for Phase 6 anticipated needing a probabilistic
(`~99%`) decision-level agreement bar for T1/T2, reasoning from XGBoost's
nonzero `p_error` (unlike LR's zero-PBS circuit) that some simulated/executed
rows might diverge from the clear-quantized computation. Measured directly:
**every** T2 row (5000/tier) and **every** T1 row (2 real rows/tier, all 3
tiers) matched `disable`'s integer output exactly -- zero mismatches. At the
circuit's own `p_error = 8.69e-13` and ~150k-250k PBS calls per row, the
expected probability of *any* divergent PBS in a single row is on the order
of `1e-7`, and across the ~15,000+ total simulated/executed rows measured
here the expected count of any divergence is well under 1 -- consistent with
observing zero. `integer_output_check` (`src/fhe/validate/correctness.py`)
still reports an `agreement_rate`-shaped result generically, but no tier
needed anything looser than exact match; the anticipated need for a
tolerance did not materialize at this scale.

## 7. What T3's failure means, and the options

Exactly Phase 5's finding, now independently reproduced for a structurally
different model: 8-bit quantization is not fine enough for either model type
on this dataset/feature-tier combination to preserve fraud-decision quality.
For XGBoost specifically, decision agreement improves somewhat as more
features are added (`top_20`: 0.649 -> `top_100`: 0.894), but even
`top_100`'s PR-AUC drop (0.285) is nearly 30x the 0.01 tolerance. Per this
session's explicit instruction, none of the following were attempted here:
raising `n_bits` (that bit-width sweep is Phase 8's scope), retraining or
altering the model, loosening the T3 tolerance, or reducing tree depth/count/
feature count to force a pass. The quantization-accuracy limitation is
recorded as a research finding, exactly as Phase 5's was.

## 8. Deviations from `docs/plan.md` and from Phase 5's own conventions

All measured, none guessed, all in `configs/phase6/xgb_poc.yaml` with the
full reasoning inline:

1. **Calibration: 3000 rows + `include_extremes`, not "all"** (Sec. 6.3) --
   Phase 5's "calibrate on the whole train partition" is infeasible for
   trees given WSL's 3.8 GiB ceiling.
2. **T0 tolerance: `1e-7`, not Phase 5's `1e-9`** (Sec. 6.4) -- comparing two
   real cross-version xgboost binaries surfaces a float32 ULP floor that a
   numpy-reimplemented reference (Phase 5's LR case) never could.
3. **T2: seeded 5000-row sample, not "all" 105,088 val rows** -- measured
   simulate cost (~0.22-0.29s/row, no batching) makes full-val T2 a
   multi-hour run per tier; Phase 5's LR circuit had zero PBS, so "all" was
   cheap there. 5000 rows is large enough to have a real chance of catching a
   divergence (Sec. 6.5) while completing in ~20-25 minutes/tier.
4. **T1 `execute_sample`: 2 rows (1 fraud, 1 legitimate), not Phase 5's
   `n=100`** -- a single real row's `run` step measured at 20.6-31.5 minutes
   (Sec. 5), driven by real PBS cost that LR's zero-PBS circuit never
   incurred; `n=100` would be 30-100+ hours per tier. `explicit_round_trip`
   gained per-row disk checkpointing (mirroring T2's existing chunk
   checkpoints) so an interrupted sample resumes at the next row rather than
   restarting -- this interactive environment's WSL2 VM / session lifecycle
   interrupted three consecutive unattended `run` attempts on `top_20` before
   the config was scoped down, so resumability is a real, not theoretical,
   need.
5. **The anticipated T1/T2 tolerance (Sec. 6.5)** never had to be used: every
   row observed matched exactly.

## 9. What Phase 6 deliberately did not do

- No benchmark harness, no repeated trials, no latency/memory statistics
  beyond single-run observations explicitly labeled as such -- Phase 7's
  scope.
- No bit-width sweep, no full feature-tier x bit-width x model grid --
  Phase 8's scope.
- No attempt to fix, work around, or loosen tolerances past T3's failure.
- No change to any Phase 1-5 committed artifact, model, or config.
