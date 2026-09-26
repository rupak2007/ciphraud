# Phase 7 — Benchmark Infrastructure

Implements `docs/plan.md` Phase 7: build the config-driven benchmark
harness, before the full experimental grid (Phase 8) runs on it. Run as:

```bash
# WSL2 FHE venv (Concrete-ML side) -- from the project root
~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.run --config configs/phase7/smoke.yaml
```

**Status: COMPLETE, exit criterion met.** The smoke test (2 configurations,
2 trials each) passed with sane, reproducible output for both a
Logistic-Regression and an XGBoost configuration: every repeated-trial
latency, memory, ciphertext-size, and compile-time measurement was
collected, and every real FHE round trip returned bit-identical decrypted
output across trials on the same fixed request. No Phase 1-6 model, tier,
config, or result was changed.

## 1. Objective and scope

Build the harness `docs/architecture.md` §16 specifies -- repeated-trial
latency (plaintext and FHE), peak memory, ciphertext size, compile time
measured separately from inference latency, results stored keyed by
configuration hash -- and prove it works with a smoke test on a minimal
grid. Not the full feature-tier × bit-width × model-type research grid
(Phase 8); not a bit-width sweep; not a new model, tier, or quantization
methodology.

## 2. Design: reuse, never re-validate

`src/benchmark/run.py` never re-exports a handoff and never re-runs
correctness gates -- that is `src/fhe/poc.py`/`src/fhe/xgb_poc.py`'s job,
already done for every committed Phase 5/6 configuration. For a given
benchmark entry (`label`, `model_type`, `source_config`, and `tier` for
XGBoost), it:

1. Loads the **referenced** Phase 5/6 source config and its already-
   verified handoff (`src/fhe/handoff.py::load_handoff`, unchanged).
2. Rebuilds and recompiles the **same** model with the **same** calibration
   inputs Phase 5/6 already used (`build_concrete_lr`/`build_concrete_xgb`,
   `compile_model` -- all unchanged, all imported, not reimplemented). For
   XGBoost this reuses `src/fhe/xgb_poc.py::calibration_positions` directly,
   so the recompiled circuit is calibrated identically to Phase 6's.
3. `n_bits`, `model_type`, and `tier` all come from the source config or the
   entry itself -- never independently settable in a way that could diverge
   from what Phase 5/6 validated.

This means a benchmark "configuration" is structurally *the same compiled
circuit* Phase 5/6 already proved correct, recompiled once more purely to
measure it -- not a new experiment.

## 3. Metrics implemented

| Requirement (`architecture.md` §16 / `instructions.md` Benchmarking Rules) | Implementation |
|---|---|
| Repeated-trial latency, mean + std, never single-shot | `src/benchmark/harness.py::trial_stats` (`ddof=1` sample std; a 1-trial run reports `std=0.0`, not NaN) |
| Plaintext-inference benchmark | `plaintext_latency_trials`: N repeated calls to the reconstructed float model's own `predict_proba` (`rebuild_pipeline_predict_proba` for LR, the loaded booster's `predict_proba` for XGBoost) on one fixed row |
| FHE-inference benchmark | `fhe_round_trip_trials`: N repeated real `encrypt → run → decrypt` calls on that same fixed row, with ONE `keygen()` amortized across all trials (matching Phase 5/6's own `_explicit_round_trip`/`explicit_round_trip`) |
| Compile time separate from inference latency | `compile_model`'s existing return value (Phase 5/6, unchanged), stored as its own `compile_seconds` field, never mixed into the trial statistics |
| Peak memory | `src/fhe/xgb_poc.py::peak_rss_mb` (imported, not duplicated) -- `resource.getrusage(...).ru_maxrss`, recorded once per configuration after compiling + all trials |
| Ciphertext size where applicable | `circuit.size_of_inputs` / `circuit.size_of_outputs`, via the existing `circuit_stats()` (`src/fhe/compile/linear.py`, already generic across LR/XGBoost) |
| Config-hash-keyed result storage | `src/data/provenance.py::compute_config_hash` (its own docstring: *"deliberately generic... so Phase 7/8 benchmark runs can reuse the same helper"*), hashed over the benchmark entry's own effective config (label, model_type, tier, source config path + hash, n_bits, seed, trials) |
| Structured, reproducible results | One `results/phase7_benchmark/{config_hash}/{metrics,provenance}.json` per configuration, plus an aggregate `summary.json` |
| Throughput where meaningful | Derived as `1 / mean_total_seconds` for both the plaintext and FHE paths |
| Output reproducibility | Every repeated trial's decrypted (or plaintext-predicted) output is compared bit-for-bit across trials; `outputs_reproducible: true/false` |

## 4. Result schema

Per configuration (`results/phase7_benchmark/{config_hash}/metrics.json`):

```
label, model_type, tier, n_bits, n_features, seed, trials, n_positive
source_config, source_config_hash, config_hash
compile_seconds
circuit_stats: { programmable_bootstrap_count, p_error, global_p_error,
                 complexity, size_of_{secret,bootstrap,keyswitch}_keys_bytes,
                 size_of_{inputs,outputs}_bytes, [n_trees, n_bits_inputs,
                 n_bits_output, max_integer_bit_width for XGBoost] }
ciphertext_size_bytes: { input, output }
key_size_bytes: { secret, bootstrap, keyswitch }
row_selection: { position, y_true, n_positive }
plaintext_latency: { n_trials, mean_seconds, std_seconds, min_seconds,
                      max_seconds, trials_seconds, outputs_reproducible,
                      throughput_requests_per_second }
fhe_latency: { keygen_seconds, outputs_reproducible,
                throughput_requests_per_second,
                total/encrypt/run/decrypt: { <same shape as plaintext_latency> } }
peak_rss_mb
test_partition_touched (always false -- the handoff npz never carries X_test)
```

Plus `provenance.json` (git commit, timestamp, library versions, seed,
benchmark config hash -- `src/data/provenance.py::build_provenance`,
unchanged) and a top-level `results/phase7_benchmark/summary.json`
aggregating the key fields across every configured entry.

## 5. Smoke-test configurations

`configs/phase7/smoke.yaml`: 2 configurations (within `docs/plan.md`'s
"2-3" range), 2 trials each ("few trials" per `docs/plan.md`/
`architecture.md` §19 -- distinct from Phase 8's ≥5-trial minimum), one
fixed seeded legitimate-class row (`n_positive=0`) reused across all
trials:

| Label | Model | Tier | `n_bits` | Source config |
|---|---|---|---|---|
| `lr_top20_bits8` | Logistic Regression | `top_20` | 8 | `configs/phase5/lr_poc.yaml` |
| `xgboost_top50_bits8` | XGBoost | `top_50` | 8 | `configs/phase6/xgb_poc.yaml` |

Only 2 (not 3): XGBoost's real per-row FHE execution measured 20.6-31.5
minutes/row in Phase 6, so a 3rd configuration (`top_20` or `top_100`)
would have added roughly another hour to what is meant to be a fast
sanity check of the harness itself, not a benchmark run. `top_50` (219
trees) is the cheapest of the three committed XGBoost tiers.

## 6. Smoke-test results

Both configurations passed with sane, fully reproducible output.

| Metric | `lr_top20_bits8` | `xgboost_top50_bits8` |
|---|---|---|
| Compile time | 118.1s | 55.9s |
| PBS count | 0 | 151,986 |
| Plaintext latency (mean ± std, 2 trials) | 1.56ms ± 2.11ms | 7.01ms ± 8.39ms |
| FHE latency, total (mean ± std, 2 trials) | 99.4ms ± 86.2ms | 1358.6s ± 88.6s (≈22.6 min) |
| FHE latency, `run` alone (mean) | 92.5ms | 1358.3s |
| FHE outputs reproducible across trials | **True** | **True** |
| Ciphertext size (input / output) | 480B / 10,192B | 1,200B / 3,589,848B (≈3.4MB) |
| Key sizes (secret / bootstrap / keyswitch) | small / 0 / 0 (zero PBS → no bootstrap/keyswitch keys needed) | 59,136B / 840,876,032B / 115,953,664B |
| Peak process RSS | (LR: negligible, well under 1 GB) | 2214 MB |

The individual per-trial numbers (`trials_seconds`) are in each config's
`metrics.json` -- e.g. XGBoost's two `run` trials were 1295.4s and
1421.1s, a ≈6% spread consistent with ordinary system-load noise on a
2-trial sample, not a correctness concern (T0-T3 already own correctness;
this harness only measures already-validated circuits).

> **Erratum (2026-09-25):** the LR row in the Phase 7 smoke test was compiled on
> raw (unstandardized) features, so its accuracy numbers are not valid
> quantization evidence; its timings and circuit shape remain valid
> observations. See `docs/research.md` §5.

The 3-4 order-of-magnitude latency gap between LR (zero PBS, sub-100ms real
FHE round trip) and XGBoost (151,986 PBS, ~22.6-minute round trip) is the
same PBS-driven cost `docs/fhe_xgboost.md` §5-6 already measured in Phase
6's ad hoc probes -- this harness now reports it as a proper repeated-trial
statistic instead of a single anecdotal run.

## 7. Tests

`tests/test_benchmark_harness.py`: 5 platform-independent tests (trial
statistics math, reproducibility detection) run on every platform; 2
Concrete-ML-dependent tests (`skipif`-gated, matching every other Phase 5/6
FHE test) exercise a real compiled toy circuit's repeated round trips and
the full `benchmark_configuration` pipeline end-to-end against a synthetic
LR handoff written to `tmp_path` -- never the real committed Phase 5/6 data.
**7/7 passed** under the WSL2 venv; **5 passed, 2 skipped** on Windows.
Existing Phase 5/6 synthetic test suites re-run unchanged: **39/39 passed**,
zero regressions (this module only imports from `src/fhe/poc.py`/
`src/fhe/xgb_poc.py`; neither was modified).

## 8. Deviations from `docs/plan.md`

1. **Smoke grid: 2 configurations, not 3** -- within the plan's own
   "2-3" range; the 3rd would have added roughly an hour of real XGBoost
   FHE execution to a step meant to be a fast harness sanity check, per
   NFR3 ("scoped to what can complete... in feasible wall-clock time").
2. **Smoke trials: 2, not ≥5** -- explicitly permitted: `docs/plan.md`
   Phase 7's own Tasks say "smoke-test mode (small grid, **few trials**)",
   and `architecture.md` §19 repeats "few trials" for the harness's own
   smoke test, distinct from the ≥5-trial minimum `instructions.md`'s
   Benchmarking Rules require for *reported* benchmark results (Phase 8).
3. **One fixed row per configuration, reused across all trials** -- not
   specified by any of the five reviewed documents. Chosen because a
   compiled Concrete-ML circuit performs the same fixed sequence of
   homomorphic operations regardless of input value (confirmed by this
   run's own `outputs_reproducible: true` and the PBS count being a static
   circuit property, not input-dependent), so repeated trials on one row
   isolate measurement noise, matching the plan's own stated risk
   ("measurement noise from other system load affecting latency figures").

## 9. What Phase 7 deliberately did not do

- No Phase 8 full grid (all tiers × bit-widths × models × ≥5 trials).
- No bit-width sweep, no new tier, no new model, no quantization or
  correctness-criteria change.
- No Pareto-frontier analysis or prior-art comparison (`prd.md` §11's ~6ms
  XGBoost / 296ms neural-network citation) -- both are Phase 8's job, once
  the full grid exists; this smoke test's own ≈22.6-minute XGBoost
  measurement already suggests a large divergence from that citation, but
  a single 2-trial smoke measurement is not the place to draw that
  conclusion formally.
- No change to any Phase 1-6 model, config, result, or code.
