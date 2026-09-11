# Phase 4 — Feature Importance & Feature-Count Tiers

Implements `docs/plan.md` Phase 4: a real, ranked feature-importance list and the
feature-count tiers every later FHE phase (5+) will reference by config
(`docs/architecture.md` Sec.4). Run as:

```bash
python -m src.features.pipeline --config configs/phase4/features.yaml
```

Writes ranking/curve/stability tables and the versioned `tiers.json` artifact to
`results/phase4_features/`, and per-tier plaintext reference models to
`models/phase4_features/<tier>/` (gitignored via the `models/` rule).

## 1. Data and model provenance

Phase 4 sources model-ready data from Phase 2 exactly like Phase 3
(`src/train/data.py::load_phase2_features` — the sole entry point, never re-derived).
It additionally depends on Phase 3's **already-trained, already-committed** XGBoost
model as the ranking source, gated by
`src/features/pipeline.py::_verify_phase3_integrity`, which:

1. Confirms `models/phase3_baselines/{logistic_regression.joblib,xgboost.json}` and
   `results/phase3_baselines/{metrics.json,provenance.json}` all exist.
2. Confirms the committed `provenance.json`'s `config_hash` matches the *current*
   `configs/phase3/baselines.yaml` — if Phase 3's config changed since its committed
   run, this fails loudly ("re-run Phase 3 first") rather than ranking against a stale
   or hand-edited model.
3. **Reloads the actual saved model artifacts** (no re-fit) and recomputes their val
   PR-AUC, asserting it matches the committed `metrics.json` value to within
   `1e-6` (`_INTEGRITY_RELOAD_TOLERANCE` — a correctness invariant, not a config value,
   since a reload-and-predict involves no re-fitting and should be near-exact).

This ties model file ↔ committed metrics ↔ current Phase 2 feature order together
before a single importance number is computed.

## 2. Importance method: XGBoost `total_gain`, not permutation importance

Per this session's explicit decision: ranking uses the Phase 3 XGBoost model's own
`total_gain` (`src/features/importance.py::xgboost_total_gain_ranking`) — `docs/plan.md`
Phase 4's first-listed option — rather than permutation importance. Permutation
importance would be slow on 832 features × 105K val rows, would dilute the true signal
across the many highly-correlated `V` columns (each permutation only measures marginal
loss with all correlated siblings intact, understating a whole block's joint
importance), and it uses `val` for both ranking and evaluating the tiers — total_gain
avoids all three.

`total_gain` (sum of gain across every split using a feature), not average `gain`, is
the ranking criterion — average gain overweights a feature used in one lucky
high-gain split. `booster.get_score()` omits any feature never split on; these are
explicitly zero-filled (459/832 features had nonzero gain in the real run — 373 were
never split on at all) rather than silently dropped, so every one of Phase 2's 832
columns gets a rank. Ties (including the large zero-gain block) break by feature name.

## 3. Real-dataset results

`results/phase4_features/metrics.json`. Split sizes: train 380,815 / val 105,088 /
test 104,637 (test untouched, Sec.7). The reproduction check (Sec.6.1 — a full-feature
refit in Phase 2's original column order, tolerance `1e-6`) passed with an **exact
`0.0` diff for both models**; see Sec.6.1 for the column-order finding behind this
check's design and why an earlier version of it used a much looser tolerance for the
wrong reason.

### Top 20 features by `total_gain`

| Rank | Feature | Total gain |
|---|---|---|
| 1 | `V258` | 613,712 |
| 2 | `C14` | 304,116 |
| 3 | `V294` | 261,835 |
| 4 | `C13` | 206,911 |
| 5 | `C1` | 204,331 |
| 6 | `TransactionAmt` | 191,108 |
| 7 | `V70` | 179,775 |
| 8 | `card1` | 172,713 |
| 9 | `C8` | 166,683 |
| 10 | `D2` | 142,305 |
| 11 | `card2` | 137,457 |
| 12 | `card6_freq` | 110,425 |
| 13 | `addr1` | 109,291 |
| 14 | `V308` | 89,651 |
| 15 | `C11` | 87,056 |
| 16 | `D15` | 80,269 |
| 17 | `C5` | 77,451 |
| 18 | `V187` | 76,937 |
| 19 | `P_emaildomain_freq` | 74,383 |
| 20 | `M4_freq` | 73,979 |

Full 832-row ranking: `results/phase4_features/feature_ranking.csv`.

### Ranking stability (top-k Jaccard vs. the Phase 3 seed)

`results/phase4_features/ranking_stability.csv` — two additional full-feature XGBoost
refits (seeds 43, 44) compared against the Phase 3 seed-42 ranking:

| k | seed 43 | seed 44 |
|---|---|---|
| 5 | 0.667 | 0.667 |
| 10 | 0.818 | 0.818 |
| 20 | 0.739 | 0.905 |
| 50 | 0.923 | 0.961 |
| 100 | 0.869 | 0.852 |
| 500 | 0.961 | 0.953 |

**A real, honest limitation, not glossed over**: the ranking at very small `k` is
noticeably less stable across seeds than at larger `k` — at `k=20` (the smallest
official tier), only 74–90% of the top-20 set agrees across seeds; at `k=50`/`k=100` it
is consistently ≥85%. This means `top_20`'s *exact* membership should be read as "the 20
most importance-concentrated features, with some seed-to-seed churn at the margin," not
as a fixed ground truth — a caveat future phases compiling FHE circuits for `top_20`
should carry forward.

### Accuracy-vs-feature-count curve (fixed Phase 3 hyperparameters)

`results/phase4_features/accuracy_vs_feature_count.csv` /
`pr_auc_vs_feature_count.png` — both models refit at each `k`, holding Phase 3's
selected hyperparameters fixed (isolating the cost of feature count alone from any
re-tuning effect):

| `k` | LR PR-AUC | XGBoost PR-AUC |
|---|---|---|
| 5 | 0.324 | 0.418 |
| 10 | 0.313 | 0.464 |
| 20 | 0.350 | 0.523 |
| 50 | 0.388 | 0.547 |
| 100 | 0.421 | 0.554 |
| 150 | 0.443 | **0.572** |
| 300 | 0.459 | 0.565 |
| 832 (full) | 0.455 | 0.563 |

**A genuinely interesting, unanticipated finding**: at Phase 3's fixed hyperparameters,
XGBoost's PR-AUC **peaks around k=150 (0.572) and is slightly lower at the full
832-feature set (0.563)** — more features is not strictly better once hyperparameters
are held fixed rather than re-tuned per feature count. This is a real signal that
Phase 3's hyperparameters were selected for the full feature set and are not
automatically optimal at every feature count, and it is direct, measured evidence for
why Sec.4's **per-tier CV re-tuning** (not just this fixed-hyperparameter curve) is the
right basis for the official tiers below. LR's curve is noisier at very small `k`
(0.324 at k=5 dropping to 0.313 at k=10) for the same fixed-hyperparameter reason —
`C=10.0` (Phase 3's selection for the full feature set) is not necessarily well-suited
to a 5-10 feature model.

### Official tiers (top-20/50/100), per-tier CV re-tuned

Per this session's decision, tiers are **pure top-k** by rank — `docs/eda.md`'s
suggestion to sample across the 15 V-column null-mask blocks is honored only as the
diagnostic below, never as something that changes tier membership. Each tier is
re-tuned via Phase 2's expanding-window CV **exactly like Phase 3** (same grids, same
CV-pool-vs-train-only discipline, `src/features/evaluate.py::evaluate_tier_retuned`) —
not the fixed-hyperparameter shortcut the curve above uses.

| Tier | LR `C` | LR PR-AUC | LR retention | XGBoost (depth, lr, n_est) | XGBoost PR-AUC | XGBoost retention | XGBoost seed std |
|---|---|---|---|---|---|---|---|
| `top_20` | 10.0 | 0.3497 | 76.7% | (6, 0.05, 358) | 0.5227 | 91.5% | 0.0033 |
| `top_50` | 1.0 | 0.3884 | 85.2% | (6, 0.1, 219) | 0.5442 | 95.3% | 0.0037 |
| `top_100` | 10.0 | 0.4223 | 92.6% | (6, 0.1, 271) | 0.5678 | **99.4%** | 0.0061 |

Retention = tier PR-AUC / Phase 3's full-feature (832-column) PR-AUC (LR 0.4561,
XGBoost 0.5711). All 6 seed-stability std values (LR is exactly 0 — deterministic given
fixed data; XGBoost's are all < 0.02, matching Phase 3's documented band) are in
`results/phase4_features/metrics.json`'s per-tier `seed_stability`.

**The headline research finding**: XGBoost at 100 features (12% of the full 832) retains
99.4% of the full-feature model's PR-AUC. Even at 20 features (2.4% of the full set) it
retains 91.5%. This is the plaintext precursor to the eventual FHE Pareto frontier
(`docs/plan.md` Phase 8) — strong evidence that aggressive feature reduction, which
directly reduces FHE compilation/inference cost (`docs/architecture.md` Sec.5), costs
little plaintext accuracy for XGBoost specifically. LR's retention is meaningfully lower
at every tier size, a real, asymmetric cost of feature reduction between the two model
types worth carrying into Phase 8's cross-model comparison.

### V-column null-mask block coverage (diagnostic only — never changes tier membership)

`docs/eda.md` Sec.4 grouped the 339 anonymized `V` columns into 15 blocks by identical
null mask, flagging (Sec.4's "Implication for Phase 4") that a small tier risks dropping
an entire upstream signal source. Measured, not assumed:

| Tier | V-blocks with ≥1 covered column | V-blocks entirely absent |
|---|---|---|
| `top_20` | 4 / 15 | **11 / 15** |
| `top_50` | 8 / 15 | 7 / 15 |
| `top_100` | 11 / 15 | 4 / 15 |

**A real, honest limitation**: `top_20` entirely drops 11 of the 15 upstream V-column
signal sources — exactly the risk `docs/eda.md` anticipated. Per this session's
explicit decision, this is reported as a documented limitation, not corrected by
block-stratified sampling (`CLAUDE.md` Sec.6 requires tiers to come from the importance
ranking, not convenience-based block sampling) — Phase 5+ compiling an FHE circuit for
`top_20` should read its accuracy numbers with this caveat in mind. Full per-block
detail: `results/phase4_features/metrics.json`'s per-tier `v_block_coverage`.

## 4. Why per-tier CV re-tuning, not just the fixed-hyperparameter curve

`docs/plan.md` asks to "re-evaluate LR/XGBoost on each tier to quantify the plaintext
accuracy cost of feature reduction." Two distinct questions need two distinct answers
(`src/features/evaluate.py`):

- `evaluate_subset_fixed`: fast, fixed-hyperparameter refits driving the dense curve
  above — isolates "cost of feature count alone."
- `evaluate_tier_retuned`: the official tiers' actual numbers, from the same
  leakage-safe expanding-window CV selection and train-only final refit Phase 3 uses —
  answers "what's actually achievable at this feature count," which is what Phase 5+
  will actually compile and benchmark.

Sec.3's curve finding (XGBoost peaking at k=150 under FIXED hyperparameters, below its
own re-tuned `top_100` number) is direct evidence these are genuinely different
questions, not a redundant double-check.

## 5. The `tiers.json` versioned artifact

Per `docs/architecture.md` Sec.4 ("each tier is a first-class, versioned artifact...
since the FHE compilation and benchmarking pipelines depend on exact tier membership
being stable and reproducible"), `src/features/tiers.py::save_tiers`/`load_tiers` write
and re-verify a SHA-256 **membership hash** over the tiers' canonical JSON on every
read — a hand-edited or corrupted `tiers.json` fails loudly rather than silently feeding
a wrong feature set into Phase 5+'s FHE compilation. The real artifact also records:

- `source_model_sha256`: the exact `models/phase3_baselines/xgboost.json` bytes the
  ranking came from.
- `phase2_feature_list_sha256`: the exact Phase 2 `feature_columns` order ranked.
- `seed`, `importance_method`.

Tier-membership determinism (`docs/plan.md`'s required test) is covered at the unit
level on every test run
(`tests/test_features_tiers.py::test_build_tiers_is_deterministic_regardless_of_row_order`,
plus the tampered-hash test) rather than by re-running the full ~40-minute real-data
pipeline a second time in CI — that full-rerun check was performed once, manually, as
part of this phase's verification (Sec.8), not wired into the automated suite.

## 6. Two real behaviors verified empirically during this phase

### 6.1 The reproduction check's diff was column order, not thread nondeterminism — an initial misdiagnosis, corrected

`run()`'s reproduction check (Sec.3: refit the full 832-feature set at Phase 3's exact
hyperparameters/seed, compare to Phase 3's committed val PR-AUC) failed on the first
synthetic test run: LR's diff was exactly `0.0`, but XGBoost's was `~0.017` on the
synthetic fixture and `0.0085` on the real dataset. **The first explanation recorded
here was wrong.** It attributed the diff to XGBoost's `n_jobs=-1` threaded
histogram-building not being bit-identical across separate `.fit()` calls, amplified by
early stopping into a different `best_iteration` — a plausible-sounding mechanism that
was never actually tested against an alternative hypothesis, and the tolerance was
loosened to `0.03` to work around it.

That explanation did not survive a repeat run: a full real-data pipeline run was
re-executed as a determinism check on `tiers.json` (unrelated to this check), and it
reproduced the **exact same** `0.0084923...` XGBoost diff and `0.0010595...` LR diff,
bit-for-bit, as the first run. Thread-scheduling nondeterminism cannot reproduce
identical output twice — this was proof the diff was actually deterministic, and the
original diagnosis was retracted rather than kept.

**The real cause, found by a controlled experiment**: the reproduction check reused the
accuracy-vs-feature-count curve's full-feature-set row, and every curve point —
including that one — selects columns in **rank order** (`ranking.sort_values("rank")`),
not Phase 2's original column order. A dedicated script refit both models at all 832
features in (a) Phase 2's original order and (b) rank order, with Phase 3's exact
hyperparameters and seed:

| Column order | LR diff vs. Phase 3 | XGBoost diff vs. Phase 3 |
|---|---|---|
| Phase 2 order | **0.0** (exact) | **0.0** (exact) |
| Rank order | 0.0010595073489057527 | 0.008492353370847194 |

The rank-order row reproduces the *exact* diffs seen in both real pipeline runs — this is
the actual cause, not a hypothesis. Mechanism: XGBoost's `colsample_bytree=0.8`
(`src/train/xgboost_model.py`) samples a random subset of column **positions** at each
split; reordering columns changes which *named* features occupy the sampled positions
under the same seed, producing a different (but equally valid) tree ensemble. LR's much
smaller residual is consistent with floating-point summation-order sensitivity in
`lbfgs`'s gradient computation over a reordered matrix — plausible given `lbfgs`'s own
near-tolerance convergence behavior (Sec.7.3's finding in `docs/baselines.md`), though
this residual is small enough that it isn't separately re-verified beyond noting LR's
Phase-2-order diff was also exactly `0.0`.

**Fix**: `run()` no longer reuses the curve's full-feature-set row for this check. It
performs a dedicated refit using `feature_names` (Phase 2's actual column order)
directly, and `reproduction_check.pr_auc_tolerance` is restored to a tight `1e-6` — this
is now a load-bearing correctness check again, not a loosened workaround.
`_verify_phase3_integrity`'s reload-the-saved-artifact check (Sec.1, tolerance `1e-6`, no
re-fit involved) remains a separate, independent guarantee. See
`configs/phase4/features.yaml`'s `reproduction_check` comment for the full record.

### 6.2 A path-resolution quirk in `src/config.py::load_config`, surfaced by testing

`src/features/pipeline.py::run()` calls `load_config(config["phase3_config"])` directly
(unlike `load_phase2_features`, which the integration test monkeypatches away
entirely) — and that path is conventionally written `"configs/phase3/baselines.yaml"`
(matching e.g. `configs/phase3/baselines.yaml`'s own `phase2_config` field).
`load_config`'s primary resolution (`CONFIGS_DIR / config_path`) double-prefixes such a
path, since `CONFIGS_DIR` already ends in `configs`, so it silently falls through to a
CWD-relative fallback. In production this is harmless because scripts are run via
`python -m src.features.pipeline` from the project root (CWD == `PROJECT_ROOT`), but the
first version of `tests/test_features_pipeline_integration.py` — which patches
`PROJECT_ROOT`/`CONFIGS_DIR` to a `tmp_path` but does not change the test process's CWD —
fell through to the **real**, unrelated `configs/phase3/baselines.yaml` on disk instead
of the test's fake one. Fixed by adding `monkeypatch.chdir(tmp_path)` to the test
fixture, reproducing production's implicit CWD assumption rather than changing
`load_config` itself (out of this phase's scope, and every other phase's tests already
work around the same quirk the same way).

## 7. The test partition stays untouched

Identical discipline to `src/train/pipeline.py` (`docs/baselines.md` Sec.6):
`features.X_test`/`features.y_test` are loaded but never referenced in `run()` except
`len()` in `split_sizes`. Verified mechanically by
`tests/test_features_pipeline_integration.py::test_test_partition_never_touched` (a
sentinel object raising on any real access) and confirmed on the real dataset by
`tests/test_features_pipeline_real_data.py::test_pipeline_test_partition_not_in_results`.
`metrics.json` carries `"test_partition_touched": false`.

## 8. Test suite

- Synthetic/unit tests (`test_features_importance.py`, `test_features_tiers.py`,
  `test_features_evaluate.py`, `test_features_pipeline_integration.py`): 31/31 passed
  (24 unit tests across importance/tiers/evaluate, 7 pipeline-integration tests).
- Real-data tests (`test_features_pipeline_real_data.py`, `skipif`-gated): 5/5 passed on
  all three real-data pipeline runs performed during this phase (~41-62 minutes each,
  dominated by ranking-stability refits, the 12-point dense curve, and per-tier CV
  re-tuning across 3 tiers, each repeating Phase 3's own grid-search cost at a smaller
  feature count). The third run is the one reflecting the Sec.6.1 reproduction-check
  fix, and is the one `reproduction_check`'s numbers below (Sec.3) are drawn from.
- Manual, one-time determinism verification (not an automated test — see Sec.5): the
  real-data pipeline was run **three times total** against the real dataset (the second
  and third runs independently, not as retries of a failure) and `tiers.json`'s
  `membership_hash` (`ffe76f3b2c4af9ca...`) was byte-identical across all three —
  including across the Sec.6.1 code change, which touches the reproduction check but not
  tier construction itself, exactly as expected.
- Full project suite (`pytest -q`, all phases, real-data tests deselected): 241 passed,
  1 skipped (the `concrete-ml` FHE-environment smoke test, expected), 0 failed.

## 9. Deviations from `docs/plan.md`

1. **No new engineered features**, despite the phase's title ("Feature Engineering &
   Selection"). Per this session's explicit decision, Phase 4 is selection-only:
   cyclical time features (which `docs/pipeline.md` deferred "to Phase 4") and
   `card1`/`card2`/`card3`/`card5` re-encoding (flagged as an open question in
   `docs/pipeline.md` Sec.2) are explicitly NOT done here — adding them would invalidate
   the committed Phase 2 feature matrix and Phase 3 baselines this phase depends on.
2. **Ranking uses XGBoost `total_gain`**, not permutation importance (Sec.2) — the
   first-listed `docs/plan.md` option, chosen explicitly over the second.
3. **Tiers are pure top-k**, with V-block coverage reported only as a diagnostic
   (Sec.3) — `docs/eda.md`'s block-sampling suggestion is not applied to membership.
4. **Scope goes beyond the literal task list**: a dense fixed-hyperparameter curve
   (12 points, not just the 3 official tiers), ranking-stability measurement across 2
   extra seeds, and per-tier XGBoost seed stability — justified by `CLAUDE.md` Sec.11's
   general testing requirements and by this phase's own Sec.3/Sec.4 findings, which
   would not have surfaced from the 3 official tiers alone.
5. **A Phase 3 bug fix landed immediately before this phase, as its own commit**: LR's
   missing feature scaling and insufficient `max_iter` (`docs/baselines.md` Sec.7.3),
   found because a genuinely-converged LR baseline was needed for Phase 4's per-tier
   comparisons to be meaningful.
6. **One XGBoost-derived ranking defines tiers for both model types** — the shared
   tier axis may favor XGBoost (a tree-based model) over LR (a linear model), which
   Sec.3's asymmetric retention numbers (XGBoost retains far more of its full-feature
   performance than LR at every tier size) make visible, not hidden.
7. **A finding in this phase's own first version was misdiagnosed, then corrected**
   (Sec.6.1): the reproduction check's deterministic column-order-driven diff was
   initially attributed to XGBoost thread nondeterminism — a plausible-sounding but
   untested explanation. It was retracted once a second full real-data run reproduced
   the *exact same* diff bit-for-bit (ruling out nondeterminism by construction), and a
   controlled column-order experiment confirmed the real cause. Recorded here per
   `CLAUDE.md` Sec.18 (never hide that an earlier documented finding was wrong) rather
   than silently rewriting Sec.6.1 as if the correct explanation had been there from the
   start.

## 10. What Phase 4 deliberately did not do

- No FHE code of any kind (Phase 5+) — plaintext-only, per this session's explicit
  instruction.
- No modification of any Phase 1/2/3 artifact (`results/phase1_eda/`,
  `results/phase2_pipeline/`, `docs/eda.md`, `docs/pipeline.md`) — verified via `git
  status`/`git diff` showing zero changes to those paths after every real-data run.
- No block-stratified or otherwise importance-ranking-overriding tier construction
  (Sec.3/Sec.9 deviation #3).
- No automated full-pipeline-rerun determinism test in CI (Sec.5/Sec.8) — manual checks
  instead (three real-data runs performed during this phase, Sec.8), given the real
  cost of repeating a ~41-62-minute run in every CI invocation.
