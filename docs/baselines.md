# Phase 3 — Baseline Models

Implements `docs/plan.md` Phase 3: Logistic Regression and XGBoost baselines,
trained and evaluated exactly per `CLAUDE.md` Sec.6. Run as:

```bash
python -m src.train.pipeline --config configs/phase3/baselines.yaml
```

Writes trained model artifacts to `models/phase3_baselines/` (gitignored via the
`models/` rule) and machine-readable metrics/provenance to `results/phase3_baselines/`.

## 1. Data provenance: Phase 2 is the only source

`src/train/data.py::load_phase2_features` is the **sole** entry point every model in
this phase uses to obtain features. It never re-derives loading, joining, splitting, or
preprocessing logic — everything comes from `src.data.pipeline`, either:

- a **verified-fresh cache** (`data/processed/{train,val,test}.pkl`), trusted only if
  `results/phase2_pipeline/provenance.json`'s recorded `config_hash` matches the
  *current* `configs/phase2/pipeline.yaml` content, or
- a **fresh re-run** of `src.data.pipeline.run()` when the config changed or the cache
  is missing — which re-verifies raw-file SHA256s and re-runs Phase 2's leakage audit.

This means Phase 3 can never silently train against a stale or hand-edited cache.

## 2. Training / evaluation methodology

### Split usage

| Partition | Used for |
|---|---|
| `train` | Final refit of both models (after hyperparameter selection) |
| `train + val` (`features.X_train_plus_val`, Sec.6) | Hyperparameter selection via CV only — never the final refit |
| `val` | XGBoost's `n_estimators` (early stopping), headline evaluation metrics, threshold selection |
| Phase 2's expanding-window **CV folds** | Hyperparameter selection (LR's `C`; XGBoost's `max_depth`/`learning_rate`) |
| `test` | **Not used anywhere in this phase** (see Sec.5) |

### Hyperparameter selection uses Phase 2's expanding-window CV, never random k-fold

`src/train/cv.py::build_positional_cv_folds` adapts
`src/data/split.py::iter_expanding_window_folds` (3 folds by default, spanning the
primary train+val region, never reaching test) into positional index arrays. For each
candidate hyperparameter setting, both models are fit on each fold's train window and
scored (PR-AUC) on that fold's eval window; the setting with the best **mean CV PR-AUC**
is selected.

- **Logistic Regression** (`src/train/logistic_regression.py`): grid over
  `C ∈ {0.01, 0.1, 1.0, 10.0}` (`configs/phase3/baselines.yaml`), `solver="lbfgs"`,
  `max_iter=200`. The selected `C` is refit on the **full** train partition.
- **XGBoost** (`src/train/xgboost_model.py`): grid over `max_depth ∈ {4, 6}` ×
  `learning_rate ∈ {0.05, 0.1}` (4 combos). `n_estimators` is never grid-searched
  directly — each fit uses `early_stopping_rounds=20` monitoring `aucpr` (XGBoost's
  PR-AUC-equivalent metric) on that fold's own eval slice. The selected `(max_depth,
  learning_rate)` combo is refit on the full train partition, this time with early
  stopping against **`X_val`** — this is the concrete meaning of "use validation data for
  model selection" for XGBoost: it decides the effective tree count, not the grid.

Both grids are deliberately small (4 settings each). Phase 3's job is a **baseline**,
not the systematic FHE-focused research sweep — that is Phase 8's explicit scope
(`docs/plan.md`).

### Why a plain `sklearn`/`xgboost` model, not a `Pipeline`

Both models are trained as a plain `sklearn.linear_model.LogisticRegression` /
`xgboost.XGBClassifier` over the already-preprocessed float32 feature matrix, not
wrapped in an `sklearn.Pipeline` with custom transformers. `concrete.ml.sklearn.LogisticRegression`
/ `XGBClassifier` (Phase 5+) are drop-in, quantization-aware replacements for exactly
these classes over the same feature matrix — preprocessing itself stays client-side and
plaintext per `docs/architecture.md` Sec.6, so nothing about *how* these models are fit
needs to change for FHE compatibility. No FHE code is introduced in this phase.

## 3. Metrics (`src/train/metrics.py`)

Per `CLAUDE.md` Sec.6, **PR-AUC is the primary metric**; accuracy is never reported.
Computed on `val` only:

- `threshold_independent_metrics`: PR-AUC (`sklearn.metrics.average_precision_score`,
  the exact area under the precision-recall curve) and ROC-AUC.
- `metrics_at_threshold`: precision, recall, F1, **F2** (weights recall over precision —
  appropriate here since a missed fraud is typically costlier than a false alarm), and a
  confusion matrix, at a given threshold.
- `select_threshold`: scans `precision_recall_curve`'s threshold grid and picks the
  threshold maximizing F1 or F2 — **always called with `(X_val, y_val)`, never
  `(X_test, y_test)`**, in `src/train/pipeline.py`. This is a leakage-relevant boundary,
  not a convention: a threshold selected against test would let test information shape
  a decision the model is later "evaluated" against.
- `error_analysis`: false-positive / false-negative counts, rates, and a capped sample
  of `TransactionID`s, at the F1-selected threshold of whichever model has the higher
  val PR-AUC (CLAUDE.md Sec.6: "error analysis... for at least the strongest plaintext
  model").

## 4. Real-dataset validation results

Produced by `python -m src.train.pipeline --config configs/phase3/baselines.yaml`
against the real IEEE-CIS data (`results/phase3_baselines/metrics.json`,
`git_commit` `8732dcb`, `config_hash` `5e8e5711171533dd`). Split sizes: train 380,815 /
val 105,088 / test 104,637 (test untouched, Sec.5).

| Metric (on `val`) | Logistic Regression | XGBoost |
|---|---|---|
| Selected hyperparameters | `C = 0.01` | `max_depth = 6`, `learning_rate = 0.05`, `n_estimators = 495` |
| Mean CV PR-AUC (selection criterion) | 0.1669 | 0.5406 |
| **PR-AUC** | **0.2036** | **0.5711** |
| ROC-AUC | 0.7783 | 0.9143 |
| F1 (at F1-optimal threshold) | 0.2827 (P=0.2869, R=0.2787, thr=0.706) | 0.5554 (P=0.6065, R=0.5122, thr=0.759) |
| F2 (at F2-optimal threshold) | 0.3581 (P=0.1567, R=0.5274, thr=0.586) | 0.5766 (P=0.4144, R=0.6391, thr=0.602) |
| Confusion matrix @ F1-optimal threshold | TN=98,164 FP=2,834 / FN=2,950 TP=1,140 | TN=99,639 FP=1,359 / FN=1,995 TP=2,095 |

Both models clear the val PR-AUC baseline the real-data test suite enforces
(`tests/test_train_pipeline_real_data.py::test_pipeline_real_data_pr_auc_beats_baseline`,
threshold 0.15 — well above the ~0.035 PR-AUC a random/constant classifier would score
at this fraud rate). **XGBoost is the strongest model** (PR-AUC 0.571 vs. 0.204) and is
therefore the one `error_analysis` runs against, at its F1-optimal threshold (0.759):
1,359 false positives (1.35% of negatives) and 1,995 false negatives (48.8% of
positives) on `val`.

Seed stability (`tests/test_train_pipeline_real_data.py`, reusing the already-selected
hyperparameters rather than re-running CV): Logistic Regression is exactly
deterministic across seeds (`lbfgs` is a deterministic solver given fixed data/params,
std < 0.001 asserted); XGBoost — nondeterministic due to `subsample`/`colsample_bytree`
row/feature sampling — stays within the documented std < 0.02 band. Both checks passed;
see Sec.9 for the full real-data test run.

## 5. Class-imbalance handling (`src/train/imbalance.py`)

docs/eda.md Sec.1 measured a 3.499% global fraud rate. Per `docs/plan.md`'s Phase 3
fallback strategy ("resampling, class weights, focal-loss-style objectives"), this
project uses **class weighting** — the first-listed, simplest option — for two concrete
reasons, not just because it's listed first:

1. No new dependency: resampling (SMOTE-style) would need `imbalanced-learn`, not
   currently justified by any other requirement.
2. Natively supported by both required model types (`class_weight` in scikit-learn,
   `scale_pos_weight` in XGBoost), so no custom resampler needs its own train-only-fit
   leakage discipline (unlike `src/data/preprocess.py`'s encoders, which do).

Both weight statistics are computed from **train only** — for the CV-based
hyperparameter selection, each fold recomputes its own weight from its own train
subset (the same train-only-fit discipline Phase 2's `TrainScopedFrequencyEncoder`/
`TrainScopedMedianImputer` use); for the final refit, from the full train partition.
Computing it over train+val (or worse, the full dataset) would leak validation-period
class balance into training.

- **Logistic Regression**: `class_weight_dict` — scikit-learn's exact `'balanced'`
  formula (`n_samples / (n_classes * n_samples_of_class_c)`), computed explicitly rather
  than passing the `'balanced'` string, so the actual numbers are visible in
  `results/phase3_baselines/metrics.json` rather than only implicit in a fitted
  estimator's internals.
- **XGBoost**: `compute_scale_pos_weight` — `n_negative / n_positive`, XGBoost's own
  documented closed-form recommendation for imbalanced binary classification.

## 6. The test partition stays untouched

`src/train/pipeline.py::run()` loads `features.X_test`/`features.y_test` (Phase 2's
`load_phase2_features` always returns them) but **never references them anywhere in the
function body** — no training call, no evaluation call, no threshold selection. The one
exception is `len(features.X_test)`, reported in `split_sizes` purely as a row count for
bookkeeping — this reveals nothing about the test partition's feature values or labels
and is treated the same as Phase 2's own `preprocessing_summary.json` reporting split
sizes.

This is verified mechanically, not just asserted in prose:
`tests/test_train_pipeline_integration.py::test_test_partition_never_touched` passes a
sentinel object as `X_test`/`y_test` that raises `AssertionError` on any access except
`__len__`, and asserts the full pipeline still completes successfully. Confirmed again
on the real dataset by
`tests/test_train_pipeline_real_data.py::test_pipeline_real_data_test_partition_not_in_results`.

**"The project's designated final evaluation"** (per this session's instructions) is
explicitly deferred to a later phase — most plausibly Phase 8 (Core Research
Experiments) or the final report — not Phase 3. `results/phase3_baselines/metrics.json`
carries `"test_partition_touched": false` as a durable, checkable record of this.

## 7. Two real behaviors verified empirically (not assumed)

### 7.1 The CV-pool must be train+val, not train-only — a real bug, found and fixed

The first real-data run of `tests/test_train_pipeline_real_data.py` crashed with
`ValueError: Found array with 0 sample(s) ... while a minimum of 1 is required by
LogisticRegression`. Root cause: Phase 2's expanding-window CV folds
(`compute_split_boundaries`) span the **primary train+val region** — `(0,
val_test_boundary]` — by design (`docs/pipeline.md` Sec.1), because CV exists to select
hyperparameters using more than just the train-only slice. But `src/train/data.py`
originally only exposed `features.X_train`/`features.train_dt` (capped at
`train_val_boundary`) to the CV-selection functions. The last CV fold's eval window
falls between `train_val_boundary` and `val_test_boundary` — i.e., inside the primary
"val" partition's own DT range — so those rows existed only in `features.X_val`, never
in `features.X_train`, and the fold's eval slice was empty.

This was not caught by the initial unit tests for `logistic_regression.py`/
`xgboost_model.py` because their synthetic fixtures happened to pass one combined
dataset spanning the full train+val(+test-adjacent) DT range for **both** the
CV-selection role and the (at the time, single) train role — the bug only manifests when
a genuinely train-only `X_train` is passed where a train+val pool is required, which the
real Phase 2 data does but the original fixtures did not.

**Fix**: `Phase2Features` (`src/train/data.py`) gained three new fields —
`X_train_plus_val`, `y_train_plus_val`, `train_plus_val_dt` — built via
`pd.concat([X_train, X_val])` etc. `select_c_via_cv`/`train_logistic_regression`
(`src/train/logistic_regression.py`) and `select_hyperparameters_via_cv`/`train_xgboost`
(`src/train/xgboost_model.py`) now take separate `(X_train, y_train)` (train-only, final
refit) and `(X_cv, y_cv, cv_dt)` (train+val, CV selection only) arguments, and
`src/train/pipeline.py` passes `features.X_train_plus_val` etc. for the CV role. The
test fixtures in `tests/test_train_logistic_regression.py`,
`tests/test_train_xgboost_model.py`, and `tests/test_train_pipeline_integration.py` were
rewritten to keep train-only and train+val genuinely distinct (mirroring the real
`Phase2Features` shape), and
`tests/test_train_logistic_regression.py::test_train_logistic_regression_cv_pool_larger_than_train_only`
was added as an explicit regression guard (`len(X_cv) > len(X_train)`). The full
real-data suite subsequently passed end-to-end (Sec.9).

### 7.2 XGBoost's `num_boosted_rounds()` vs. `best_iteration`

While testing `src/train/xgboost_model.py`, `num_boosted_rounds()` on a fit model did
**not** equal `best_iteration + 1` (the count this project uses as
`selected_n_estimators`) — XGBoost trains `early_stopping_rounds` (20) additional trees
past the best iteration as its patience window, and retains all of them in the booster.
Verified directly (not assumed) that this doesn't affect correctness:
`model.predict_proba()` automatically truncates to `best_iteration` by default, and —
critically, since `src/train/pipeline.py` saves XGBoost via its native
`save_model()`/`load_model()` (JSON format), not `joblib`/pickle — `best_iteration` is
preserved across a save/reload round trip, and a reloaded model's `predict_proba()`
matches the original's exactly with no extra caller action needed. Regression test:
`tests/test_train_xgboost_model.py::test_train_xgboost_save_and_reload_preserves_early_stopping_predictions`.

## 8. Reproducibility

`configs/phase3/baselines.yaml` fixes `seed: 42`, used for every stochastic step (LR's
`random_state`, XGBoost's `random_state`). `results/phase3_baselines/provenance.json`
records the config hash, git commit, timestamp, and library versions
(`src/data/provenance.py::build_provenance`, the same helper Phase 1/2 use).

## 9. Test suite

- Synthetic/unit tests (`test_train_metrics.py`, `test_train_imbalance.py`,
  `test_train_cv.py`, `test_train_logistic_regression.py`, `test_train_xgboost_model.py`,
  `test_train_data.py`, `test_train_pipeline_integration.py`): 51/51 passed.
- Real-data tests (`test_train_pipeline_real_data.py`, `skipif`-gated on the real Phase 2
  cache being available): 6/6 passed, ~42 minutes (dominated by XGBoost's CV grid search
  over ~380K rows × 832 features) — end-to-end pipeline run, PR-AUC-beats-baseline check,
  non-accuracy-only metric check, test-partition-not-in-results check, and both
  seed-stability checks.
- Full project suite (`pytest -q`, all phases): 215 passed, 1 skipped (the
  `concrete-ml` FHE-environment smoke test, expected — no Windows wheels; runs under the
  WSL2 venv per `docs/environment.md`), 0 failed.

## 10. Deviations from `docs/plan.md`

1. **Two small hyperparameter grids are searched** (LR's `C`; XGBoost's
   `max_depth`/`learning_rate`) via Phase 2's CV folds, beyond `docs/plan.md`'s literal
   Phase 3 task list ("train LR and XGBoost with class-imbalance handling"). Justified
   by this session's explicit requirement to "use validation data for model
   selection/tuning" and to "preserve the temporal/expanding-window evaluation design
   from Phase 2" — CV folds would otherwise go entirely unused in this phase.
2. **`src/train/` groups data loading, metrics, imbalance handling, CV adaptation, and
   both models** rather than a single monolithic training script, for the same
   testability reasons `src/data/` was split in Phases 1–2 (`CLAUDE.md` Sec.15: "small
   testable units").
3. **Model artifacts use `joblib` (LR) and XGBoost's native JSON format (XGBoost)**, not
   a single uniform serialization scheme — each is the standard, most portable format
   for its respective library, and XGBoost's native format is specifically what
   preserves `best_iteration` correctly (Sec.7.2 above).
4. **A test-assertion bug was found and fixed** while building
   `src/train/xgboost_model.py`'s test suite: an incorrect assertion (not application
   code) compared `num_boosted_rounds()` to `best_iteration + 1` directly; investigated
   empirically (Sec.7.2) rather than assumed, and corrected to check the actually
   load-bearing property (prediction correctness after save/reload).
5. **A real application-code bug was found and fixed on the real dataset**: the
   CV-pool/train-only routing bug documented in Sec.7.1. `Phase2Features` gained
   `X_train_plus_val`/`y_train_plus_val`/`train_plus_val_dt`, and both models' CV
   selection functions were repointed at the train+val pool while the final refit
   stayed train-only. Not anticipated in the original design; found only once the real
   Phase 2 data (not the initially too-permissive synthetic fixtures) was used.
