# Phase 2 — Split & Leakage-Safe Preprocessing Pipeline

Implements `docs/plan.md` Phase 2: the time-based expanding-window split and the
leakage-audited preprocessing pipeline that every later phase builds on. Run as:

```bash
python -m src.data.pipeline --config configs/phase2/pipeline.yaml
```

Writes machine-readable artifacts to `results/phase2_pipeline/` and, optionally, cached
transformed feature matrices to `data/processed/` (gitignored). Uses `docs/eda.md`'s
Phase 1 findings as the source of truth for every design decision below.

## 1. Split methodology

### Boundary computation

Boundaries are quantiles of the `TransactionDT` **range** (linear interpolation between
`dt_min` and `dt_max`), the exact methodology Phase 1 established in
`src/data/eda.py::compute_candidate_boundaries` (docs/eda.md Sec.8) — **not** quantiles
of row count. docs/eda.md Sec.8 measured that transaction volume is not uniform over
time (the first 60% of the time range holds 64.5% of rows), so a row-count quantile
would not actually partition the data by time; a genuinely time-based split must use the
time axis itself as the partitioning variable.

Default config (`configs/phase2/pipeline.yaml`):

```yaml
split:
  train_val_quantile: 0.6
  val_test_quantile: 0.8
  cv_n_folds: 3
```

### Primary train/val/test split

A single threshold-based partition, computed by `src/data/split.py::assign_primary_split`:

- `train`: `TransactionDT <= train_val_boundary`
- `val`: `train_val_boundary < TransactionDT <= val_test_boundary`
- `test`: `TransactionDT > val_test_boundary`

**Test is touched exactly once**, at final model evaluation (a later phase, not this
one). This is the headline train/val/test split later phases report against.

### Expanding-window CV folds

Random k-fold cross-validation is prohibited (`CLAUDE.md` Sec.6). In its place,
`src/data/split.py::iter_expanding_window_folds` provides a genuine walk-forward
temporal CV: `cv_n_folds` boundaries are placed at quantiles of the time range strictly
within `(0, val_test_quantile]`. Fold *k*'s train window is `DT <= boundary[k]` and its
eval window is `boundary[k] < DT <= boundary[k+1]` — each fold's train window strictly
grows, its eval window always immediately follows, and **no fold's boundary ever reaches
the primary test partition** (audited directly — see §3). This gives later phases
(hyperparameter tuning, seed-stability checks) a way to get multiple temporally-valid
evaluation estimates without ever touching the final test set or shuffling rows randomly.

### Why threshold comparison, not row-index slicing

Every partition membership check is `dt <= boundary` / `dt > boundary` against a DT
*value*, never a row-index cut. This makes duplicate-DT ties safe **by construction**:
docs/eda.md Sec.2 measured 17,191 adjacent duplicate `TransactionDT` values in the real
dataset, and every row sharing an exact DT value compares identically against any
boundary, so a tied block can never be split across two partitions. This property has a
dedicated regression test
(`tests/test_split.py::test_duplicate_dt_never_straddles_boundary`) built on a
synthetic worst case, plus a direct audit check on the real split
(`src/data/leakage.py::check_no_duplicate_dt_straddles_boundary`).

### The split artifact

Per `docs/plan.md` Phase 2 ("split indices must be saved as a versioned artifact, not
regenerated ad hoc"), every run writes `results/phase2_pipeline/split_assignments.csv`
(`TransactionID`, `TransactionDT`, `split`, `cv_eval_fold`) — a literal per-row record,
not just the boundary values, so the exact partition membership stays reproducible even
if the boundary-computation code changes later. `split_boundaries.json` records the
boundary values and quantiles used to derive it.

## 2. Preprocessing methodology

### Transaction ↔ identity join

`src/data/preprocess.py::join_transaction_identity` performs a left join on
`TransactionID` and adds a `has_identity` indicator **before** merging. This is a pure
structural join — no statistic is fit, so it carries no leakage risk on its own.
docs/eda.md Sec.7 found fraud transactions are >2x as likely to carry identity data
(54.8% vs 23.3%), a genuine, prediction-time-available signal worth an explicit feature
rather than leaving implicit in a merge.

### Column roles are measured, not guessed

`src/data/preprocess.py::infer_column_roles` classifies every non-excluded column as
numeric or categorical from its **actual pandas dtype** on the joined train partition,
the same "measure, don't guess" discipline `src/data/load.py`/`schema.py` already use
(`CLAUDE.md` Sec.17) — not a hardcoded IEEE-CIS column list.

### Missing-value handling

`src/data/preprocess.py::TrainScopedMedianImputer` fits a per-column median on the
train partition **only**, and separately emits a `{col}_was_missing` 0/1 indicator
computed row-wise at transform time (needs no fitted statistic, so it is leakage-safe on
any partition, train or not).

docs/eda.md Sec.3 found a column block (`M7`, `M8`, `M9`, `V1`–`V11`, `D11`) whose
missingness rate swings from ~69–84% early in the collection window to ~28–39% late —
invisible in the global rate. A **global**-statistic fill would implicitly encode "this
row is from a later, more-complete period" into the fill value used for earlier rows.
Because every imputer here is fit strictly on whatever DataFrame is passed to `.fit()`
— and the pipeline always passes exactly `train_df`, verified by the leakage audit
(§3) — this risk doesn't require special-casing per column: the general train-only-fit
contract already covers it.

### Categorical encoding

`src/data/preprocess.py::TrainScopedFrequencyEncoder` fits a per-category frequency
table on the train partition only; an unseen category at transform time (including
identity columns for a transaction with `has_identity=0`, i.e. all-NaN) maps to `0.0` —
a legitimate encoding of "never observed in training", not a special case.

docs/eda.md Sec.6 measured **0.0% OOV** at every candidate boundary for the 14
low-cardinality categoricals profiled in Phase 1 (`ProductCD`, `card4`, `card6`,
`M1`–`M9`, `P_emaildomain`, `R_emaildomain`) — safe territory. It also explicitly
flagged `card1`/`card2`/`card3`/`card5` (numeric-typed but categorical in nature) and
the identity table's categorical columns as **not profiled** in Phase 1.

Phase 2 closes **half** of that gap, not all of it, and the actual run's measured
result (`results/phase2_pipeline/preprocessing_summary.json`) is what determined which
half: `infer_column_roles` classifies by **actual measured dtype**, and the identity
table's `id_12`, `id_15`, `id_16`, `id_23`, `id_27`–`id_38`, `DeviceType`, and
`DeviceInfo` load as `category` — so they are genuinely fit, frequency-encoded, and
OOV-measured (`oov_rate_on_val`/`oov_rate_on_test` in the summary; the real run measured
0.0% OOV for all of them). `card1`/`card2`/`card3`/`card5`, however, load as **numeric**
(`float32`) dtype in the real data, exactly as docs/eda.md Sec.6 described — so
`infer_column_roles` correctly routes them through `TrainScopedMedianImputer`
(median-fill + `_was_missing` indicator), not the frequency encoder, and they are
**not** in `oov_rate_on_val`/`oov_rate_on_test` at all. This is a defensible, measured
consequence of the project's dtype-driven classification (`CLAUDE.md` Sec.17), not a
silent gap: `card1`/`card2`/`card3`/`card5`'s suitability for frequency-encoding-as-a-
high-cardinality-identifier instead of continuous-numeric treatment remains an open
question for Phase 3/4 to revisit if their predictive behavior as median-imputed
numerics looks wrong.

### Excluded columns

```yaml
preprocess:
  exclude_columns: ["TransactionID", "TransactionDT", "isFraud"]
  passthrough_columns: ["has_identity"]
```

- **`TransactionID`**: banned per explicit project mandate. docs/eda.md Sec.2 measured
  `Spearman(TransactionID, TransactionDT) ≈ 0.99999999999974` — it is a near-perfect
  proxy for row/time order and would functionally leak temporal position into the model.
- **`TransactionDT`**: excluded from the default feature set as a **documented judgment
  call**, not a literal project mandate — for the same underlying reason as the
  `TransactionID` ban. A raw absolute time offset would let a model implicitly key off
  "which period is this row from", which cannot generalize the way a model evaluated on
  strictly-later data needs it to. This is configurable
  (`preprocess.exclude_columns`) — a future phase deriving genuine cyclical time
  features (hour-of-day, day-of-week) from it is legitimate feature engineering and
  belongs to Phase 4, not Phase 2.
- **`isFraud`**: the label, never a feature.
- **`has_identity`**: passes through unchanged — already a clean 0/1 indicator, needs
  neither imputation nor frequency encoding.

`TransactionID` is additionally used only as the output feature matrix's *index*
(`src/data/preprocess.py::build_feature_matrix`), never a column — defense in depth on
top of its exclusion from `infer_column_roles`.

## 3. Leakage protections

Every check in `src/data/leakage.py` **fails loudly**: it raises `LeakageError`
immediately rather than warning, matching the `SchemaValidationError` pattern Phase 1
established. `run_leakage_audit` runs all of them in sequence and only returns a report
if every check passes; a failing check crashes the pipeline run.

| Check | What it verifies | Catches |
|---|---|---|
| `check_split_partitions_are_disjoint_and_complete` | Every row has exactly one of `{train, val, test}`; no null labels | A row silently dropped or mislabeled |
| `check_split_is_temporally_ordered` | `max(train.DT) ≤ min(val.DT)`, `max(val.DT) ≤ min(test.DT)`, `max(train.DT) ≤ min(test.DT)` | A future row assigned to an earlier partition |
| `check_no_duplicate_dt_straddles_boundary` | No single `TransactionDT` value spans two partitions | A boundary landing inside a tied-DT block |
| `check_cv_folds_never_reach_test` | No CV fold boundary reaches the primary test partition | CV silently touching the sacred test set |
| `check_banned_columns_absent` | `TransactionID`/`TransactionDT`/`isFraud` absent from the feature matrix | A banned column leaking into `X` |
| `check_fitted_statistic_uses_only_allowed_index` | A fitted encoder/imputer's own recorded `fit_index_` is a subset of the train index | A frequency table or median computed over train+val, not just train |

The last check is the audit's teeth: every `TrainScopedFrequencyEncoder`/
`TrainScopedMedianImputer` records exactly which row index it was fit on
(`fit_index_`), so the audit **verifies**, rather than merely trusts, that
`fit_preprocessor` was only ever given `train_df`.

`tests/test_leakage.py` includes a failing-case test for every check above — each
constructs a deliberately leaky scenario (a future row labeled `train`, an encoder fit
on train+val, a tied-DT block split across a boundary, a CV boundary reaching test) and
asserts the audit catches it. A leakage check with no failing-case test would be
unverified; none here are.

## 4. Deviations from `docs/plan.md`

1. **`src/data/load.py::resolve_raw_paths`/`verify_or_report_digest` were generalized**
   out of `src/data/eda.py` (Phase 1) so Phase 2's pipeline can share the exact same
   data-integrity-check logic rather than duplicating it. `eda.py` re-exports both names
   unchanged for backward compatibility; all Phase 1 tests still pass against the
   refactor.
2. **`configs/phase2/pipeline.yaml` duplicates `configs/phase1/eda.yaml`'s `data:`
   block** (same raw file paths/hashes) rather than importing it, so each phase's config
   stays independently runnable per `CLAUDE.md` Sec.4/Sec.15 — deliberate, not
   oversight.
3. **A full single-shot load (`load_full`), not Phase 1's chunked pass, is used for both
   raw tables.** Phase 1's Step C measured a ~893MB in-memory footprint for all 378
   numeric columns; adding the 14 compact `category` columns and two `int64` id/time
   columns is a small addition, safely within this machine's 7.95GB RAM
   (`docs/environment.md`). Phase 1 needed chunking because it computed *exact*
   aggregates (missingness, null-mask hashes) across the whole file without materializing
   it; Phase 2 needs the whole joined table in memory anyway to split, fit, and
   transform it, so chunking would add complexity without a memory benefit here.
4. **Preprocessing (`src/data/preprocess.py`) lives in `src/data/`,** not a new
   top-level module. `docs/architecture.md` Sec.15 charters `src/data/` for "ingestion,
   validation, time-based split, leakage audit" without explicitly naming
   imputation/encoding, but `docs/plan.md`'s own Phase 2 "Tasks" list ("implement
   missing-data handling; implement categorical encoding") and "Deliverables" ("`src/data`
   module") place these tasks in this phase and this module.
5. **The cached transformed feature matrices use pandas pickle (`.pkl`), not Parquet.**
   `pyarrow` is not a project dependency (`docs/environment.md`), and this cache is
   private, gitignored, and same-machine only (`data/processed/`) — pickle needs no
   extra library and preserves the `float32`/`int8`/`category` dtypes exactly, unlike
   CSV. This cache is an optional Phase 3 convenience, not a Phase 2 deliverable in
   itself, and can be disabled via `output.cache_processed_data: false`.
6. **The identity table's categorical columns (`id_12`–`id_38` subset, `DeviceType`,
   `DeviceInfo`) are now frequency-encoded and OOV-measured**, closing half the gap
   docs/eda.md Sec.6 flagged as unmeasured in Phase 1. `card1`/`card2`/`card3`/`card5`
   measured as numeric-dtype in the real run and were routed through median imputation
   instead (see §2 above) — their OOV behavior as categorical identifiers remains
   unmeasured, an open question flagged for Phase 3/4, not silently resolved.
7. **A real bug was found and fixed during the first real-data pipeline run**:
   `TrainScopedFrequencyEncoder.transform` crashed (`TypeError: Cannot setitem on a
   Categorical with a new category`) because `.map()` on a `category`-dtype column
   returns a Categorical container whose categories are the mapped frequency values,
   and `.fillna(0.0)` then fails unless `0.0` already happens to be one of them. This
   never appeared against the synthetic test fixtures used while building this module
   because none of them exercised a `category`-dtype column through the full
   `.map().fillna()` path — a real gap in test coverage, not caught until the real-data
   pipeline run. Fixed by casting to `float64` before `fillna`
   (`src/data/preprocess.py`), and closed with a dedicated regression test using an
   actual `pd.Categorical` column
   (`tests/test_preprocess.py::test_frequency_encoder_handles_category_dtype_column`).

## 5. What Phase 2 deliberately did not do

- No model training of any kind (Phase 3).
- No feature engineering beyond `has_identity` (a join-presence indicator with no
  fitted statistic, and therefore no leakage risk) and the two preprocessing artifacts
  (`_freq`, `_was_missing`) that are the literal Phase 2 "categorical encoding" /
  "missing-data handling" tasks. Importance-ranked feature tiers are Phase 4.
- No persisted encoders/imputers for every individual CV fold — Phase 2 validates the
  fit/transform/audit machinery once on the primary train/val/test split and provides
  `iter_expanding_window_folds` as tested, reusable machinery; fitting per-CV-fold models
  is Phase 3's concern when it actually trains something.
- No modification of any Phase 1 artifact (`results/phase1_eda/`, `docs/eda.md`).
