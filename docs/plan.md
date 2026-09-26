# Implementation Plan — Phased Roadmap

Central research question (must stay consistent across all phases):

> How do feature count, quantization bit-width, and ML model complexity
> affect the predictive performance and computational cost of
> privacy-preserving fraud detection under FHE?

---

## Phase 0 — Environment & Repository

**Objective**: reproducible environment and repository skeleton.

**Tasks**: initialize repo with structure from `architecture.md` §20;
pin Python version; pin Concrete-ML, Concrete-Python, XGBoost,
scikit-learn, pandas versions; set up config-loading utility; set up
structured logging; set up test runner.

**Technical details**: record exact library versions in a
`requirements.lock` or equivalent; document the actual CPU/RAM available
for FHE benchmarking (this constrains later phases).

**Deliverables**: initialized repo, pinned environment, empty pipeline
module skeletons, CI-less local test runner working on a trivial test.

**Tests**: environment smoke test (imports succeed, versions match lock
file).

**Exit criteria**: `pytest` runs (even with zero real tests yet);
documented hardware spec committed to repo.

**Risks/blockers**: Concrete-ML installation issues on the target OS/
hardware (known to have platform-specific constraints).

**Fallback strategy**: if local hardware cannot install/run Concrete-ML
at all, document the specific failure and evaluate a cloud VM with
adequate specs — do not substitute a different, easier FHE library
without documenting why Concrete-ML specifically failed.

---

## Phase 1 — Dataset & EDA

**Objective**: understand IEEE-CIS's structure, missingness, and
imbalance before writing any pipeline code.

**Tasks**: load transaction + identity tables; document schema, missing-
value rates per column group, class imbalance ratio, `TransactionDT`
range/semantics; identify high-cardinality/anonymized (`V`-column) groups.

**Technical details**: confirm `TransactionDT` is a delta, not an
absolute timestamp, and confirm implications for split design.

**Deliverables**: EDA notebook/report with missingness table, imbalance
ratio, cardinality summary.

**Tests**: none (exploratory), but findings must be committed as a
markdown/notebook artifact.

**Exit criteria**: documented understanding of missingness, imbalance,
and time semantics sufficient to design the split and preprocessing in
Phase 2.

**Risks/blockers**: dataset size making full EDA slow on limited
hardware.

**Fallback strategy**: sample-based EDA on a stratified subset, clearly
labeled as such, while full pipeline still runs on complete data later.

**Implementation deviations (recorded per CLAUDE.md Sec.2)**:
1. **Chunked-exact aggregation used instead of the stratified-sample
   fallback above**, as the default/primary strategy rather than a
   fallback. Sampling cannot establish exact positive counts in tail time
   buckets (which set split boundaries), exact all-null/constant columns,
   or exact null-mask identity between V columns -- chunked aggregation
   gives exact statistics in bounded memory instead. The sampling
   fallback remains available if chunked aggregation proves infeasible.
2. **`src/data/load.py` and `src/data/schema.py` were built in Phase 1**,
   ahead of where this document lists "src/data module" (Phase 2).
   Phase 2 adds `split.py` and `leakage.py` on top of this foundation.
   Consistent with `architecture.md` Sec.15's component charter.
3. **Deliverable is scripts + a generated Markdown report**
   (`src/data/eda.py`, run as `python -m src.data.eda`, writing
   `results/phase1_eda/` and `docs/eda.md`), not a notebook. Chosen for
   determinism, testability, and git-diffability (CLAUDE.md Sec.15);
   this document's "notebook/report" wording permits either.
4. **A small test suite exists** (`tests/test_schema.py`,
   `tests/test_load.py`, `tests/test_eda.py`,
   `tests/test_eda_integration.py`, `tests/test_eda_real_data.py`)
   where this section specifies none, because the validator and
   aggregation logic is real logic that CLAUDE.md Sec.11/Sec.19 require
   tested. All but the last run on synthetic data; the real-data test is
   skipif-gated on `data/raw/` being populated.
5. **`matplotlib` added to `requirements.txt`**, justified by
   `docs/prd.md` FR10 (Pareto-frontier plots are a hard Phase 8
   requirement regardless); Phase 1 uses it for four EDA plots.

---

## Phase 2 — Leakage-Safe ML Pipeline

**Objective**: build the time-based split and leakage-audited
preprocessing pipeline — the foundation everything else depends on.

**Tasks**: implement expanding-window time-based split by
`TransactionDT`; implement missing-data handling; implement categorical
encoding (e.g., frequency encoding using only past-observed data per the
split); implement and run the leakage audit for every engineered feature.

**Technical details**: split indices must be saved as a versioned
artifact, not regenerated ad hoc by later phases.

**Deliverables**: `src/data` module, saved split artifact, leakage audit
report.

**Tests**: unit tests asserting no test-set or future-window information
leaks into training-set feature computation.

**Exit criteria**: leakage audit passes for every engineered feature;
split artifact is deterministic and reproducible from config.

**Risks/blockers**: subtle leakage in categorical frequency encoding if
not carefully scoped to past-only data.

**Fallback strategy**: if a particular feature construction cannot be
made leakage-safe within available time, drop that specific feature and
document why, rather than shipping a leaky pipeline.

---

## Phase 3 — Baseline Models

**Objective**: train and evaluate Logistic Regression and XGBoost on the
full feature set (plaintext) as the ML baseline.

**Tasks**: train LR and XGBoost with class-imbalance handling; evaluate
with PR-AUC, ROC-AUC, precision, recall, F1, F2; produce confusion
matrices and threshold analysis.

**Technical details**: use the fixed split artifact from Phase 2; log
all hyperparameters via config.

**Deliverables**: trained baseline models, evaluation report.

**Tests**: metric-computation unit tests; seed-stability check (metrics
stable within a documented variance band across seeds).

**Exit criteria**: both baselines trained, evaluated, and results
committed with config files.

**Risks/blockers**: class imbalance handling insufficient, producing
misleadingly high accuracy but poor PR-AUC.

**Fallback strategy**: iterate on imbalance handling (resampling, class
weights, focal-loss-style objectives for XGBoost) until PR-AUC is
genuinely reasonable — do not accept a baseline with poor PR-AUC and
move on regardless.

---

## Phase 4 — Feature Engineering & Selection

**Objective**: produce a real, ranked feature-importance list and define
the feature-count tiers used throughout the FHE experiments.

**Tasks**: compute feature importances from the Phase 3 XGBoost model
(or permutation importance); define at least 3 feature-count tiers
(e.g., top-20/top-50/top-100); re-evaluate LR/XGBoost on each tier to
quantify the plaintext accuracy cost of feature reduction before FHE is
even introduced.

**Technical details**: tiers are versioned artifacts referenced by
config in all later FHE phases.

**Deliverables**: feature importance ranking, tier definitions,
plaintext accuracy-vs-feature-count curve (a precursor to the eventual
FHE Pareto frontier).

**Tests**: tier membership determinism test.

**Exit criteria**: tiers defined, plaintext PR-AUC recorded per tier.

**Risks/blockers**: feature reduction degrading accuracy more than
expected, especially given how much signal is concentrated in the
anonymized `V` columns.

**Fallback strategy**: adjust tier sizes based on where the plaintext
accuracy-vs-feature-count curve actually bends, rather than sticking
rigidly to arbitrary round numbers.

---

## Phase 5 — FHE Proof of Concept

**Objective**: get one model (start with Logistic Regression, the
simplest) compiling and running correctly under Concrete-ML end-to-end
before building the full grid.

**Tasks**: quantize and compile LR on the smallest feature tier; run
encrypted inference on a small held-out sample; validate against
plaintext output.

**Technical details**: this phase exists specifically to surface
Concrete-ML environment/version issues early, on the simplest possible
case.

**Deliverables**: one working encrypted LR inference path with a
correctness report.

**Tests**: encrypted-vs-plaintext agreement test.

**Exit criteria**: at least one full encrypt→infer→decrypt round trip
completes correctly.

**Risks/blockers**: compilation failures, environment issues, or
unexpectedly long compile/inference times even for the simplest case.

**Fallback strategy**: if the smallest possible configuration cannot run
in reasonable time on available hardware, document the exact resource
usage observed and consider cloud compute before considering any
reduction of the research question itself.

**Status: CLOSED, exit criterion met.** At least one full encrypt→infer→decrypt
round trip completes correctly (T1, `docs/fhe_poc.md` Sec.5) -- the literal
exit criterion above, independently re-verified at `n_bits=8` after a config
fix (`docs/fhe_poc.md` Sec.9.4). This session's own additional accuracy gate
(T3) fails at both `n_bits=8` and `n_bits=16` and is preserved as a
documented, unresolved research finding (`docs/fhe_poc.md` Sec.8-9) -- not
remediated, not silently dropped. Phase 5 is closed per this phase's own
written exit criterion; the T3 limitation carries forward into Phase 6 and
is independently reproduced there for XGBoost (below).

**Erratum (2026-09-25):** the LR T3 failure recorded above was mostly caused by
a defect found later: the compiled LR consumed raw features while its
coefficients were trained on standardized features. It is fixed, and the
corrected LR results (16-bit passes T3 at every tier; 8-bit still fails) are in
`docs/research.md` §5. The "independently reproduced for XGBoost" claim does not
apply to LR; the Phase 5 result files are left unmodified as history.

---

## Phase 6 — FHE Fraud Inference (XGBoost)

**Objective**: extend the working FHE path to XGBoost across the defined
feature tiers.

**Tasks**: quantize/compile XGBoost per feature tier at an initial
bit-width; validate correctness per configuration; record compile time.

**Technical details**: use Concrete-ML's XGBClassifier with `n_bits`
controlling input precision and a calibration-set-driven bit-width
computation, per `architecture.md` §5.

**Deliverables**: working encrypted XGBoost inference across all feature
tiers at one initial bit-width setting.

**Tests**: correctness validation per (tier) configuration.

**Exit criteria**: all feature tiers compile and validate correctly for
XGBoost at the initial bit-width.

**Risks/blockers**: larger trees/feature tiers hitting compile-time or
memory limits.

**Fallback strategy**: if a specific tier fails to compile/run in
feasible time, document the specific limitation (tree depth, number of
trees, feature count) rather than silently shrinking scope elsewhere.

**Status: CLOSED, exit criterion met.** "Validate correctly" is defined
consistently elsewhere in this document set as circuit output vs. plaintext
(quantized) output, not quantized-vs-float accuracy: `architecture.md` §5
("validate compiled-circuit output against plaintext output"), `prd.md` FR7,
and `instructions.md`'s FHE Rules all define it the same way. By that
definition, all three tiers compile and validate correctly for XGBoost at
`n_bits=8`: T0/T1/T2 pass exactly for `top_20`, `top_50`, and `top_100`,
including a real (non-simulated) encrypt→run→decrypt round trip per tier
whose decrypted output matches that circuit's own simulation bit-for-bit
(`docs/fhe_xgboost.md` Sec.5). This phase's exit criterion is met.

T3 -- this project's own additional accuracy gate (quantized output vs. the
float reference; see `docs/fhe_poc.md`'s framing of the same gate for LR) --
fails for every tier at `n_bits=8`: decision agreement 0.649-0.894 (need
≥0.99), PR-AUC dropping 0.19-0.29 (need ≤0.01). This is the same
quantization limitation Phase 5 found for Logistic Regression, independently
reproduced here for a structurally different model (`docs/fhe_xgboost.md`
Sec.7). It is preserved as a documented research finding, not remediated: no
tree depth/count/feature count was reduced and no tolerance was loosened to
force a pass, per this phase's own fallback strategy above. The systematic
investigation this finding calls for -- the bit-width dimension of this
project's central research question -- is Phase 8's explicit mandate (the
feature-count × bit-width × model-type grid), not Phase 6's. Compile times,
PBS counts, and real per-row execution times (20.6-31.5 minutes/row,
dominated by `run`) were recorded as observations, not benchmarks -- Phase
7's scope. Deviations from this plan's implicit assumptions (calibration
size, T2/T1 sample sizes, T0 tolerance) are in `docs/fhe_xgboost.md` Sec.8,
all measured and justified, none guessed.

---

## Phase 7 — Benchmark Infrastructure

**Objective**: build the config-driven benchmark harness before running
the full experimental grid.

**Tasks**: implement repeated-trial latency/memory/ciphertext-size
measurement; implement results storage keyed by config hash; implement
smoke-test mode (small grid, few trials).

**Technical details**: separate compilation-time measurement from
per-request inference-latency measurement, per `architecture.md` §16.

**Deliverables**: working benchmark runner, passing smoke test.

**Tests**: smoke test on a minimal 2-3-configuration grid.

**Exit criteria**: smoke test passes with sane, reproducible output.

**Risks/blockers**: measurement noise from other system load affecting
latency figures.

**Fallback strategy**: run benchmarks on a quiesced machine/VM and
document conditions; increase trial count if variance is high rather
than reporting noisy single-trial numbers.

**Status: CLOSED, exit criterion met.** The config-driven harness
(`src/benchmark/`) implements every required measurement -- repeated-trial
latency (plaintext and FHE) with mean/std, peak memory, ciphertext size,
compile time kept separate from inference latency, and config-hash-keyed
structured result storage -- by reusing, never re-validating, the existing
Phase 5/6 compile/handoff/provenance infrastructure. The smoke test
(`configs/phase7/smoke.yaml`, 2 configurations, 2 trials each) passed with
sane, fully reproducible output: a Logistic Regression configuration (zero
programmable bootstraps, ~99ms real FHE round trip) and an XGBoost
configuration (`top_50`, 151,986 bootstraps, ~22.6-minute real FHE round
trip), both with bit-identical decrypted output across every trial
(`docs/benchmark.md` Sec.6). No Phase 1-6 model, tier, config, or result was
touched. Deviations (2 configurations rather than 3, 2 trials rather than
Phase 8's eventual ≥5) are documented in `docs/benchmark.md` Sec.8, both
explicitly permitted by this phase's own "few trials"/"2-3" wording, not
guessed.

---

## Phase 8 — Core Research Experiments

**Objective**: execute the full feature-count × bit-width × model-type
(LR, XGBoost) benchmark grid — the heart of the project's research
contribution.

**Tasks**: run the full grid per `prd.md` §12 (minimum 3 tiers × 2
bit-widths × models, ≥5 trials each); generate accuracy-vs-latency
Pareto-frontier plots; sanity-check against cited prior-art latency
numbers.

**Technical details**: every run tied to a config file; results stored
with full provenance (config hash, git commit, timestamp).

**Deliverables**: complete benchmark results (CSV/JSON), Pareto-frontier
plots, comparison table against prior-art numbers.

**Tests**: results-schema validation; spot-check that reported means
match raw per-trial data.

**Exit criteria**: full grid executed for LR and XGBoost; Pareto
frontiers generated; prior-art comparison written up (agreement or
honest divergence explanation).

**Risks/blockers**: total grid runtime exceeding available time/compute
budget.

**Fallback strategy**: reduce grid density (fewer bit-width steps) with
a documented rationale before reducing trial count (variance reporting
is higher priority than grid density).

**Status: CLOSED, exit criteria met (2026-09-26); full write-up in `docs/research.md`.**
- **Full grid executed:** 12 of 12 configurations (LR and XGBoost × `top_20`/`top_50`/`top_100` × two bit-widths each), 5 repeated executions of one fixed input per configuration, plus a 2-row real encrypt→run→decrypt correctness sample per configuration. T0/T1/T2 pass in all 12; T3 passes in 9 (LR 16 bits and XGBoost 14 bits at all three tiers) and fails in 3+3 (all 8-bit LR and XGBoost configurations), which is a reported finding, not a defect.
- **Pareto frontiers generated:** accuracy vs latency, vs peak memory and vs ciphertext size, plus feature-count / bit-width / model effect tables (`python -m src.analysis.phase8_pareto`; outputs in `results/phase8_research/pareto/`).
- **Prior-art comparison written up as an honest divergence:** the ~6 ms XGBoost figure in `docs/prd.md` §11 is not reproduced (measured 931–4,099 s); no explanation is asserted (`docs/research.md` §6.4).
- **Tests:** results-schema validation and means-vs-raw-trial checks with negative controls (`tests/test_phase8_results.py`); runner tests (`tests/test_phase8_grid.py`).
- **Deviations from this phase's task text** (all recorded in `docs/research.md` §9): bit-widths differ by model (LR 8/16, XGBoost 8/14; XGBoost 15/16 do not compile — a measured Concrete limit); latency is repeated executions of one input, not independent samples; a 2-row rather than 100-row correctness sample for XGBoost (cost); committed Phase 5/7 **code** was edited to fix the LR `StandardScaler` defect (`docs/research.md` §5) while committed Phase 1–7 **result** files were not modified; the per-configuration `provenance.json` files record `git_commit = e85d16a` (the pre-Phase-8 commit) because the Phase 8 code was uncommitted when the runs happened.
- **Not established by Phase 8:** test-partition accuracy, latency across different inputs, more than two bit-widths per model, any GPU/cloud run (`docs/research.md` §11).

---

## Phase 9 — Quantized MLP Extension

**Objective**: extend the same benchmark grid to the quantized MLP model
type.

**Tasks**: train/quantize (e.g., via Brevitas) a small MLP; compile via
Concrete-ML; validate correctness; run the same feature-tier × bit-width
grid as Phases 6–8, respecting the accumulator bit-width ceiling
constraint from `architecture.md` §5.

**Technical details**: log TLU count and compile time per configuration,
per `instructions.md` FHE rules.

**Deliverables**: MLP results integrated into the same Pareto-frontier
plots as LR/XGBoost, enabling a genuine tree-vs-neural-net comparison
under FHE.

**Tests**: correctness validation per MLP configuration.

**Exit criteria**: MLP results present across at least the feature tiers
and bit-widths that are actually feasible under the accumulator
constraint; infeasible combinations explicitly logged as such.

**Risks/blockers**: accumulator bit-width ceiling blocking larger
feature-count/bit-width combinations for the MLP specifically (this is
expected — see `prd.md` §9 and the technical assessment).

**Fallback strategy**: characterize and report the accumulator
constraint's practical effect (i.e., "MLP could not run beyond tier X at
bit-width Y due to accumulator overflow") as a first-class finding rather
than treating it as a project failure.

---

## Phase 10 — Client/Server Implementation

**Objective**: build the real encrypt→send→infer→return→decrypt demo.

**Tasks**: implement client (key gen, quantize, encrypt, decrypt) and
server (FastAPI endpoint serving encrypted inference) per
`architecture.md` §6–7; run at least one full round trip per model type.

**Deliverables**: working client/server demo, integration test passing.

**Tests**: full round-trip integration test per model type.

**Exit criteria**: at least one successful end-to-end encrypted request
per model type, matching the correctness-validated compiled model from
earlier phases.

**Risks/blockers**: serialization/transport overhead for ciphertext not
accounted for in earlier local benchmarks.

**Fallback strategy**: measure and report network/serialization overhead
separately from raw FHE compute latency, rather than conflating them.

---

## Phase 11 — API / Docker (optional polish)

**Objective**: package the client/server demo for easier review.

**Tasks**: containerize the server (and optionally client) via Docker,
only after Phases 0–10 are complete.

**Deliverables**: Dockerfile(s), run instructions.

**Tests**: container starts and passes the same integration test as
Phase 10.

**Exit criteria**: containerized demo runs identically to the
non-containerized version.

**Risks/blockers**: Concrete-ML's native-code dependencies complicating
containerization.

**Fallback strategy**: if containerization proves disproportionately
time-consuming, document the attempt and ship the non-containerized demo
with clear run instructions instead — this phase is explicitly
nice-to-have, not required.

---

## Phase 12 — Research Analysis & Report

**Objective**: synthesize all results into the final technical report.

**Tasks**: write Related Work (citing prior art per `prd.md` §11),
Methodology, Results (Pareto frontiers, tables), Threat Model (from
`architecture.md` §10), Limitations, Conclusion.

**Deliverables**: `report.md` (or PDF).

**Tests**: every number in the report cross-checked against a specific
results file/config.

**Exit criteria**: report complete with all required sections; no
unsupported claims present.

**Risks/blockers**: temptation to overstate novelty or security
guarantees.

**Fallback strategy**: none needed — this is a discipline requirement,
not a technical risk; re-read `instructions.md` prohibited-shortcuts
section before finalizing.

---

## Phase 13 — GitHub / Resume / Paper Preparation

**Objective**: finalize the repository and supporting materials for
portfolio/application use.

**Tasks**: clean repository structure, write a strong top-level README
summarizing the research question, methodology, and headline results
(with honest scoping language, not overclaimed novelty); prepare a short
resume bullet/summary; optionally prepare a preprint-style writeup of the
Pareto-frontier methodology.

**Deliverables**: polished public repository, README, optional preprint
draft.

**Tests**: none (documentation phase); manual review against
`instructions.md` prohibited-shortcuts and scope-control rules.

**Exit criteria**: repository is self-explanatory to an external
reviewer without requiring this internal documentation set.

**Risks/blockers**: none beyond time management.

**Fallback strategy**: if time runs short, prioritize a clear README and
complete Phase 8/9 results over Phase 11 (Docker) or preprint polish.
