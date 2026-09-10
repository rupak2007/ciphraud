# Instructions for the Implementing Coding Agent

## Mission

Implement the ambitious version of this project as scoped in `prd.md` and
`architecture.md`: a leakage-safe ML pipeline on IEEE-CIS, FHE-compiled
Logistic Regression / XGBoost / quantized MLP via Concrete-ML, a real
client/server encrypted-inference demo, and a systematic feature-count ×
quantization-bit-width benchmark grid reported as Pareto frontiers. Do not
reduce scope, difficulty, or technical depth because the developer is an
undergraduate. Attempt the hard version. Where something turns out to be
genuinely infeasible, characterize *why*, document it honestly, and propose
a scoped-down fallback — do not silently substitute an easier project.

## Engineering Principles

- Correctness before speed. A slow, correct benchmark is useful; a fast,
  wrong one is worse than nothing.
- Every claim in documentation or the final report must trace to a
  reproducible artifact (a config file, a test, a logged result).
- Prefer explicit, boring code over clever abstractions. This is a
  benchmarking/research codebase; readability and reproducibility matter
  more than elegance.
- Keep the four pipeline stages (training / FHE compilation / runtime
  inference / benchmarking) structurally separate, per `architecture.md`.
  Do not let benchmarking code depend on runtime client/server code in a
  way that couples their correctness.

## Coding Standards

- Python, typed where reasonable (type hints on public functions).
- Config-driven: every experiment is parameterized by a versioned config
  file, never by editing constants in code.
- No hardcoded paths; use a central config or environment variables.
- Structured logging (not print statements) for anything that runs as
  part of a benchmark or pipeline stage.
- Deterministic seeding everywhere a random process occurs (data splits,
  model initialization, calibration sampling).

## ML Rules

- Time-based (expanding-window) split based on `TransactionDT` is
  mandatory. Standard random k-fold cross-validation is prohibited for
  this dataset and must never appear in the codebase.
- Every engineered feature must pass a leakage check: it may only use
  information available strictly before the transaction's timestamp
  within whatever split (train/val/test) it is computed for. Document
  this check per feature group.
- PR-AUC is the primary reported metric. Accuracy alone must never be
  reported as the headline result given the ~3.5% fraud rate.
- Feature importance ranking must come from a real, documented method
  (e.g., trained-model feature importances or permutation importance),
  not arbitrary selection, and must be the basis for feature-count tiers.
- Report error analysis (false positive / false negative characteristics)
  for at least the strongest plaintext model.

## FHE Rules

- Concrete-ML is the default and expected framework. Do not introduce
  OpenFHE/TenSEAL/SEAL unless you can document a specific Concrete-ML
  limitation that blocks a required experiment from `prd.md`.
- Every compiled FHE model must be validated against its plaintext
  counterpart on a held-out correctness-check sample before it is used
  in any benchmark. Log the comparison, not just a pass/fail.
- Respect and document Concrete-ML's real constraints: integer-only
  arithmetic, the accumulator bit-width ceiling for built-in neural
  networks (which cannot be precisely pre-set and must be tuned per
  configuration), and the cost of table lookups (TLUs) for nonlinear/
  rescaling operations. Log TLU count and compile time per configuration.
- Calibration datasets used for quantization must be disjoint from the
  final test set used to report predictive performance.
- If a given (feature tier × bit-width × model type) configuration cannot
  be compiled or run in feasible time/memory, do not silently drop it —
  log it as a failed/infeasible configuration with the specific error or
  resource limit hit. This is itself a valid, reportable finding.

## Security / Threat-Model Rules

- The threat model in `architecture.md` §10 is authoritative. Every piece
  of documentation, code comment, and the final report must be consistent
  with it. Do not introduce new implicit security claims anywhere.
- Never write or imply: "fully private," "fully secure," "HIPAA
  compliant," "GDPR compliant," "PCI compliant," or any other blanket
  compliance/security claim.
- Model confidentiality is explicitly out of scope. Do not imply the
  server-held model is hidden from anyone.
- Any newly discovered leakage channel (metadata, timing, ciphertext size
  correlation, etc.) during implementation must be added to the threat
  model document, not left undocumented.

## Data-Leakage Rules

- No feature, split, or preprocessing step may use information from the
  future relative to the transaction it is applied to.
- No feature, split, or preprocessing step may use test-set statistics
  (e.g., global means/frequencies computed over the full dataset before
  splitting).
- Any leakage found during implementation must be fixed and the fix
  documented, not silently patched without a record.

## Benchmarking Rules

- Minimum 5 repeated trials per (model × feature tier × bit-width)
  configuration; report mean and standard deviation, never a single-shot
  number, for any latency/memory figure.
- Compilation time must be measured and reported separately from
  per-request inference latency.
- Every benchmark run must be tied to a specific config file checked
  into the repository; untraceable numbers must not appear in the report.
- Where feasible, sanity-check results against the published prior-art
  numbers cited in `prd.md` §11 (e.g., the ~6ms XGBoost / ~296ms neural
  network encrypted-inference figures from the cited thesis). Report
  agreement or divergence honestly, with a hypothesis for any divergence
  (hardware, library version, dataset difference).

## Reproducibility Rules

- Pin and record exact versions of Concrete-ML, Concrete-Python, XGBoost,
  scikit-learn, and any other library materially affecting FHE
  performance or ML results.
- Fix and record random seeds for every stochastic process.
- Every reported number in the final report must be regenerable by
  running a specific script against a specific config, both named in the
  report.

## Testing Rules

- Unit tests required for: time-based split correctness, leakage audit
  logic, quantization-parameter consistency between client and compiled
  model, and encrypted-vs-plaintext correctness per model type.
- Integration test required for at least one full client → server →
  client round trip per model type.
- A benchmark smoke test (small grid, minimal trials) must pass before
  any full benchmark run is executed or reported.

## Git Rules

- Commit incrementally per logical unit of work (one pipeline stage, one
  bug fix, one experiment addition per commit where reasonable).
- Commit messages describe what changed and why, not just "update."
- Large data files and generated artifacts (raw datasets, large model
  binaries) must not be committed directly; use `.gitignore` and document
  how to regenerate them.

## Documentation Rules

- Any deviation from `prd.md`, `architecture.md`, or `plan.md` during
  implementation must be reflected back into those documents, not left
  as an undocumented discrepancy.
- The final report must include an explicit "Related Work" section citing
  the prior art named in `prd.md` §11, and an explicit "Limitations"
  section.

## Scope-Control Rules

- Do not implement encrypted-data training, the AML dataset extension, or
  Docker/production deployment until all "Must-have" and "Should-have"
  items in `prd.md`'s priority table are complete.
- Do not add a technology, library, or component that is not required by
  a specific requirement in `prd.md` or `architecture.md`.

## Prohibited Shortcuts

- Never fabricate, estimate, or "reasonably guess" a benchmark number.
  Every number must come from an actual measured run.
- Never claim a model "should work under FHE" without actually compiling
  and running it.
- Never claim a security/compliance property not explicitly demonstrated.
- Never hide a failed experiment or an infeasible configuration — log and
  report it.
- Never substitute a simpler dataset, simpler split strategy, or simpler
  model than what `prd.md` specifies without explicit sign-off and a
  documented reason.

## Handling Uncertainty

When a technical question cannot be resolved from documentation alone
(e.g., exact Concrete-ML accumulator behavior for a specific model
configuration), investigate empirically (small controlled test), document
the finding, and cite the specific behavior observed rather than assuming
prior knowledge is correct.

## Handling Library Limitations

If Concrete-ML cannot support a required configuration (e.g., a feature
count/bit-width combination exceeds the accumulator limit), do not force
it or silently shrink scope. Document the exact limitation, the
error/behavior observed, and either (a) adjust the configuration grid
with a documented rationale, or (b) flag it as a known, reported
limitation of the current FHE tooling — which is itself a valid finding
for the final report.

## FHE Correctness Validation

For every compiled model, run plaintext inference and FHE (or FHE-
simulated, where full FHE execution is too slow for iteration) inference
on the same held-out sample and record the agreement rate and any
divergence. This check must exist before the model is used in any
benchmark run.

## ML Validation

Cross-check that plaintext model metrics (PR-AUC, etc.) are stable across
repeated training runs with different seeds within a reasonable variance
band before treating a single run's numbers as final.

## Definition of Done

A phase (per `plan.md`) is done only when:
- Its exit criteria (as defined in `plan.md`) are met.
- All required tests for that phase pass.
- Any deviation from the PRD/architecture for that phase is documented.
- Any failed or infeasible experiment within that phase is logged, not
  hidden.
- Documentation (`prd.md`/`architecture.md`/`plan.md`) is updated if the
  phase's actual implementation diverged from what was originally
  specified.
