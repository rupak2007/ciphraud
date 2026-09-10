# PRD — Latency-Bounded Privacy-Preserving Fraud Detection Under FHE

## 1. Project Overview

This project builds and rigorously benchmarks a privacy-preserving fraud detection
system in which a client encrypts transaction features using Fully Homomorphic
Encryption (FHE), an untrusted inference server evaluates a trained ML model
directly on ciphertext, and the client decrypts the returned encrypted score.
The server at no point observes plaintext transaction features or the plaintext
fraud decision.

The project has two integrated components that must both be taken seriously:

1. **A working system**: a leakage-safe ML pipeline on IEEE-CIS Fraud Detection,
   trained models (Logistic Regression, XGBoost, quantized MLP) compiled to FHE
   via Concrete-ML, and a real client/server implementation of the encrypt →
   infer → decrypt flow.
2. **A research-style experimental study**, answering:

   > **How do feature count, quantization bit-width, and ML model complexity
   > affect the predictive performance and computational cost of
   > privacy-preserving fraud detection under FHE?**

This is explicitly framed as a **reproduce-and-extend** contribution, not a
claim of first-of-its-kind novelty. Prior published work (notably an encrypted
XGBoost/neural-network fraud detection thesis on the ULB/Vesta datasets, and a
peer-reviewed paper doing FHE-encrypted Logistic Regression/XGBoost fraud
detection on the same datasets) has already established that encrypted tree
and neural inference on fraud data is feasible with acceptable accuracy loss.
This project's differentiation is: (a) applying the same rigor to IEEE-CIS, a
harder dataset with severe missingness, high cardinality, and well-known
temporal-leakage traps that the prior work's datasets don't have, and (b)
running a **systematic Pareto-frontier sweep** across feature count and
quantization bit-width — reported as curves, not single-point estimates —
which existing published work does not do.

## 2. Exact Problem Statement

A party that wants ML-based fraud scoring (a bank, PSP, or fraud-detection
vendor) must send transaction features to an inference provider it does not
fully trust with plaintext data (e.g., a third-party model vendor, a shared
cloud inference service). The provider should be able to return an accurate
fraud score without ever observing the plaintext transaction features or the
plaintext decision.

This is a **narrow, single-client / single-server, inference-outsourcing**
problem. It is explicitly **not** a claim to solve cross-institution AML data
collaboration (a multi-party problem better suited to MPC) or federated
model training (a different threat model, solved by federated learning with
its own known leakage issues). The PRD and all other project documents must
maintain this scope discipline.

## 3. Target Users

- **Primary (real-world, illustrative)**: a fraud-detection vendor or bank
  wanting to outsource inference to a less-trusted compute provider without
  exposing transaction features.
- **Primary (actual, for this project)**: ML engineering / applied ML
  internship reviewers, and graduate admissions reviewers (target: U of T
  MScAC AI), evaluating the project as a portfolio artifact.
- **Secondary**: future contributors/readers of the open-source repository
  who want a reference benchmark of FHE cost vs. ML model complexity for
  tabular fraud classification.

## 4. Project Goals

- Build a correct, leakage-safe plaintext ML pipeline on IEEE-CIS.
- Convert Logistic Regression, XGBoost, and a quantized MLP to FHE-compatible
  form via Concrete-ML and validate correctness of encrypted inference against
  plaintext inference.
- Build a real (not simulated) client/server split implementing the full
  encrypt → send → infer → return → decrypt flow.
- Run a systematic experimental sweep across feature count and quantization
  bit-width, producing accuracy-vs-latency (and memory, and where meaningful
  throughput) Pareto frontiers, with repeated trials and variance reporting.
- Produce a precise, honest threat-model writeup, an honest limitations
  section, and a technical report suitable for a portfolio and (optionally)
  a short paper/preprint.

## 5. Non-Goals

- **Not** building encrypted training (only encrypted inference is in scope).
- **Not** building a multi-institution MPC-based AML system.
- **Not** building a production-grade deployment (Kubernetes, autoscaling,
  managed cloud infra, monitoring stack).
- **Not** claiming novelty of "FHE for fraud detection" as a concept — this
  is prior art and must be cited as such.
- **Not** claiming any regulatory compliance (GDPR, HIPAA, PCI-DSS, etc.).
- **Not** protecting model confidentiality, request-timing side channels, or
  request-volume/ciphertext-size metadata leakage — these are explicitly
  out of scope and must be documented as such, not silently ignored.
- The AML dataset extension is a **stretch goal only**, not a core deliverable.

## 6. Functional Requirements

- FR1: Ingest and validate the IEEE-CIS Fraud Detection dataset
  (transaction + identity tables).
- FR2: Implement a **time-based (expanding-window) train/validation/test
  split** based on `TransactionDT`; standard random k-fold is explicitly
  disallowed as a leakage risk.
- FR3: Implement missing-data handling, categorical encoding, and feature
  engineering with an explicit, documented leakage audit (no
  target-dependent or future-dependent feature construction).
- FR4: Train and evaluate Logistic Regression and XGBoost as plaintext
  baselines with class-imbalance handling appropriate to a ~3.5% fraud rate.
- FR5: Train and evaluate a quantized MLP as a plaintext baseline.
- FR6: Perform feature selection/importance ranking, and define at least
  three feature-count tiers (e.g., top-20 / top-50 / top-100) for the FHE
  sweep.
- FR7: Compile each model (each feature tier × each bit-width configuration)
  to FHE via Concrete-ML, and validate that encrypted predictions match
  plaintext predictions on a held-out sample within documented tolerance.
- FR8: Implement a client/server architecture where the client encrypts
  a feature vector, sends ciphertext to the server, the server performs
  encrypted inference, and the client decrypts the result.
- FR9: Implement a benchmark harness that measures latency, memory, and
  (where meaningful) ciphertext size / throughput, across the full
  feature-count × bit-width × model-type grid, with multiple repeated
  trials per configuration.
- FR10: Produce Pareto-frontier plots (accuracy vs. latency at minimum)
  from the benchmark results.
- FR11: Produce a written threat-model document precisely scoping what is
  and is not protected.
- FR12: Produce a final technical report synthesizing ML results, FHE
  results, and the Pareto-frontier analysis, with explicit citation of and
  comparison against prior published work.

## 7. Non-Functional Requirements

- NFR1: All experiments must be reproducible from a fixed configuration
  (seeds, library versions, hyperparameters) with no undocumented manual
  steps.
- NFR2: All library versions (Concrete-ML, Concrete-Python, XGBoost, etc.)
  must be pinned and recorded, since FHE performance and even correctness
  can shift across versions.
- NFR3: The system must run end-to-end on the hardware actually available
  (document CPU/RAM ahead of time); model/config sizes must be scoped to
  what can complete compilation and benchmarking in feasible wall-clock time.
- NFR4: Code must be organized so the plaintext ML pipeline, the FHE
  compilation step, and the runtime inference step are independently
  testable and independently benchmarkable.
- NFR5: No component may claim a security or compliance property that has
  not been explicitly demonstrated and documented.

## 8. ML Requirements

- Time-based split is mandatory, not optional.
- PR-AUC is the primary metric; ROC-AUC, precision, recall, F1, F2, and a
  cost-sensitive/threshold analysis are required secondary metrics.
  Accuracy alone is explicitly disallowed as a headline metric given the
  ~3.5% fraud rate.
- Feature importance ranking must be produced and used to justify feature-
  count tiers used in the FHE sweep (not arbitrary/convenience-based
  selection).
- Error analysis (false positive / false negative characterization) is
  required for at least the best-performing plaintext model.

## 9. FHE Requirements

- Concrete-ML is the primary framework. A switch to or addition of
  OpenFHE/TenSEAL/SEAL is permitted only if a specific, documented
  technical limitation of Concrete-ML blocks a required experiment —
  not for its own sake.
- Every FHE-compiled model must have its encrypted output cross-validated
  against its plaintext output on a held-out sample; discrepancies beyond
  documented quantization-induced tolerance must be investigated and
  reported, not silently accepted.
- The project must explicitly document Concrete-ML's known precision
  constraints (integer-only arithmetic, accumulator bit-width limits for
  built-in neural networks, and table-lookup (TLU) cost for nonlinear/
  rescaling operations) and how these constraints shaped the feature-count
  and bit-width tiers chosen — not treat them as a surprise discovered
  mid-implementation.
- Calibration datasets used for quantization must be documented and
  distinct from final test data used for benchmarking predictive
  performance.

## 10. Security / Threat-Model Requirements

- The threat model document (see architecture.md) must explicitly state:
  who holds the secret key (client), what the client knows (plaintext
  features and decrypted result), what the server knows (ciphertext,
  model parameters, request metadata), what is encrypted (transaction
  feature vector, inference result), what remains plaintext on the server
  side (the model itself, unless a stretch goal states otherwise), and
  what is explicitly out of scope (model confidentiality from the client,
  request timing/volume side channels, network-level metadata).
- No claim of "fully private," "HIPAA compliant," "GDPR compliant," or
  equivalent blanket security claims is permitted anywhere in the
  documentation, code comments, or report.
- A written comparison against at least TEEs and MPC as alternative
  architectures for the same or adjacent threat models is required,
  including an honest statement of where FHE is *not* the better choice
  (see architecture.md).

## 11. Research Requirements

- The project must explicitly cite and, where feasible, sanity-check
  against at least the following prior work: the encrypted XGBoost/
  feedforward-NN fraud detection thesis (ULB/Vesta datasets, reporting
  encrypted XGBoost inference latency as low as 6ms vs. 296ms for a neural
  network), and the peer-reviewed Logistic Regression/XGBoost FHE fraud
  detection paper on the same datasets.
- Findings that contradict or diverge from prior published latency/accuracy
  numbers must be reported honestly (e.g., due to hardware, dataset, or
  library-version differences), not adjusted to match expectations.
- All experimental claims in the final report must be traceable to a
  specific benchmark run and configuration file.

## 12. Benchmarking Requirements

- Every reported latency/memory number must be the mean and standard
  deviation over a minimum of N repeated trials per configuration (N to be
  fixed and documented, minimum 5).
- The benchmark grid must at minimum cover: 3 feature-count tiers × at
  least 2 quantization bit-width settings × 3 model types (LR, XGBoost,
  quantized MLP) = minimum 18 configurations, each separately benchmarked
  for both plaintext and FHE inference.
- Compilation time must be measured and reported separately from
  per-request inference latency.

## 13. Success Criteria

- A correct, leakage-audited plaintext pipeline with documented time-based
  split and PR-AUC-anchored evaluation.
- At least one working encrypted inference path per model type (LR,
  XGBoost, quantized MLP), each validated for correctness against its
  plaintext counterpart.
- A complete benchmark grid (per §12) executed with repeated trials and
  reported as Pareto-frontier plots.
- A working client/server demonstration of the full encrypt/infer/decrypt
  flow.
- A written threat-model section and a written comparison against TEE/MPC
  alternatives.
- A final technical report that explicitly positions the work relative to
  cited prior art.

## 14. Acceptance Criteria

- No component of the system claims a security property it has not
  demonstrated.
- No benchmark number appears in the report without a corresponding
  reproducible run and config file.
- The plaintext pipeline's time-based split is verifiable in code (not
  just described).
- Every FHE-compiled model has at least one recorded correctness check
  against its plaintext counterpart.
- The final report contains an explicit "Limitations" section and an
  explicit "Related Work" section with the citations listed in §11.

## 15. Deliverables

- Public (or portfolio-ready) GitHub repository with: data pipeline code,
  model training code, FHE compilation/inference code, client/server
  implementation, benchmark harness, experiment configs, and results
  (plots + raw benchmark data).
- `prd.md`, `architecture.md`, `instructions.md`, `plan.md` (this
  documentation set).
- Final technical report (Markdown or PDF) with Related Work, Methodology,
  Results (Pareto frontiers), Threat Model, Limitations, and Conclusion
  sections.

## 16. Constraints

- Timeline: 3–5 months, part-time alongside coursework.
- Hardware: whatever CPU/RAM is actually available to the developer;
  model/config sizes and sweep breadth must be scoped to this constraint
  and documented explicitly if reduced from the ideal grid.
- Solo project; no team engineering capacity assumed.

## 17. Stretch Goals

- AML dataset extension (clearly separated arm, only after the core
  IEEE-CIS study is complete).
- Encrypted-data training experiments for the Logistic Regression model
  (Concrete-ML supports this for some model types) as a small additional
  arm, not a core requirement.
- A short preprint/paper writeup of the Pareto-frontier methodology and
  results.
- Containerized (Docker) deployment of the client/server demo, as a final
  polish item only after all core deliverables are complete.

## Priority Classification

| Requirement | Priority |
|---|---|
| Leakage-safe time-based split | Must-have |
| PR-AUC-anchored evaluation, imbalance handling | Must-have |
| LR + XGBoost plaintext & FHE | Must-have |
| Feature-count × bit-width benchmark grid | Must-have |
| Repeated-trial variance reporting | Must-have |
| Threat-model document | Must-have |
| Client/server encrypt/infer/decrypt demo | Should-have |
| Quantized MLP plaintext & FHE | Should-have |
| TEE/MPC written comparison | Should-have |
| Final technical report with Related Work | Should-have |
| Docker packaging | Nice-to-have |
| AML dataset extension | Stretch/research extension |
| Encrypted-data training arm | Stretch/research extension |
| Preprint/paper writeup | Stretch/research extension |
