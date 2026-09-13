# CLAUDE.md — Ciphraud

## 1. Project Identity

Project: **Ciphraud** — Latency-Bounded Privacy-Preserving Fraud Detection Under FHE

The project builds a rigorous, research-grade fraud detection system in which a client encrypts transaction features with Fully Homomorphic Encryption (FHE), an inference server evaluates the ML model directly on ciphertext, and the client decrypts the returned encrypted result.

The project's central research question is:

> How do feature count, quantization bit-width, and ML model complexity affect the predictive performance and computational cost of privacy-preserving fraud detection under FHE?

This is explicitly a **reproduce-and-extend** research project, not a claim of first-of-its-kind novelty.

---

## 2. Source of Truth

Before making implementation decisions, read and follow these files:

- `docs/prd.md` — product/project requirements and acceptance criteria
- `docs/architecture.md` — authoritative system architecture and threat model
- `docs/plan.md` — phased implementation roadmap and exit criteria
- `docs/instructions.md` — detailed engineering, ML, FHE, security, benchmarking, testing, Git, and documentation rules

These documents are authoritative together. Do not silently contradict them.

When implementation diverges from the documented PRD, architecture, or plan, update the documentation to record the deviation and its reason.

Do not invent missing requirements.

---

## 3. Mission

Implement the ambitious version of the project as specified in the documentation:

- Leakage-safe IEEE-CIS Fraud Detection ML pipeline
- Logistic Regression, XGBoost, and quantized MLP
- FHE compilation primarily through Concrete-ML
- Real client/server encrypted-inference flow
- Systematic feature-count × quantization-bit-width × model benchmark grid
- Repeated-trial measurements and variance reporting
- Pareto-frontier analysis
- Precise threat-model and limitations documentation
- Reproducible final technical report

Do not reduce the technical depth merely because the developer is an undergraduate.

If something is genuinely infeasible:
1. characterize exactly why,
2. measure/document the limitation,
3. propose a scoped fallback,
4. never silently substitute an easier project.

---

## 4. Architecture

The project has four structurally separate pipelines:

```text
OFFLINE TRAINING
        ↓
FHE COMPILATION
        ↓
RUNTIME ENCRYPTED INFERENCE
        ↓
BENCHMARKING
```

These stages must remain independently testable and independently runnable.

### Offline training

Input:
- IEEE-CIS `train_transaction.csv`
- IEEE-CIS `train_identity.csv`

Flow:

```text
raw data
  ↓
schema validation
  ↓
time-based split
  ↓
leakage-audited preprocessing / feature engineering
  ↓
class-imbalance handling
  ↓
LR / XGBoost / MLP training
  ↓
plaintext evaluation
  ↓
feature-importance ranking
  ↓
versioned feature tiers
```

### FHE compilation

Input:
- trained plaintext model
- one feature tier
- one quantization bit-width configuration
- calibration dataset disjoint from final test set

Flow:

```text
trained model
  ↓
quantization
  ↓
Concrete-ML FHE compilation
  ↓
compiled circuit
  ↓
held-out plaintext-vs-FHE correctness validation
```

One compiled artifact must be associated with each
`model type × feature tier × bit-width` configuration.

### Runtime encrypted inference

```text
CLIENT
  raw features
      ↓
  quantize
      ↓
  encrypt
      │
      │ ciphertext
      ▼
SERVER
  compiled model
      ↓
  encrypted inference
      │
      │ encrypted result
      ▼
CLIENT
  decrypt
      ↓
  fraud score / decision
```

The client holds the secret key. The server never receives the secret key and never decrypts the transaction or result.

### Benchmarking

A config-driven benchmark runner evaluates the experiment grid and records:
- mean latency
- standard deviation
- peak memory
- ciphertext size where meaningful
- compilation time separately from inference latency
- configuration provenance

Minimum repeated trials per configuration: **5**.

---

## 5. Repository Structure

Target structure:

```text
/
├── CLAUDE.md
├── .gitignore
├── docs/
│   ├── prd.md
│   ├── architecture.md
│   ├── instructions.md
│   └── plan.md
├── data/                  # raw/processed data; large data must not be committed
├── src/
│   ├── data/
│   ├── features/
│   ├── train/
│   ├── fhe/
│   │   ├── compile/
│   │   └── validate/
│   ├── client/
│   ├── server/
│   ├── benchmark/
│   └── analysis/
├── configs/
├── results/               # generated experiment outputs
├── tests/
└── report.md
```

Component responsibilities:

- `src/data/` — ingestion, validation, time-based split, leakage audit
- `src/features/` — feature engineering, importance ranking, feature tiers
- `src/train/` — plaintext training and evaluation
- `src/fhe/compile/` — quantization and Concrete-ML compilation
- `src/fhe/validate/` — encrypted-vs-plaintext correctness checks
- `src/client/` — key generation, quantization, encryption, decryption
- `src/server/` — FastAPI encrypted inference service
- `src/benchmark/` — repeated-trial grid benchmarks
- `src/analysis/` — Pareto frontiers and report tables

---

## 6. ML Rules

### Dataset split

Use a **time-based expanding-window split based on `TransactionDT`**.

Random k-fold cross-validation is prohibited.

The split artifact must be saved and versioned rather than regenerated ad hoc.

### Leakage prevention

Temporal leakage is a critical correctness issue.

Every engineered feature must use only information available strictly before the transaction timestamp within the split where it is computed.

Do not use:
- future information
- test-set statistics
- global statistics computed before splitting
- target-dependent leakage
- future-window information

Frequency/categorical encodings must be scoped to information available in the appropriate past window.

Every feature group must have an explicit leakage audit.

### Metrics

Primary metric:

- **PR-AUC**

Secondary metrics include:
- ROC-AUC
- precision
- recall
- F1
- F2
- threshold/cost-sensitive analysis
- confusion matrices

Do not use accuracy as the headline metric because of the severe fraud-class imbalance (~3.5%).

Perform false-positive / false-negative error analysis for at least the strongest plaintext model.

### Model baselines

Required plaintext models:
- Logistic Regression
- XGBoost
- quantized MLP

Feature importance must come from a real documented method, such as trained-model importance or permutation importance.

Feature-count tiers must be based on this ranking, not arbitrary convenience.

---

## 7. FHE Rules

### Framework

Concrete-ML is the primary/default FHE framework.

Do not introduce OpenFHE, TenSEAL, SEAL, or another FHE framework unless a specific documented Concrete-ML limitation blocks a required experiment.

### Correctness

Every compiled FHE model must be validated against its plaintext counterpart on a held-out correctness-check sample before it is benchmarked.

Record:
- plaintext output
- FHE or FHE-simulated output where appropriate
- agreement rate
- divergences
- documented tolerance

Never claim a model works under FHE without actually compiling and running it.

### Concrete-ML constraints

Respect and document real framework constraints, including:
- integer-only arithmetic
- accumulator bit-width ceilings for built-in neural networks
- empirical tuning required for neural-network configurations
- table lookup (TLU) cost for nonlinear/rescaling operations
- XGBoost `n_bits` behavior and calibration-driven intermediate bit-width computation

For MLP experiments, log TLU count and compile time.

### Calibration

Calibration data used for quantization must be disjoint from the final test set used for predictive-performance reporting.

### Failed or infeasible configurations

Never silently drop a configuration.

For every failed/infeasible
`model × feature tier × bit-width` configuration, record:
- configuration identity
- exact error/limitation
- resource constraint if relevant
- whether compilation or inference failed
- any documented fallback decision

An infeasible configuration is a valid research finding.

---

## 8. Security and Threat Model

The authoritative threat model is defined in `docs/architecture.md`.

### Trust model

Client:
- trusted data owner
- holds plaintext transaction features
- holds the secret key
- decrypts the result

Server:
- untrusted with plaintext transaction data
- holds the compiled model and public/evaluation material
- performs encrypted inference
- must never receive the secret key
- does not decrypt anything

### What the server can know

The server may know:
- ciphertext
- plaintext model parameters
- request timing
- request size
- request frequency
- ciphertext size

### What stays hidden from the server

The server must not receive:
- plaintext transaction feature values
- plaintext inference result

### Out of scope

Do not silently expand the security claim to cover:
- model confidentiality
- request timing/volume side channels
- ciphertext-size metadata leakage
- malicious-server result tampering / verifiable computation
- side-channel attacks on the client's local environment
- network metadata analysis beyond what is explicitly documented
- adversarial/poisoning attacks on training data
- multi-party/cross-institution collaboration
- regulatory compliance

The project's single security claim is intentionally narrow:

An honest, hardware-untrusted inference server cannot recover the plaintext transaction features or plaintext fraud decision from the protected inference data it receives.

Never write or imply:
- "fully private"
- "fully secure"
- "HIPAA compliant"
- "GDPR compliant"
- "PCI compliant"
- other blanket privacy/security/compliance claims

### Alternatives

The final project must discuss TEE and MPC alternatives honestly.

FHE is not automatically the practical choice in every threat model:
- TEEs may provide much lower latency when hardware can be trusted, at the cost of hardware/enclave trust and side-channel considerations.
- MPC is better suited to genuine multi-institution collaborative computation.
- Federated learning addresses training-data locality, not the same inference-time confidentiality problem.
- Differential privacy and tokenization solve different problems.

---

## 9. Benchmarking and Experiments

Every experiment is defined by a versioned config containing relevant:
- model type
- feature tier
- bit-width
- calibration-set reference
- random seed
- library versions
- other experiment parameters

Every reported result must map to a specific config.

Minimum benchmark grid:
- at least 3 feature-count tiers
- at least 2 quantization bit-width settings
- 3 model types
- minimum 18 configurations
- plaintext and FHE inference benchmarking where applicable
- at least 5 trials per configuration

Report:
- mean
- standard deviation
- compile time separately
- per-request inference latency
- peak memory
- ciphertext size where meaningful
- throughput where meaningful

Generate accuracy-vs-latency Pareto frontiers at minimum.

Benchmark results must be reproducible and traceable to:
- config hash
- code revision / Git commit
- timestamp
- raw trial data

Never fabricate, estimate, or guess benchmark values.

---

## 10. Reproducibility

Pin and record exact versions of materially relevant libraries, especially:
- Concrete-ML
- Concrete-Python
- XGBoost
- scikit-learn
- pandas
- other FHE/ML dependencies

Record hardware used for FHE benchmarking:
- CPU
- RAM
- relevant environment details

Fix and record random seeds for stochastic processes.

Every number in the final report must be regenerable from a named script and named config.

---

## 11. Testing Requirements

Required unit tests include:
- time-based split correctness
- no-future/no-test leakage
- feature leakage audit logic
- quantization-parameter consistency between client and compiled model
- encrypted-vs-plaintext correctness for each model type
- metric computation
- deterministic feature-tier membership where applicable

Required integration testing:
- at least one full client → server → client round trip per model type

Benchmarking:
- maintain a small smoke-test grid
- smoke test must pass before any full benchmark run is executed or reported

Plaintext ML:
- cross-check stability across multiple random seeds
- document meaningful variance before treating a single run as final

---

## 12. Phased Development Order

Follow `docs/plan.md`.

### Phase 0 — Environment & Repository
- initialize repository structure
- pin Python and dependencies
- configure experiment loading
- configure structured logging
- configure tests
- record hardware
- verify environment

### Phase 1 — Dataset & EDA
- load IEEE-CIS transaction and identity tables
- inspect schema
- missingness
- class imbalance
- `TransactionDT`
- high-cardinality/anonymized groups

### Phase 2 — Leakage-Safe ML Pipeline
- expanding-window split
- missing-data handling
- categorical encoding
- feature engineering
- leakage audit
- versioned split artifact

### Phase 3 — Baseline Models
- Logistic Regression
- XGBoost
- class-imbalance handling
- PR-AUC-centered evaluation
- threshold analysis
- confusion matrices
- seed stability

### Phase 4 — Feature Engineering & Selection
- feature importance
- feature-count tiers
- plaintext accuracy-vs-feature-count analysis

### Phase 5 — FHE Proof of Concept
- start with Logistic Regression
- smallest feature tier
- quantize
- compile
- encrypt → infer → decrypt
- validate against plaintext

### Phase 6 — FHE Fraud Inference (XGBoost)
- compile per feature tier
- validate every configuration
- record compile time

### Phase 7 — Benchmark Infrastructure
- config-driven runner
- repeated trials
- latency/memory/ciphertext-size measurement
- configuration hashing
- smoke tests

### Phase 8 — Core Research Experiments
- full LR/XGBoost feature-tier × bit-width grid
- repeated measurements
- Pareto frontiers
- prior-art sanity checks
- honest divergence analysis

### Phase 9 — Quantized MLP Extension
- train/quantize MLP
- compile with Concrete-ML
- respect accumulator constraints
- log TLU count and compile time
- benchmark feasible configurations
- explicitly report infeasible configurations

### Phase 10 — Client/Server Implementation
- client key generation
- client quantize/encrypt/decrypt
- FastAPI server
- encrypted inference
- end-to-end round trips
- measure serialization/network overhead separately

### Phase 11 — Docker/API Polish
Only after core phases are complete.

### Phase 12 — Research Analysis & Report
- Related Work
- Methodology
- Results
- Pareto frontiers
- Threat Model
- Limitations
- Conclusion
- cross-check every reported number

### Phase 13 — GitHub / Resume / Paper Preparation
- repository cleanup
- strong README
- portfolio/resume summary
- optional preprint

Do not prioritize stretch/polish work over incomplete core experiments.

---

## 13. Scope Control

Do not implement before the core requirements are complete:
- encrypted-data training
- AML dataset extension
- Docker/production deployment

The core scope is encrypted inference, not encrypted training.

Do not add a technology, library, service, infrastructure component, or abstraction unless it is required by the documented requirements or justified by a specific documented limitation.

No cloud production infrastructure is required. Local/single-machine deployment is the intended project scope, with cloud compute only as a documented fallback if local hardware cannot run required FHE experiments.

---

## 14. Git Rules

Commit incrementally by logical unit of work:
- one pipeline stage
- one bug fix
- one coherent experiment addition

Commit messages should explain what changed and why.

Do not commit:
- raw datasets
- large generated model binaries
- large generated outputs unless explicitly required
- secrets
- `.env` files
- virtual environments
- machine-specific temporary artifacts

Keep `.gitignore` updated.

---

## 15. Coding Standards

Prefer:
- Python
- type hints on public functions where reasonable
- explicit, readable code
- config-driven behavior
- structured logging instead of `print` for pipeline/benchmark execution
- deterministic seeding
- central configuration/environment handling
- small testable units

Avoid clever abstractions that reduce reproducibility or readability.

Do not hardcode machine-specific paths.

---

## 16. Working Style for the Developer

The developer is learning the system while Claude Code implements it.

Therefore:

### Before a major implementation task
Briefly explain:
- what is about to be built
- why it exists
- where it fits in the architecture
- what dependencies it has

### During implementation
- implement the focused task
- do not unnecessarily stop for teaching
- do not conceal important architectural, ML, cryptographic, FHE, or engineering decisions
- ask the developer only when an actual architectural decision or unresolved ambiguity requires input

### After a major subsystem
Provide a learning checkpoint containing:
1. What was built
2. Why it exists
3. Where it fits
4. Key concepts
5. Important files/functions
6. Input → processing → output flow
7. Important tradeoffs
8. What to study next

Do not explain every line unless asked.

The goal is for the developer to eventually be able to:
- explain the complete system
- understand the mathematics and ML
- understand the FHE/security model
- debug the implementation
- modify the system
- defend the design and experimental methodology

---

## 17. Handling Uncertainty

When documentation does not determine a technical behavior:

1. Do not guess.
2. Identify the uncertainty.
3. Prefer a small controlled empirical test.
4. Record the observed behavior.
5. Cite/document the specific result.
6. Update project documentation when the finding changes implementation assumptions.

This is especially important for Concrete-ML behavior, compilation limits, accumulator constraints, precision, runtime, and supported configurations.

---

## 18. Research Integrity

Never:
- fabricate results
- adjust measurements to match prior work
- hide failed experiments
- silently drop infeasible configurations
- claim unsupported novelty
- claim unsupported security
- claim a model works under FHE without compiling/running it
- substitute a simpler dataset/model/split without documented approval

The project's value comes from rigorous, reproducible engineering and honest experimental analysis.

---

## 19. Definition of Done

A phase is complete only when:
- its documented exit criteria are met
- required tests pass
- deviations are documented
- failed/infeasible experiments are logged
- relevant documentation is updated
- results are reproducible from configs and code

Do not mark a phase "done" merely because the implementation exists.

---

## 20. Default Operating Rule

**Inspect first. Plan the focused change. Implement. Test. Validate. Document. Then move to the next phase.**

Never rush ahead to later FHE experiments while earlier data, leakage, ML, or correctness foundations are unresolved.
