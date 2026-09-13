# Architecture — Ciphraud (Latency-Bounded Privacy-Preserving Fraud Detection Under FHE)

## 1. System Overview

The system has four distinct pipelines that must remain structurally
separate in code, not just conceptually:

```
OFFLINE TRAINING  →  FHE COMPILATION  →  RUNTIME ENCRYPTED INFERENCE  →  BENCHMARKING
```

Each stage has different inputs, outputs, and correctness criteria, and each
must be independently testable and independently runnable without re-running
the others.

## 2. High-Level Architecture

```
┌────────────────────────┐        ┌──────────────────────────┐
│   OFFLINE (plaintext)  │        │   RUNTIME (client/server) │
│                        │        │                          │
│  Raw IEEE-CIS data     │        │  ┌────────────┐           │
│      │                 │        │  │  CLIENT     │          │
│      ▼                 │        │  │(secret key) │          │
│  Time-based split      │        │  └─────┬───────┘          │
│      │                 │        │        │ plaintext        │
│      ▼                 │        │        ▼ features          │
│  Feature engineering   │        │  Quantize + Encrypt        │
│  + selection tiers     │        │        │ ciphertext         │
│      │                 │        │        ▼                   │
│      ▼                 │        │  ┌────────────┐            │
│  Train LR / XGB / MLP  │───────▶│  │  SERVER     │            │
│      │                 │compiled│  │ (model,     │            │
│      ▼                 │ model  │  │  no key)    │            │
│  Concrete-ML compile   │        │  └─────┬───────┘            │
│  (per feature tier ×   │        │        │ encrypted score    │
│   bit-width config)    │        │        ▼                    │
│                        │        │  Decrypt (client)           │
└────────────────────────┘        │        │                    │
                                   │        ▼                    │
                                   │  Fraud score / decision      │
                                   └──────────────────────────┘
                                              │
                                              ▼
                                   ┌──────────────────────────┐
                                   │       BENCHMARKING        │
                                   │ latency / memory / size   │
                                   │ repeated trials, Pareto   │
                                   │ frontier generation       │
                                   └──────────────────────────┘
```

## 3. ML Training Pipeline

- Input: raw `train_transaction.csv` + `train_identity.csv` (IEEE-CIS).
- Steps: schema validation → missing-data handling → categorical encoding
  → feature engineering → **time-based split** (sort by `TransactionDT`,
  expanding-window train/val/test, never random k-fold) → class-imbalance
  handling → model training (LR, XGBoost, MLP) → evaluation (PR-AUC primary).
- Output: trained plaintext models + feature importance rankings + a fixed
  train/val/test split artifact (saved indices, not re-derived per run).
- **Leakage audit is a mandatory sub-step**: every engineered feature must
  be checked that it uses only information available strictly before the
  transaction's timestamp within the split it's used in.

## 4. Feature Engineering Pipeline

- Consumes the leakage-audited training split only.
- Produces a ranked feature-importance list (e.g., from the plaintext
  XGBoost model) used to define **feature-count tiers**: e.g., top-20,
  top-50, top-100.
- Each tier is a first-class, versioned artifact (not recomputed ad hoc),
  since the FHE compilation and benchmarking pipelines depend on exact
  tier membership being stable and reproducible.

## 5. FHE Compilation Pipeline

- Input: a trained plaintext model + a specific feature tier + a specific
  quantization bit-width configuration + a calibration dataset (a subset
  of the training split, disjoint from the final test set).
- Steps: quantize (Concrete-ML built-in quantization for LR/XGBoost, or
  Brevitas/QAT-based quantization for the MLP) → compile to an FHE circuit
  → validate compiled-circuit output against plaintext output on a
  held-out correctness-check sample.
- Known hard constraints that must be respected, not discovered late:
  - Concrete is integer-only arithmetic with a hard accumulator bit-width
    ceiling; for neural networks this cannot be precisely pre-set and
    must be tuned empirically against the feature-count × bit-width
    combination in use.
  - Nonlinear/rescaling operations are implemented via table lookups
    (TLUs), which are the dominant latency cost for the MLP arm; TLU
    count per configuration must be logged as an experiment variable.
  - XGBoost compilation uses an `n_bits` parameter controlling input
    feature precision and a calibration-set-driven bit-width computation
    per intermediate value; this must be logged per configuration.
- Output: one compiled FHE artifact per (model type × feature tier ×
  bit-width) combination, each independently benchmarkable, plus a
  correctness-check report.

## 6. Runtime Inference Architecture

- **Client**: holds the FHE secret key. Responsible for quantizing raw
  features (using the same quantization parameters as the compiled
  model), encrypting the quantized vector, sending ciphertext to the
  server, receiving the encrypted result, and decrypting it locally.
- **Server**: holds the compiled FHE model (public evaluation
  material only — never the secret key). Receives ciphertext, performs
  encrypted inference, returns encrypted output. The server does not
  decrypt anything at any point.
- Transport: a minimal FastAPI service is sufficient; this is a
  correctness/demo layer, not a production deployment target (see §20).

## 7. Client/Server Architecture

```
Client (secret key holder)          Server (model holder)
  │  raw features                        │
  │  quantize                            │
  │  encrypt ──────ciphertext──────────▶ │
  │                                       │  encrypted inference
  │  ◀─────encrypted result────────────  │
  │  decrypt                             │
  │  fraud score                         │
```

## 8. Data Flow

Plaintext features exist **only on the client**, before encryption and
after decryption of the *result*. The server never receives plaintext
features and never produces a plaintext result. Model parameters exist
in plaintext on the server (model confidentiality is explicitly out of
scope — see §10).

## 9. Trust Boundaries

- Client is trusted with plaintext data and the secret key (it is the
  data owner).
- Server is untrusted with plaintext data but trusted to correctly
  execute the agreed-upon inference circuit (no malicious-server /
  verifiable-computation guarantees are provided — a malicious server
  could return an incorrect encrypted result and the client would not
  detect this; this is explicitly out of scope).
- No third party is modeled.

## 10. Threat Model (authoritative — must match instructions.md and prd.md)

| Question | Answer |
|---|---|
| Who owns the secret key? | The client only. |
| What does the client know? | Plaintext features, plaintext decrypted result. |
| What does the server know? | Ciphertext, the model (plaintext parameters), request metadata (timing, request size, request frequency). |
| What is encrypted? | The transaction feature vector (in transit and during computation), the inference result (until client decryption). |
| What remains plaintext? | The model itself; server-side infrastructure/logs of request metadata. |
| What is explicitly OUT of scope? | Model confidentiality from the client; protection against request timing/volume side channels; protection against a malicious (result-tampering) server; multi-party/cross-institution data collaboration (that is an MPC problem, not this project's problem); any regulatory compliance claim. |
| What attacks are NOT considered? | Side-channel attacks on the client's local environment; network-layer metadata analysis beyond what's noted above; adversarial/poisoning attacks on training data. |

**This project makes exactly one claim**: an honest, hardware-untrusted
inference server cannot recover plaintext transaction features or the
plaintext fraud decision from what it is given. It does not claim general
"privacy" or "security" beyond this narrow, explicitly scoped guarantee.

### Comparison against alternative architectures (required by prd.md §10)

- **TEE (e.g., Intel SGX/TDX, Nitro Enclaves)**: would give near-plaintext
  latency with attestation, at the cost of trusting the hardware vendor
  and enclave implementation against side-channel attacks. For a threat
  model where the operator's *hardware* is untrusted (not just the
  operator's software), FHE is strictly stronger; where hardware is
  trusted, a TEE is the more practical real-world choice and this project
  does not claim otherwise.
- **MPC**: better suited to genuine multi-institution AML collaboration
  (several parties jointly compute without any one party seeing the
  union of data). This project's single-client/single-server architecture
  is not that problem, and MPC is not implemented here.
- **Federated learning**: solves training-data locality, not inference-time
  input confidentiality, and has its own known gradient-inversion leakage
  risk. Not a substitute; not implemented here.
- **Differential privacy / tokenization**: solve different problems
  (statistical disclosure, storage/PCI scope) and are noted but not
  implemented.

## 11. Key Ownership

Secret key: client, generated and held client-side, never transmitted.
Evaluation/public material required for server-side computation: generated
client-side and provided to the server ahead of inference (standard
Concrete-ML client/server deployment pattern).

## 12. Model Ownership

Model parameters are plaintext and held by the server. Model confidentiality
is explicitly not a goal of this project (see §10).

## 13. What Is Encrypted / What The Server Can and Cannot See

- Encrypted: transaction feature vector (client → server), inference
  result (server → client) until client-side decryption.
- Plaintext, server-visible: model parameters, request timing, request
  size/frequency, ciphertext byte size (which may correlate with feature
  count if feature count varies — held constant per deployed model to
  avoid this leaking additional information beyond the fixed public
  feature-tier choice).
- Never visible to the server: plaintext feature values, plaintext
  inference result.

## 14. Metadata / Side-Channel Limitations

Explicitly documented, not fixed: request timing and frequency could leak
information about transaction volume/patterns; this is out of scope and
must be stated plainly in the final report rather than silently ignored.

## 15. Component Responsibilities

| Component | Responsibility |
|---|---|
| `data/` pipeline | Ingestion, validation, time-based split, leakage audit |
| `features/` pipeline | Feature engineering, importance ranking, tier definition |
| `train/` pipeline | Plaintext model training (LR, XGBoost, MLP) and evaluation |
| `fhe/compile/` | Quantization + Concrete-ML compilation per (model × tier × bit-width) |
| `fhe/validate/` | Encrypted-vs-plaintext correctness checks |
| `client/` | Key generation, quantize/encrypt, decrypt |
| `server/` | Load compiled model, serve encrypted inference (FastAPI) |
| `benchmark/` | Repeated-trial latency/memory/size measurement across the full grid |
| `analysis/` | Pareto-frontier plot generation, report tables |

## 16. Benchmark Architecture

A config-driven runner iterates over the full (model type × feature tier ×
bit-width) grid, executing both a plaintext-inference benchmark and an
FHE-inference benchmark per configuration, each repeated a fixed minimum
number of trials (≥5), recording mean/std latency, peak memory, and
ciphertext size where applicable, to a structured results file (e.g.
CSV/JSON) keyed by configuration hash.

## 17. Experiment Architecture

Every experiment run is defined by a single versioned config file
(model type, feature tier, bit-width, calibration set reference, random
seed, library versions). No experiment result is accepted into the final
report without a corresponding config file checked into the repository.

## 18. API Architecture

A minimal FastAPI service exposing:
- `POST /encrypt-infer`: accepts ciphertext, returns encrypted result
  (server-side).
- Client-side logic (quantize/encrypt/decrypt) is a library/script, not a
  separate network service, since the client is the trusted party and has
  no reason to expose these operations over a network boundary.

This API layer exists to demonstrate the real client/server split; it is
not intended as a scalable production deployment (see §20).

## 19. Testing Architecture

- Unit tests for: time-based split correctness (no future leakage), feature
  engineering leakage audit, quantization parameter consistency between
  client and compiled model, encrypted-vs-plaintext correctness per model.
- Integration test for the full client → server → client round trip on at
  least one configuration per model type.
- Benchmark harness has its own smoke test (small grid, few trials) that
  must pass before a full benchmark run is executed.

## 20. Repository Structure

```
/data                  # raw + processed data artifacts (not committed if large)
/src
  /data                # ingestion, split, leakage audit
  /features             # engineering, importance ranking, tiers
  /train                # plaintext model training + evaluation
  /fhe
    /compile             # Concrete-ML compilation per config
    /validate            # correctness checks
  /client                # key gen, quantize, encrypt, decrypt
  /server                # FastAPI encrypted-inference service
  /benchmark             # grid runner, repeated trials
  /analysis               # Pareto-frontier plots, report tables
/configs                # versioned experiment configs
/results                # benchmark outputs, plots
/tests                  # unit + integration tests
/docs
  prd.md
  architecture.md
  instructions.md
  plan.md
report.md               # final technical report
```

## 21. Deployment Architecture

Local/single-machine deployment only for this project's scope: a client
script and a locally-run FastAPI server process (optionally containerized
as a final polish item, per prd.md §17 stretch goals). No cloud
infrastructure, autoscaling, or managed deployment is in scope.
