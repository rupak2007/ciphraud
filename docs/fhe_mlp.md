# Phase 9 — Quantized MLP under FHE (living document)

**Status: IN PROGRESS (2026-09-27).** Code, tests, the export, the API probe and the feasibility probe are done; the plaintext evaluation and the 6-configuration grid are running. Sections that need results say so. Every number below traces to a file under `results/phase9_mlp/` or to `configs/phase9/mlp_grid.yaml`; nothing is estimated.

## 1. Objective

`docs/plan.md` Phase 9: extend the LR/XGBoost benchmark to a **quantized MLP** (Brevitas quantization-aware training, per `docs/architecture.md` §5), compile it with Concrete-ML, validate its correctness, log **TLU (programmable-bootstrap) count and compile time** per configuration, and integrate it into the same Pareto analysis. It also closes **PRD FR5** ("train and evaluate a quantized MLP as a plaintext baseline"), which Phase 3 never did.

## 2. Decisions (approved 2026-09-26)

| | Decision | Why it was needed |
|---|---|---|
| D1 | One bit-width knob `b = n_w_bits = n_a_bits`; accumulator capped at `n_accum_bits = 16`. Grid bit-widths = the largest `b` feasible for all three tiers, plus the next lower one. 8 is probed and logged | The plan's "the same bit-width grid as Phases 6–8" has no single meaning for an MLP, which has weight, activation and accumulator precision |
| D2 | **T3 = QAT clear-quantized model vs a float twin** (same architecture, data, seed, class weights, optimizer, epoch budget; no quantization), full validation, Phase 8's bars (decision agreement ≥ 0.99, PR-AUC drop ≤ 0.01) | QAT trains the network already quantized, so it has no float version of itself; the comparison therefore includes training-run variance, which the seed-stability spread (D5) sizes |
| D3 | **T0 = the checkpoint written after training reloads to exactly the same clear-quantized outputs** | Phase 8's T0 checked a Windows→WSL transfer; the MLP is trained inside WSL, so nothing crosses that boundary |
| D4 | One fixed small architecture (Concrete-ML `NeuralNetClassifier` defaults: `n_layers=3`, hidden width = 4 × input features, ReLU), full train split, class-weighted loss, early stopping and threshold on **validation**, **no hyperparameter search** | Deviates from Phase 4's per-tier CV re-tuning of LR/XGBoost; validation-based early stopping carries the same caveat as XGBoost's (`docs/research.md` §7.7) |
| D5 | Plaintext seed stability over 3 seeds × 3 tiers at the primary bit-width | CLAUDE.md §11 |
| D6 | Local commits before the grid runs, so `provenance.json` records the real code commit (Phase 8's stale `e85d16a` problem) | Pushing still only on the user's instruction |

## 3. Concrete-ML facts, measured (`src/fhe/probes/mlp_api_probe.py` → `results/phase9_mlp/api_probe.json`, synthetic data)

- `module__n_outputs` and `module__input_dim` are **forbidden** as kwargs (set in `.fit` from the data); `random_state` is **rejected** — training is seeded with `torch.manual_seed` (verified reproducible).
- `n_layers=3` gives `input → 4·input → 4·input → 2`; e.g. 20 features → weights (80×20), (80×80), (2×80).
- **The accumulator ceiling is enforced by the compiler, not by refusing to compile:** with `n_accum_bits=16`, widths 2–8 all compiled on synthetic data (max integer bit-width 6 → 16). The binding constraint turned out to be **key size** (§5), not the accumulator.
- Integer level: `quantized_module_.quantized_forward(int_input, fhe="disable")` **equals** `circuit.simulate` exactly, and `post_processing(dequantize_output(ints))` reproduces `predict_proba`; so T1/T2 can be exact integer checks like XGBoost's. The input quantizer has `n_bits = n_a_bits`; the output quantizer 5 bits at `b=4`.
- **Concrete-ML's two-class `predict_proba` applies an element-wise sigmoid to both logits** (rows do not sum to 1). All Phase 9 probabilities are `sigmoid(logit_1 − logit_0)` from the dequantized logits (`src/fhe/compile/mlp.py::fraud_probability`), for the QAT model and the float twin alike.
- Serialization: `pickle` fails after compile (`ctypes objects containing pointers`); Concrete-ML's `dumps/loads` works **only with `callbacks="disable"`**, and reloads to exactly the same clear-quantized outputs. A model trained with `criterion__weight` serializes but `loads` then fails (`Unexpected key(s) in state_dict: "weight"`), so the loss weight is dropped for the dump only (`dump_model`); tested.
- `set_params(max_epochs=...)` changes only the Concrete-ML wrapper, **not the inner skorch net that trains**; the epoch count must be set on `model.sklearn_model` (found because a "zero-epoch refit" silently trained another epoch; tested in `test_restore_weights_...`).
- **Training cost:** every Concrete-ML `fit` call re-exports the network to ONNX and re-quantizes on the whole training set — a fixed ≈ 60–70 s per call on `top_20` — while one epoch itself is ≈ 20 s. The trainer therefore fits once, runs later epochs on the inner net (`partial_fit`), chooses the stopping epoch from the validation PR-AUC of the QAT (fake-quantized) forward, and exports the best weights once. The reported score is the **clear-quantized** validation PR-AUC of the final model (recorded beside the selection curve).
- Convergence check that fixed the pinned hyperparameters (one check on `top_20`, 4 bits, not a search): lr 0.003 with batch 2048 did **not** learn (validation PR-AUC 0.13); lr 0.001 with batch 512 did (0.31 → 0.34 in three epochs).

## 4. Data handoff (`src/fhe/export_mlp.py`, `configs/phase9/mlp_export.yaml`)

Every earlier handoff lacks the training labels (their models were trained on Windows), so `save_handoff` gained an optional `y_train` and an optional `reference_val_prob` (additive; existing handoffs keep their exact layout, tested). The three tier matrices are identical to the Phase 8 LR handoffs (asserted against the real files), NaN-free, with train fraud rate 3.42% and validation 3.89%. The MLP standardizes with a **train-only** scaler (mean/scale in `results/phase9_mlp/export/{tier}/mlp_{tier}_scaler.json`, hash in the manifest; a leak check is tested). The test partition is never referenced.

## 5. Feasibility (R5) — `results/phase9_mlp/feasibility.json`

Method: a short training run (2 epochs, 20,000 seeded train rows) on the real standardized tier data, compile against 3,000 train rows + each feature's min/max row, and — when the predicted key material is ≤ 2.4 GB — real key generation and one timed encrypted run. One process per candidate. A short-trained network only **indicates** the final circuit; the grid re-verifies every chosen configuration.

| Tier | b | PBS (TLUs) | Max integer bits | Bootstrap key | Key material | Compile | Real run | Outcome |
|---|---|---|---|---|---|---|---|---|
| top_20 | 2 / 3 / 4 | 400 / 560 / 640 | 6 / 7 / 10 | 687 / 747 / 888 MB | 0.83 / 1.01 / 1.26 GB | 8.7 / 8.6 / 9.5 s | 7.3 / 19.8 / 34.7 s | feasible; decrypted == simulated |
| top_50 | 2 / 3 / 4 | 1,200 / 1,600 / 2,000 | 7 / 9 / 11 | 489 / 451 / 908 MB | 0.61 / 0.65 / 1.32 GB | 8.0 / 8.6 / 8.4 s | 14.8 / 32.0 / 95.0 s | feasible; decrypted == simulated |
| top_100 | 2 / 3 / 4 | 2,800 / 3,200 / 4,400 | 7 / 9 / 11 | 656 / 451 / 1,114 MB | 0.77 / 0.65 / 1.63 GB | 11.1 / 19.9 / 19.6 s | 41.5 / 73.1 / 191.0 s | feasible; decrypted == simulated |
| all | 5 | 800 / 2,400 / 5,200 | 11 / 13 / 13 | 2.3 / 2.3 / 2.9 GB | 3.08 / 3.12 / 3.88 GB | ≈ 8–21 s | — | compiles; **infeasible here**: keys exceed the 3.8 GiB WSL memory (keygen not attempted) |
| all | 6 | 960 / 2,800 / 5,600 | 13 / 15 / 16 | 7.0 / 8.9 / 10.8 GB | 8.65 / 10.53 / 12.16 GB | ≈ 9–15 s | — | compiles; **infeasible here** (keys 8.7–12.2 GB) |
| all | 8 | — | — | — | — | — | — | **compile fails: `NoParametersFound`** at every tier |

(PBS/keys per tier are listed top_20 / top_50 / top_100 in the 5- and 6-bit rows.) The plan's expectation that the accumulator ceiling would block wider MLPs was **not** what limits this machine: 5-bit and 6-bit circuits compile; their bootstrap keys simply do not fit in 3.8 GiB, and 8 bits has no valid Concrete parameter set. Peak process RSS in the probe reached 3.43 GB for `top_100` at 4 bits (1.63 GB of keys), which is why the grid runner frees the training matrix before holding keys.

**Resulting grid (decision D1):** `b = 4` (largest feasible for all three tiers) and `b = 3` (next lower) × `top_20 / top_50 / top_100` = **6 configurations** (`configs/phase9/mlp_grid.yaml`). `b = 5, 6, 8` are recorded findings, not silently dropped. Measured probe latency of the 4-bit circuits (26–191 s per request) is 1–2 orders of magnitude below XGBoost's 14-bit runs (2,651–4,099 s) and 3–4 orders above LR's (8.6–31 ms) — an indication only; the grid measures it properly.

## 6. Gates

T0 exact checkpoint reload (D3); T3 QAT-vs-float-twin at Phase 8's bars (D2); T2 `circuit.simulate` vs the clear integer outputs, **exact**, on 5,000 seeded validation rows (simulate costs 0.02–0.10 s per row, so all 105,088 rows would take up to ≈ 3 h per configuration); T1 real encrypt→run→decrypt on the 2-row stratified sample, **exact**; latency = 5 repeated executions of validation row 32,148 (the row Phase 8 used), mean ± sample std, keygen and compile separate. A configuration whose keys exceed `max_key_material_gb` (2.4) is recorded as `infeasible_key_memory` with its plaintext gates instead of being attempted.

## 7. Results

*Pending — plaintext evaluation (FR5 + seed stability) and the 6-configuration grid are running.*
