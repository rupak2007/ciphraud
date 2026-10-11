# Ciphraud — portfolio summary, resume bullets and preprint outline (DRAFT)

Everything here is limited to what `report.md` and the committed results establish. Nothing is published, deployed or peer reviewed.

## Project summary (about 120 words)

Ciphraud is a research project on privacy-preserving fraud detection with fully homomorphic encryption. Using the IEEE-CIS fraud dataset
with a leakage-audited time-based split, I trained logistic regression, XGBoost and a quantization-aware MLP, compiled them to TFHE circuits with
Concrete-ML, and measured accuracy, latency, memory and key size across feature counts and bit-widths, with accuracy gates and Pareto analysis.
I built a FastAPI client/server implementation in which the server never receives the secret key or plaintext features, and ran encrypted
round trips for all three models with exact agreement to the clear computation. Findings include that 8-bit quantization fails for LR and XGBoost,
the MLP fails the accuracy gate, and a published 6 ms encrypted-XGBoost latency was not reproduced (measured 15-68 minutes per request).

## Truthful resume bullets

Pick the ones that fit the role; keep the caveats if asked.

- Built a research-grade pipeline for fraud detection under FHE (Concrete-ML/TFHE) on the IEEE-CIS dataset, with a time-based split and six automated leakage audits; compared
  LR, XGBoost and a quantized MLP over 3 feature-count tiers and 2-3 bit-widths, with documented infeasible configurations rather than dropped ones.
- Found and fixed a missing-standardization defect in my own encrypted logistic-regression circuit through an audit and regression tests; re-ran the affected grid and kept the pre-fix results as history.
- Implemented client/server encrypted inference (FastAPI, httpx, Concrete-ML deployment API): the application protocol does not transmit the secret key or plaintext features to the server,
  verified by a wire audit; exact integer agreement with clear computation for LR (7/7), MLP (7/7) and one XGBoost request (219/219 outputs).
- Designed fail-closed memory supervision and pre-registered abort guards for a 1.9 GB-key FHE workload on a 8 GB machine; documented two aborted attempts and a successful third run without loosening any guard.
- Measured the accuracy/latency/memory trade-off: 16-bit LR answers in 9-31 ms, 14-bit XGBoost takes 44-68 minutes per encrypted request for about 0.15-0.17 higher PR-AUC; reported that a literature figure of about 6 ms was not reproduced.
- Maintained research-integrity practices: provenance by per-file hashes, evidence validators, 500+ automated tests, and a traceability table linking every reported number to a script and config.

Avoid claiming: production deployment, a speed-up, publication, a validated MLP, or general "secure" or "private" systems.

## Preprint outline (DRAFT — not submitted, not reviewed)

1. Title idea: "Cost and accuracy trade-offs of encrypted fraud scoring with TFHE: LR, XGBoost and quantized MLP on IEEE-CIS".
2. Introduction and threat model (single honest, hardware-untrusted server; narrow claim).
3. Related work (FHE/TFHE, Concrete-ML, TEE/MPC/FL/DP alternatives).
4. Methods: temporal split and leakage controls, quantization, gates T0-T3, measurement protocol.
5. Results: Pareto frontiers, bit-width and feature-count effects, MLP negative result, key-size and memory limits.
6. Client/server demonstration and what it does and does not establish.
7. Limitations (validation-only accuracy, one machine, clock caveat, n=1 for XGBoost round trip) and a pre-registered test evaluation.
Before any submission: run the frozen test-partition evaluation, repeat latency measurements over many inputs on a controlled machine, and verify all citations.
