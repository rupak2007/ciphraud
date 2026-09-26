# Pre-fix LR results (historical evidence - do not delete, do not treat as valid LR accuracy results)

Copied on 2026-09-25 before the six Phase 8 LR configurations were re-run. These were produced by code in which the
compiled Logistic Regression was built, calibrated and evaluated on RAW features while holding the coefficients of a
model trained on STANDARDIZED features (StandardScaler omitted). Their T3 / quantized-PR-AUC numbers therefore mix a
model-definition mismatch with quantization error and must not be cited as quantization results. Compile time, circuit
shape (PBS = 0) and ciphertext sizes are shown for reference only.
See Ciphraud_Project_Audit.md, Problem 15. The matching raw latency checkpoints are in
data/fhe_handoff/phase8/checkpoints_prefix_lr_scaler_bug/.
