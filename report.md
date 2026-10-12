# Ciphraud: Latency-Bounded Privacy-Preserving Fraud Detection Under FHE

Technical report (draft for review). Every number below is read from a committed result file or a named document; the
traceability table in section 14 maps each headline figure to its script, config and record. Nothing here is estimated.

## 1. Abstract and research question

Ciphraud evaluates how feature count, quantization bit-width and model complexity affect the predictive performance and
the computational cost of fraud detection when the inference server only sees data encrypted with TFHE (Concrete-ML 1.9.0).
On the IEEE-CIS Fraud Detection data (time-based split, validation partition) we compile Logistic Regression (LR),
XGBoost and a quantization-aware-trained MLP, measure each configuration, and run the three model types through a real
client/server round trip.

Main findings (validation partition; latency from one input repeated five times unless stated):
(1) Quantization bit-width decides whether encrypted inference is accurate: 8 bits fails the project's accuracy gate for
every LR and XGBoost tier, while 16-bit LR and 14-bit XGBoost pass at every tier. (2) Cost differs by orders of magnitude:
LR answers in 8.6-30.9 ms, XGBoost needs 2,651-4,099 s per encrypted request, at about 0.15-0.17 higher PR-AUC.
(3) The quantized MLP sits in between on cost but **all six MLP configurations failed the T3 quantization-accuracy gate**
(three of them were also infeasible under the memory limit), so the MLP is not a validated fraud model.
(4) A client/server implementation completed encrypted round trips for all three model types with exact integer agreement;
the XGBoost round trip is a single request that needed a raised WSL memory limit after two aborted attempts.
(5) A literature latency of about 6 ms for encrypted XGBoost was not reproduced. We make no speed-up, production-readiness
or general-security claim. All accuracy numbers are validation numbers; no test-partition evaluation had been run when this report was written (section 12). **Update:** a single frozen test evaluation was run afterwards; see the README section "Frozen test-partition evaluation" and `results/phase12_test_eval/summary.md`.

## 2. Motivation and problem definition

Card-fraud scoring needs transaction features that are sensitive. With FHE, a client encrypts the feature vector, a server
evaluates the model on ciphertext without decrypting, and only the client, who holds the secret key, can read the result.
FHE inference is expensive, so the practical question is how much accuracy and latency the available knobs trade off.
This is a reproduce-and-extend study, not a claim of novelty. It covers encrypted **inference**, not encrypted training.

## 3. Related work

- **FHE and TFHE.** Fully homomorphic encryption was first constructed by Gentry (2009, "Fully homomorphic encryption using
  ideal lattices", STOC, pp. 169-178). TFHE (Chillotti, Gama, Georgieva, Izabachene, *J. Cryptology* 33, 2020) is the fast-bootstrapping
  scheme family that Concrete-ML's backend builds on; the programmable-bootstrapping (table-lookup) form used by Concrete is described in
  Zama's documentation and the Stoian et al. paper below, not in the TFHE paper itself.
- **Concrete-ML.** Zama's library compiles quantized scikit-learn-style models to TFHE circuits. Stoian et al. describe
  designing neural networks under TFHE's integer and table-lookup constraints (arXiv:2302.10906 / ePrint 2023/257).
- **Models.** XGBoost (Chen and Guestrin, KDD 2016, arXiv:1603.02754). Quantization-aware training uses Brevitas
  (github.com/Xilinx/brevitas; a PyTorch quantization library).
- **Alternatives.** Trusted execution environments: see Costan and Devadas, "Intel SGX Explained" (ePrint 2016/086), for the
  hardware-trust model. MPC: Yao, "How to generate and exchange secrets" (FOCS 1986, pp. 162-167), which introduced secure two-party computation. Federated learning: McMahan et al.
  (AISTATS 2017, PMLR 54:1273). Differential privacy: Dwork, McSherry, Nissim, Smith (TCC 2006).
- **Dataset.** IEEE-CIS Fraud Detection (Kaggle competition; transaction + identity tables).
- **Prior fraud-under-FHE work.** `docs/prd.md` section 11 cites an encrypted-XGBoost/NN fraud thesis (about 6 ms XGBoost, 296 ms
  NN) and a peer-reviewed LR/XGBoost FHE paper. The source documents are not in this repository and were not re-read, so
  their setups are unverified (section 8.4).

Sources that were looked up for this draft: Gentry (DOI 10.1145/1536414.1536440), Yao (DOI 10.1109/SFCS.1986.25), Brevitas repository, TFHE (DOI 10.1007/s00145-019-09319-x, ePrint 2018/421), arXiv:2302.10906,
ePrint 2016/086, arXiv:1602.05629 (PMLR v54 mcmahan17a), TCC 2006 LNCS 3876:265-284, arXiv:1603.02754. The authors of the
Concrete-ML paper are Stoian, Frery, Bredehoft, Montero, Kherfallah and Chevallier-Mames.

## 4. Threat model and trust boundaries

Authoritative text: `docs/architecture.md` section 10. The client holds plaintext features and the secret key and decrypts. The
server holds the compiled model (plaintext parameters) and evaluation keys, and sees ciphertexts and request metadata
(timing, sizes, frequency). The only security claim: an honest, hardware-untrusted inference server cannot recover plaintext
features or the plaintext decision from what it is given.

Out of scope and **not** protected: model confidentiality, timing/volume/size side channels, a malicious server returning wrong
results (no verifiable computation), client-side compromise, network metadata, training-data poisoning, multi-institution
collaboration, regulatory compliance. The Phase 10 implementation uses plain loopback HTTP with no TLS and no
authentication; it demonstrates the narrower statement that **the application protocol does not transmit the client's secret
key or plaintext feature vector to the inference server** (a wire audit recorded only evaluation keys and input ciphertexts).
It does not demonstrate OS-level isolation (client and server ran on one machine) or protection against any listed exclusion.
New metadata observed in Phase 10: a persistent key identifier links a client's requests, and upload/ciphertext sizes reveal
the model type.

## 5. Dataset, split, preprocessing and leakage controls

(`docs/eda.md`, `docs/pipeline.md`.) Fraud rate 3.499%. Split by quantiles of the `TransactionDT` range (0.6 / 0.8):
train 380,815, validation 105,088, test 104,637 rows. Random k-fold CV is not used; hyperparameter selection uses an
expanding-window temporal CV that never reaches the test partition. Encoders and imputers are fitted on train only and
record their fit index. `src/data/leakage.py` runs six fail-loud audits (disjoint/complete partitions, temporal ordering,
no tied timestamp across a boundary, CV never reaching test, banned columns absent, fitted statistics only from the allowed
index), each with a failing-case test. Class imbalance uses train-only class weights. LR standardization is train-only.
`TransactionID`, `TransactionDT` and `isFraud` are excluded from the features.

## 6. Plaintext models and methodology

Primary metric PR-AUC (random baseline about 0.035), with ROC-AUC, precision, recall, F1, F2, confusion matrices and
FP/FN analysis for the strongest model. Full-feature baselines (832 columns, validation, `docs/baselines.md`): LR PR-AUC 0.4561,
XGBoost 0.5711 (ROC-AUC 0.9143; at the F1-optimal threshold 1,359 false positives and 1,995 false negatives).
Feature importance is XGBoost `total_gain`. Tiers `top_20/50/100` are pure top-k by rank, re-tuned by temporal CV
(`docs/features.md`): LR PR-AUC 0.3497 / 0.3884 / 0.4223, XGBoost 0.5227 / 0.5442 / 0.5678 (retention of the full model
76.7-92.6% and 91.5-99.4%). XGBoost seed standard deviations are below 0.007; LR is deterministic.
The validation partition selected the early-stopping tree count, decision thresholds and tiers.

## 7. FHE compilation and encrypted-inference methodology

Concrete-ML 1.9.0 / concrete-python 2.10.0, Python 3.12.3, seed 42, one machine (Intel i5-9500, 6 cores, 7.95 GB RAM,
WSL2 Ubuntu 24.04). Each model is quantized, calibrated on training data only, and compiled once per
model x tier x bit-width. Four result classes are kept distinct: float, clear-quantized, simulated, real FHE. Gates:
T0 transfer from the plaintext model; T1 real encrypt-run-decrypt exact match on a 2-row stratified sample (one fraud,
one legitimate); T2 simulation exact match (LR full validation, XGBoost and MLP 5,000 seeded rows); T3 quantization
accuracy (decision agreement >= 0.99 and PR-AUC drop <= 0.01 versus float; project-chosen bars). Latency is five repeated
executions of validation row 32,148 (one legitimate transaction), with compile time, key generation, encryption,
evaluation and decryption timed separately. Infeasible configurations are recorded, never dropped.

## 8. Results

### 8.1 Logistic Regression and XGBoost (Phase 8, `docs/research.md`)

LR scaler erratum: an early LR circuit omitted the standardization step, so the first LR quantization "failure" was mostly that
defect; it was fixed, all six LR configurations were re-run, and the pre-fix results are retained. The table is the corrected
set. Latency is mean +/- std of five repeats of one input. PR-AUC columns are validation-partition values, not test results.

| Config | T3 | Quantized / float PR-AUC | Latency | Compile | Peak RSS | Bootstraps | Ciphertext in / out |
|---|---|---|---|---|---|---|---|
| lr_top20_bits8 | fail | 0.3218 / 0.3484 | 6.48 ms +/- 0.59 | 92.9 s | 922 MB | 0 | 480 / 10,192 B |
| lr_top20_bits16 | pass | 0.3482 / 0.3484 | 8.57 ms +/- 0.87 | 93.7 s | 920 MB | 0 | 480 / 17,656 B |
| lr_top50_bits8 | fail | 0.2496 / 0.3889 | 11.60 ms +/- 3.17 | 97.1 s | 1,440 MB | 0 | 1,200 / 9,456 B |
| lr_top50_bits16 | pass | 0.3886 / 0.3889 | 15.40 ms +/- 2.74 | 94.5 s | 1,442 MB | 0 | 1,200 / 16,928 B |
| lr_top100_bits8 | fail | 0.2829 / 0.4224 | 18.49 ms +/- 0.72 | 96.0 s | 2,104 MB | 0 | 2,400 / 10,656 B |
| lr_top100_bits16 | pass | 0.4221 / 0.4224 | 30.92 ms +/- 1.26 | 96.3 s | 2,105 MB | 0 | 2,400 / 18,128 B |
| xgboost_top20_bits8 | fail | 0.3309 / 0.5227 | 1,096.7 s +/- 12.9 | 36.3 s | 2,896 MB | 248,452 | 480 / 5,868,336 B |
| xgboost_top20_bits14 | pass | 0.5218 / 0.5227 | 4,098.7 s +/- 1,148.8 | 34.4 s | 3,433 MB | 383,776 | 480 / 5,868,336 B |
| xgboost_top50_bits8 | fail | 0.3064 / 0.5442 | 931.1 s +/- 212.8 | 56.1 s | 2,417 MB | 151,986 | 1,200 / 3,589,848 B |
| xgboost_top50_bits14 | pass | 0.5429 / 0.5442 | 2,650.6 s +/- 771.9 | 73.1 s | 3,356 MB | 234,768 | 1,200 / 3,589,848 B |
| xgboost_top100_bits8 | fail | 0.2824 / 0.5678 | 1,021.2 s +/- 237.3 | 54.8 s | 2,645 MB | 188,074 | 2,400 / 4,442,232 B |
| xgboost_top100_bits14 | pass | 0.5710 / 0.5678 | 3,687.5 s +/- 840.3 | 111.0 s | 3,468 MB | 290,512 | 2,400 / 4,442,232 B |

XGBoost at 16 and 15 bits is infeasible for all tiers (Concrete supports table lookups of at most 16 bits; the tree circuit
needs n_bits + 2). LR above 16 bits fails with `NoParametersFound`. Hence bit-widths are model-specific (LR 8/16, XGBoost 8/14).
Observations: more features help accuracy once quantization is adequate; XGBoost cost follows tree count (358 / 219 / 271), not
feature count; the XGBoost 14-bit bootstrap key is about 1.8-1.9 GB. `xgboost_top100_bits14` quantized PR-AUC is 0.0032
above the float model, which is single-split noise and not evidence that quantization helps.

### 8.2 Quantized MLP (Phase 9, `docs/fhe_mlp.md`)

Six cells (validation-only accuracy; 3 tiers x 3/4 bits; QAT with Brevitas, accumulator capped at 16 bits). 5, 6 and 8 bits were probed and recorded as
infeasible here (keys exceed memory, or no valid parameter set).

| Config | Status | Quantized / float PR-AUC | Decision agreement | T3 | T2 | Key material | FHE latency |
|---|---|---|---|---|---|---|---|
| mlp_top20_bits3 | failed gate | 0.3924 / 0.4892 | 0.9788 | **fail** | pass | 0.65 GB | 5.17 s +/- 0.59 |
| mlp_top20_bits4 | failed gate | 0.4070 / 0.4892 | 0.9789 | **fail** | pass | 2.03 GB | 72.61 s +/- 5.67 |
| mlp_top50_bits3 | failed gate | 0.4173 / 0.4696 | 0.9703 | **fail** | **fail (1 of 5,000 rows)** | 1.19 GB | 17.13 s +/- 1.14 |
| mlp_top50_bits4 | infeasible (key memory) | 0.4324 / 0.4696 | 0.9656 | **fail** | not run | 4.44 GB | none |
| mlp_top100_bits3 | infeasible (key memory) | 0.4494 / 0.4908 | 0.9764 | **fail** | not run | 3.24 GB | none |
| mlp_top100_bits4 | infeasible (key memory) | 0.4392 / 0.4908 | 0.9706 | **fail** | not run | 4.80 GB | none |

**All six MLP configurations failed T3.** A three-seed plaintext study (seeds 42-44) shows the QAT-minus-float gap
(0.034-0.086 PR-AUC) is 4-20 times the seed standard deviation and has the same sign in all nine runs, so it is a property of
the quantized architecture, not noise. The single T2 failure (`mlp_top50_bits3`, validation row 97,498) is a calibration-coverage
effect: one row with extreme feature combinations overflowed the circuit's 8-bit accumulator range. The gate was not loosened.
`mlp_top20_bits4` was trained before two memory-only fixes (D7/D8); a later retrain reproduced its PR-AUC exactly.
The three infeasible cells were excluded by the pre-existing 2.4 GB key-material gate, which was not changed.

### 8.3 Pareto frontiers

Accuracy is the quantized validation PR-AUC; cost axes are latency, peak RSS and request ciphertext. Over the T3-passing set,
the frontier on every axis is the three 16-bit LR configurations followed by `xgboost_top50_bits14` and
`xgboost_top100_bits14`. `xgboost_top20_bits14` is dominated on the means, but not within one standard deviation, so the latency
ordering of the XGBoost tiers is not statistically established; ordering by ciphertext size and bootstrap count is
deterministic. No MLP point is on the T3-passing frontier. Between LR and XGBoost, the trade is about 0.15-0.17 PR-AUC for
1.2e5-4.8e5 times the latency, about 200-320 times the ciphertext and 1.6-3.7 times the peak memory. Figures:
`results/phase8_research/pareto/` and `results/phase9_mlp/pareto/`.

### 8.4 Prior-art comparison

Measured encrypted XGBoost latency (219-358 trees) is 931-4,099 s, which is 1.6e5-6.8e5 times the cited 6 ms. One programmable
bootstrap costs 4.4-12.7 ms here, so 6 ms is about one bootstrap. The explanation of the gap is not known; candidates (much
smaller model, different scheme or library, different hardware, different definition of latency) are untested. Measured MLP
latency is 17-245 times the cited 296 ms for the three cells with FHE data, with the same caveat. LR latencies are of the same
order as 6 ms but LR has no bootstraps and is a different model, so that is not agreement.

### 8.5 Client/server round trips (Phase 10, `docs/client_server.md`)

FastAPI server and HTTP client over loopback on one machine; secret key stays client-side; evaluation keys and one input
ciphertext are the only request bodies (wire audit). Exact agreement means the decrypted integers equal the clear-quantized
integers and the recorded Phase 8/9 values.

| Model | Requests | Exact | Server FHE time (mean of requests) | Notes |
|---|---|---|---|---|
| lr_top20_bits16 | 7 | 7/7 | 75 ms | first request 297 ms (cold) |
| mlp_top20_bits3 | 7 | 7/7 | 16.17 s | pipeline demonstration only (see below) |
| xgboost_top50_bits14 | 1 | 1/1 | 2,445.06 s (guest monotonic clock; about 2,650 s real) | 219 of 219 integers equal; probability 0.0999311; key upload 308,396,952 B; response 3,590,120 B; server peak 3,112 MB |

The MLP deployment used `mlp_top20_bits3`, which passed the relevant execution/integrity checks but failed the Phase 9 quantization-accuracy gate. It demonstrates the pipeline and is not a validated or production fraud model.
XGBoost needed two attempts to be aborted by the pre-registered memory guards at the default WSL limit of about 3.9 GiB (registration
needs about 2.1 GB transiently and the first evaluation expands the keys to about 3 GB); the third attempt, with identical code,
configuration and guards and the WSL limit raised to 5 GB, passed. The XGBoost result is **one cold request (n = 1)** and is not
a benchmark or speed-up claim. The Phase 10 LR and MLP request times in this table are about 2-3 times higher than the Phase 8/9
in-process timings and than the first (superseded) Phase 10 recordings of the same code path (LR 4-115 ms, MLP 6.1-9.5 s per request); this is not explained (section 12).

## 9. Latency, memory, key material, ciphertext and accuracy trade-offs

| Model (best passing or demonstrated config) | PR-AUC (validation) | Latency | Key material | Ciphertext in + out | Peak RSS |
|---|---|---|---|---|---|
| LR 16-bit, top_100 | 0.4221 | 30.92 ms | about 0 (no bootstrap key) | 20,528 B | 2,105 MB |
| XGBoost 14-bit, top_100 | 0.5710 | 3,687.5 s | bootstrap key 1.94 GB | 4,444,632 B | 3,468 MB |
| XGBoost 14-bit, top_50 | 0.5429 | 2,650.6 s | 1.80 GB | 3,591,048 B | 3,356 MB |
| MLP 3-bit, top_20 (fails T3) | 0.3924 | 5.17 s | 0.65 GB | see `docs/fhe_mlp.md` | 1,671 MB |

Raising XGBoost from 8 to 14 bits turns every failing T3 into a pass at 1.54 times the bootstraps, 2.14-2.31 times the key size and
2.9-3.7 times the latency. For LR, 8 to 16 bits costs 1.3-1.7 times the latency and no extra memory.
In the client/server setting, serialization and framework overhead was small next to evaluation (XGBoost 0.109 s of 2,445.18 s per request;
LR about 13 ms; MLP about 16 ms), and key upload (6.10 s, 308 MB compressed) is a one-time cost per key set.

## 10. Client/server architecture and key ownership

Endpoints: `GET /v1/models`, `POST /v1/keys`, `POST /v1/infer`, `DELETE /v1/keys`; bodies are Concrete's serialized bytes. The
server loads only `server.zip` and has no code path to the secret key; keys live in server memory only, one key set at a time,
bound to the circuit by a client-artifact hash. Deployment artifacts come from Concrete-ML's `FHEModelDev`. XGBoost uses a
three-process staged client (keys, serve, decrypt) to keep client and server key sets from being resident together, with a
fail-closed memory monitor (abort below 250 MB available or above 700 MB swap). Details: `docs/client_server.md`, `docs/architecture.md`.

## 11. Comparison with TEE, MPC, federated learning, differential privacy and tokenization

These address different threat models; none was implemented here and no measurement below is ours.
- **TEE:** much lower latency (near plaintext) when the hardware vendor and enclave can be trusted, but a hardware-trust and
  side-channel surface. FHE only needs the server to run the circuit honestly; in this project's measurements that costs
  seconds to about an hour per request, so a TEE is the practical choice whenever hardware trust is acceptable.
- **MPC:** suited to several institutions jointly computing without a trusted party; it needs interaction and a multi-party setting. This is a single-client, single-server design.
- **Federated learning:** keeps training data local; it does not hide a transaction from the party that scores it at inference time.
- **Differential privacy:** bounds what a released model or statistic reveals about training records; it does not protect a live query's features.
- **Tokenization:** replaces identifiers such as card numbers; it does not make the remaining features or the score confidential from the server.

## 12. Limitations and unresolved questions

- **Validation only.** No number is a test-partition number; the validation set also chose trees, thresholds and tiers, so the
  absolute PR-AUCs are optimistic. Differences between adjacent tiers are single-split point estimates without intervals.
- **Latency evidence is thin.** Five repeats of one legitimate row on one machine; XGBoost relative standard deviation is 23-29% in five of six configurations; within-run ordering of the 14-bit tiers is not established.
- **One machine, one library version, one operating environment.** No GPU, no network, no cloud.
- **Clock caveat (investigated, cause unresolved).** The WSL VM's monotonic clock, used for all recorded durations, ran behind real time. Phase 8 measured this directly with a sleep probe
  (monotonic 180.003 s vs real-time 201.505 s, about 10-11%) and found a 5-11% gap in checkpoint timestamps (`docs/research.md` section 7.5). Phase 10 measured about 8% from sampler gaps (about 2,650 s real vs 2,445 s monotonic). Hence recorded durations
  in Phases 8-10 are lower bounds on real elapsed time by roughly 5-11%. Phase 9 timings come from the same kind of VM and were not separately probed, so the same direction of bias is expected but its size is not measured. The cause (WSL2/Hyper-V timekeeping under full-core load is the leading suspect) was not tested.
- **Unexplained timing differences.** Phase 10 recorded LR and MLP evaluation times about 2-3 times those of the earlier recordings of the same circuits (LR 15-297 ms vs 4-115 ms; MLP 13.9-20.3 s vs 6.1-9.5 s per request) and about 3-9 times Phase 8/9's in-process means. Both
  Phase 10 recordings ran the same code path (the difference between them is only a removed import in the sender process). Candidate causes (host load, thermal or power state, memory configuration, the clock effect above) were not tested, so the numbers are reported as recorded and no cause is asserted.
- **Memory:** the XGBoost run needs about 5 GB of WSL memory; whether it fits in less was not tested. An unexplained +216 MB final memory step appears in aborted attempt 2.
- **MLP:** all six configurations fail T3; the MLP is not a validated model; the 3 larger cells are infeasible under the 2.4 GB key gate; one T2 exact-match failure is recorded.
- **Security:** loopback HTTP, no TLS or authentication, same-machine client and server, no protection against malicious servers or metadata leakage (section 4).
- **Not encrypted training.** Models were trained in plaintext.
- **Reproducibility:** provenance records per-file source hashes plus a base commit (the code was not committed before the runs); some Phase 8 records name the commit before Phase 8; raw logs for attempt 3 (about 11 MB) are local-only with hashes recorded; the committed source hashes match LF line endings and will differ on a CRLF checkout.

**Is a final test-partition evaluation needed?** Not for the research question as posed. That question concerns relative cost and the effect of
quantization (clear-quantized versus float, on identical rows), and all headline comparisons are relative. It would be needed to state an
absolute fraud-detection performance. It is also not cheap to do now: the FHE handoffs contain validation rows only, so the clear-quantized
circuits would have to be rebuilt for the test rows, and the test partition has stayed untouched across phases by design. If it is done, the protocol
should be frozen first: the three-tier models, quantization, decision thresholds (chosen on validation), and the exact metrics, run once, no tuning
afterward. No such evaluation was run for this report.

### 12.1 Which claims rest on which evidence (and the test partition)

| Claim type | Examples in this report | Needs a test-partition evaluation? |
|---|---|---|
| Absolute predictive metrics | PR-AUC values, decision agreement, confusion counts, plaintext baselines | These are validation-only and labelled so. An untouched test run is needed only to state an absolute expected performance; none of the conclusions depends on it |
| Relative accuracy comparisons | 8 vs 14/16 bits; LR vs XGBoost gap; tier ordering; T3 pass/fail (quantized vs float on the same rows) | Not needed for the quantization effects (same-row comparisons). Adjacent-tier differences are single-split estimates and the validation set selected trees, thresholds and tiers, so those small differences are weakly supported |
| Relative FHE cost | latency, memory, ciphertext and key sizes, bootstrap counts, compile time, round-trip exactness | No: independent of labels and of the data split |

Availability: the processed test matrix exists locally (`data/processed/test.pkl`, 104,637 rows, about 223 MB), produced by the train-fitted Phase 2
pipeline, and the test labels have not been read by any phase (`"test_partition_touched": false`). A test run would need no fitting or tuning: transform
with the already-fitted train-only preprocessors, use the validation-chosen thresholds, and evaluate the already-selected models. What it would need that does not exist yet: the FHE handoffs
(`data/fhe_handoff`, about 2.1 GB) hold validation rows only, so test rows would have to be exported per tier, and the quantized circuits rebuilt in clear mode (no encrypted execution).
Rough cost, by arithmetic from recorded compile times and sizes and not measured: plaintext and clear-quantized scoring of nine LR/XGBoost/MLP configurations in minutes,
about 10-15 minutes of Concrete-ML compilation (35-111 s per circuit in Phases 8-9), tens of MB of exported test arrays and a few small result files; no FHE runs and no keys.
**Proposed frozen protocol (awaiting approval, not run):** (1) freeze the model files, tiers, bit-widths, calibration sets and thresholds by hash before touching test labels;
(2) apply only the stored train-fitted transforms; (3) compute PR-AUC, ROC-AUC, precision/recall/F1/F2 at the frozen thresholds once, for float and clear-quantized models, in one script with a config;
(4) record the result files with hashes, regardless of outcome; (5) no retuning or re-selection afterward, and report any drop from validation as found.

## 13. Conclusion and future work

Under Concrete-ML 1.9.0 on a 6-core CPU, the cost of privacy-preserving fraud scoring depends sharply on the model: 16-bit LR is milliseconds,
14-bit XGBoost is tens of minutes per request for about 0.15-0.17 more PR-AUC, and a 3-4 bit MLP lies between but fails the accuracy gate. Bit-width,
not feature count, determined whether quantization was accurate, and XGBoost cost followed tree count. Encrypted client/server inference was demonstrated
for all three models with exact integer agreement, with the stated scope and caveats. Future work: a pre-registered single test-partition evaluation; repeated
latency measurements over many inputs and a clock-controlled environment; GPU-accelerated FHE; a smaller-memory XGBoost; TLS/authentication and
containerization (Phase 11); an investigation of the literature latency gap with the cited setup.

## 14. Reproducibility and traceability

Versions: `requirements-fhe.txt` (concrete-ml 1.9.0, concrete-python 2.10.0, xgboost 1.6.2, scikit-learn 1.5.0, numpy 1.26.4, pandas 3.0.5). Git: Phase 8 `f11d750`, Phase 9 `afb09da`, Phase 10 `4945dea`.

| Result | Script / config | Record |
|---|---|---|
| Split, leakage audit | `python -m src.data.pipeline --config configs/phase2/pipeline.yaml` | `results/phase2_pipeline/` |
| Plaintext baselines | `python -m src.train.pipeline --config configs/phase3/baselines.yaml` | `results/phase3_baselines/metrics.json` |
| Tiers and tier PR-AUC | `configs/phase4/`, `src/features/` | `results/phase4_features/` |
| LR / XGBoost grid, Pareto | `python -m src.benchmark.phase8_grid --config configs/phase8/research_grid.yaml`; `python -m src.analysis.phase8_pareto` | `results/phase8_research/*/metrics.json`, `pareto/` |
| MLP grid, combined Pareto | `python -m src.benchmark.phase9_grid`, `configs/phase9/mlp_grid.yaml`; `python -m src.analysis.phase9_pareto` | `results/phase9_mlp/` |
| Client/server round trips | `python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model <id> [--staged]` | `results/phase10_client_server/` |
| Evidence validators | `python -m src.analysis.phase8_results`, `phase9_results`, `phase10_results --check-code-hashes` | exit status |

Narrative sources: `docs/research.md`, `docs/fhe_mlp.md`, `docs/client_server.md`, `docs/architecture.md`. Where a document's numbers and the
`metrics.json` files disagree, the `metrics.json` files govern. This report was checked against `docs/research.md`, `docs/fhe_mlp.md`, `docs/client_server.md` and the Phase 10 `summary.json`; the
cross-check script is `src/analysis/report_check.py`.
