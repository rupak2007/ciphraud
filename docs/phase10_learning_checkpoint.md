# Phase 10 learning checkpoint — client/server encrypted inference

Written for the developer (CLAUDE.md section 16). Full evidence and numbers are in `docs/client_server.md`; this file explains the system.

## 1. What was built

A real client/server split around the already-validated Phase 8/9 circuits. Before Phase 10, every FHE run happened inside one
Python process (compile, keygen, encrypt, run and decrypt on the same object), so the security claim "the server never sees the
secret key or the plaintext" had never been exercised. Phase 10 puts a FastAPI server and an HTTP client around the circuits
and records what actually crosses the boundary.

## 2. Why it exists

The project's single security claim is only meaningful if keys and ciphertexts really travel between two parties.
Phase 10 demonstrates the narrow statement "the application protocol does not transmit the client's secret key or plaintext
feature vector to the inference server". It does not add TLS, authentication or isolation, and it is not a benchmark.

## 3. Where it fits

Runtime encrypted inference stage (architecture section 4). It consumes the compiled circuits of Phases 5-9 (LR, XGBoost, MLP) and
changes none of their models, bit-widths, gates or results.

## 4. Key concepts

- **Secret key vs evaluation keys.** The client generates both. Only the evaluation keys (large; 308 MB compressed for XGBoost,
  1.93 GB expanded) go to the server; the secret key stays client-side and is used to encrypt and decrypt.
- **Deployment artifacts.** `FHEModelDev.save` splits a compiled model into `server.zip` (circuit, no secrets) and `client.zip`
  (quantizer parameters and key specifications). The server loads only the first.
- **Quantize on the client.** The client converts floats to the integers the circuit expects before encrypting; the dequantized
  probability is computed after decryption.
- **Key binding.** A key set is bound to one circuit (sha256 of `client.zip` in the header). Foreign keys crash the native runtime,
  so the server rejects a mismatch up front. One resident key set at a time (memory).
- **Fail-closed memory supervision.** `MemWatch` samples memory every 0.1 s and kills only our own processes if
  pre-registered limits trip, so a 45-minute run cannot take down the VM.
- **Provenance by per-file hash.** Results carry sha256 of each source file and the config instead of a pre-run commit.

## 5. Important files and functions

- `src/fhe/deploy.py` rebuilds each circuit, proves it reproduces the recorded integers, saves the artifacts and manifest.
- `src/server/{app,registry,__main__}.py` implement `GET /v1/models`, `POST /v1/keys`, `POST /v1/infer`, `DELETE /v1/keys`.
- `src/client/{client,http,preprocess}.py` do keygen, preprocessing, quantize, encrypt, decrypt and the HTTP calls.
- `src/benchmark/phase10_roundtrip.py` is the runner: in-process LR/MLP flow and the staged XGBoost flow
  (`stage_keys`, `stage_serve`, `stage_decrypt`).
- `src/benchmark/memwatch.py` is the memory guard. `src/analysis/phase10_results.py` is the evidence validator.
- `src/analysis/phase10_correct_record.py` applies post-run corrections beside unchanged raw fields.

## 6. Flow

client: raw row -> preprocess -> quantize -> encrypt -> (HTTP) -> server: deserialize -> FHE evaluation -> serialize ->
(HTTP) -> client: decrypt -> dequantize -> probability and decision. The one-time step before that: the client uploads its evaluation keys.

## 7. What succeeded, what failed, what it established

- LR (`lr_top20_bits16`): 7/7 round trips exact. MLP (`mlp_top20_bits3`): 7/7 exact, as a pipeline demonstration only;
  that MLP failed the Phase 9 quantization-accuracy gate and is not a validated fraud model.
- XGBoost (`xgboost_top50_bits14`): attempts 1 and 2 aborted on the pre-registered memory guards at the default WSL limit of
  about 3.9 GiB. Attempt 3, same code, config and guards, with the WSL limit raised to 5 GB, passed: one cold request, 219 decrypted
  integers equal to the clear, simulated and recorded Phase 8 values, probability 0.0999311, secret keyset deleted afterwards.
- Established: the full encrypted path works across process boundaries for all three model types, and the only data bodies
  on the wire were evaluation keys and input ciphertexts (wire audit).
- **Not** established: any latency or speed-up claim (one request, n=1); that XGBoost runs in less than 5 GB; behaviour over a
  network, with TLS, authentication or a malicious server; any validity of the MLP as a fraud model.

## 8. Lessons from the aborts

1. The 2.4 GB key-material gate said nothing about the real runtime footprint: registration needs about 2.1 GB transient memory
   and the first evaluation expands the keys to about 3 GB. The 2.4 GB gate was left unchanged.
2. Fail-closed guards worked: two aborts cost minutes and no VM crash, and became recorded evidence instead of silent failures.
3. Do not stack infrastructure tweaks. A loader change would have saved only about 99 MB, so the clean answer was more memory,
   with everything else identical.
4. Derived "did the server start" flags can be wrong when a stage is killed; the correction mechanism fixes them beside the raw
   fields instead of rewriting history.
5. The VM's monotonic clock ran about 8% slower than real time, so recorded durations understate real time; this is disclosed
   and its cause is uninvestigated.

## 9. Tradeoffs

Loopback HTTP on one machine (no network effects, no isolation). Sync handler plus one lock (simple; one request at a time).
Staged XGBoost client (memory) instead of a single in-process client. No pre-run commit (user rule), replaced by file hashes.

## 10. What to study next

Concrete-ML `FHEModelClient/Server` source; TFHE bootstrapping and key sizes; why evaluation keys are so large for XGBoost
(many programmable bootstraps); threat-model differences between honest-but-curious and malicious servers; TLS and
authenticated transport for a real deployment; TEEs as the low-latency alternative.
