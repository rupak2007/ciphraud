# Finalization audit — pre-commit (2026-10-12 14:50 IST)

Read-only audit plus documentation fixes. No commit, push, test scoring, Docker rebuild, deletion or change to the freeze was made. Status labels: PASS, PARTIAL, BLOCKED, NOT RUN, NEEDS APPROVAL.
This file is new and does not replace any earlier audit (`Ciphraud_Project_Audit.*`).

## 1. Verified now (not copied from the earlier report)

| Item | Result |
|---|---|
| Branch / HEAD | `main` at `7ae3b82`, `origin/main` behind by 4 (4 local commits unpushed: `4945dea`, `41894df`, `7e750d5`, `7ae3b82`); nothing staged |
| `freeze.json` sha256 (complete) | `3a192a945f62e937f7230191f7a2d86d609465414a156f1ebc7846c2d24a2ced`, frozen 2026-10-12T05:23:10Z, git head recorded `7ae3b82`, `labels_opened: false` |
| Freeze contents | 18 records, all validation-reproduction `equal: true`; 12 code-file hashes, evaluation config hash, phase 8/9 grid config hashes, 3 feature-array hashes, 2 small pinned JSON hashes all match the live files; the 18 `freeze_partial` checkpoints are identical to the freeze records; superseded folders v1 (6 files) and v2 (18 files) preserved |
| Run artifacts | none: no `run_started.json`, `run_complete.json`, `test_results.json`, `summary.md`; no evaluation process running; only `stage_run` parses the label file, after the marker |
| Sealed files | `data/processed/test.pkl` and `y_test.npy` were deliberately **not** read (hashed) in this audit; metadata only |
| `report.md` | byte-identical to its HEAD blob (27,959 bytes, LF both sides, mtime = commit time 2026-10-11 23:30:53); the "changed on disk" flag was the earlier edit before that commit |
| Phase 1-10 | `git diff HEAD` over `results/`, phase 1-10 configs, `src/`, Phase 10 tests, `report.md` and the Phase 8/9/10 docs: empty |
| Docker | engine stopped, no Docker processes; image `ciphraud-server:phase11` (3,326,746,012 bytes) built earlier; not rebuilt |
| OpenSky | running (see section 4) |
| Resources | free Windows commit about 9.8 GB, free physical RAM about 0.2 GB (OpenSky training holds about 5.9 GB) |

## 2. Could not be verified / discrepancies

1. **Label file access time.** `y_test.npy` last-access 2026-10-12 12:02:37 (modified 00:18:57 at export, size unchanged 837,224 bytes). `guard_v2_state.log` was accessed in the same second and no other file under `data/phase12` was. No Ciphraud code that parses labels ran and no result exists, so there is no evidence of scoring or tuning; but the reader is **unidentified** (antivirus, indexer or another session are possible, none shown). I did not hash the file to settle integrity because that would open the sealed bytes; `--run` verifies the sha256 before use and fails closed. **Needs your judgment** on whether "labels unopened" may be claimed without this qualification. Recorded in `docs/test_evaluation_protocol.md` section 11.
2. The earlier report's `tail -F` processes on old gate logs belong to another session (parent processes not mine).
3. Code hashes (freeze and Phase 10) were computed on LF files and will not match on a CRLF checkout (`core.autocrlf=true` on this machine; git stores LF).
4. `docs/docker.md` contained sections written before the build that contradicted the later results (e.g. "Live container test: NOT RUN"); they are now labelled HISTORICAL/SUPERSEDED, nothing deleted. A stale "DRAFT: not yet built" comment in the `Dockerfile` was corrected (comment only).
5. Two tests (`inspect.getsource`-based) failed on Windows because a WSL-compiled `.pyc` carries `/mnt/c/...` paths; they now fall back to parsing the file (assertions unchanged), and one Concrete-ML test skips where the stack is absent.

## 3. Tests (this audit) — commands, exit codes, counts

| Command (env) | Exit | Result |
|---|---|---|
| `.venv/Scripts/python.exe -m pytest tests/test_phase12_test_eval.py tests/test_phase11_docker.py -q -p no:cacheprovider` (Windows .venv, Python 3.12.0, after the fixes) | 0 | **24 passed, 2 skipped** (skips: MLP-adapter test needs Concrete-ML; live container test needs `CIPHRAUD_DOCKER_TEST=1` and Docker) |
| same files before the fix | non-zero | 5 failed (4 `OSError: could not get source code`, 1 `No module named concrete`), 20 passed, 1 skipped; cause above, not a code defect |
| `... pytest tests/test_phase10_results.py` (Windows) | 0 | 39 passed |
| `python -m src.analysis.phase10_results --check-code-hashes` (Windows) | 0 | OK, no problems |
| `python -m src.analysis.report_check` (Windows) | 0 | OK, all quoted numbers found |
| freeze integrity script (Python, no sealed-byte reads) | 0 | all checks PASS |

**Previously passed, not rerun now:** WSL Phase 12 + Phase 11 files 25 passed/1 skipped (before the last test edits); WSL live container test 8 passed in 32.6 s (`data/phase11/logs/pytest_live_manual2.log`); WSL Phase 10 group 100 passed (before the Phase 10 commit); the WSL-only MLP-adapter test therefore has no run since its Windows-skip edit.
**Not run:** full suites; any MLP or XGBoost container test; any test-set scoring; any Docker rebuild. A skipped test is not counted as passed.
Full logs: `data/phase11/logs/` and `data/phase12/logs/` (gitignored, 335 KB in total).

## 4. OpenSky (read-only)

Queue process tree at 14:46: `queue_week3.py` (pid 18248 -> 11076) started 14:44:25 -> `opensky.train.train` (pids 9736 -> 11888, about 5.9 GB commit) for **gru_seed2** (`data/queue/gru_seed2.log`, created 14:44). `state.json` lists `mlp_seed1`, `mlp_seed2`, `gru_seed0`, `transformer_seed0`, `gru_seed1` as done; `transformer_seed0` is marked done with the note "PARTIAL … stopped at epoch 31/40, best.pt = epoch-30 weights, no optimizer state", so **the transformer was not resumed from epoch 31**; the queue moved on to other jobs (gru_seed1 finished 13:03). The resume watcher launched the queue at 12:12:05; no active `RESUME_BLOCKED.txt` (four superseded copies kept). Nothing was stopped, restarted or modified.
Trade-off: the frozen `--run` stage needs about 3.5 GB peak (rebuilds all 18 models, about 35-40 min) and free physical RAM is about 0.2 GB while OpenSky trains; running it concurrently risks paging both jobs. Recommend running it after the current OpenSky job finishes or at a moment you choose. Not decided here.

## 5. Capability matrix

| Capability | Evidence | Status |
|---|---|---|
| Research freeze integrity | `freeze.json` + checks in section 1; labels-accessed observation in section 2 | PASS (with the open observation) |
| Evaluation tooling readiness | 18 Phase 12 tests (WSL) / 17 passed + 1 skipped (Windows); `--run` gated by freeze, hash and marker checks | PASS |
| Sealed test evaluation | not run | NEEDS APPROVAL |
| Docker image build | `docker build` exit 0, 560 s, 3.33 GB, CPU-only torch, no nvidia packages, non-root | PASS |
| Live LR container inference | 8/8 incl. one encrypted round trip with exact integer agreement (single request, n=1) | PASS |
| Live MLP container inference | **updated 15:09:** 2 rows exact vs recorded Phase 9 integers in the container, wire audit clean, keys deleted, no key material in logs (`docs/docker.md` Container validation 2) | PASS (local loopback, n=2) |
| Live XGBoost container inference | not tested (about 5 GB VM memory, about 45 min); separately gated | NOT RUN |
| Failure handling (malformed, wrong/incompatible keys, restart) | **updated 15:09:** 4 new live tests, exact documented status codes, 12/12 Docker tests passed | PASS |
| API/container security (local demo) | non-root, read-only root fs, only `server.zip` + manifest mounted read-only, loopback publish, no secrets in logs | PASS for a local demo |
| TLS / authentication | not implemented | NOT RUN (BLOCKED for any remote use) |
| Reproducibility / automated tests | pinned requirements, validators OK; hash checks assume LF line endings | PASS (with the CRLF caveat) |
| Dataset compliance (`split_assignments.csv`) | file is on the public `origin/main` since Phase 2; Kaggle rules not yet read | NEEDS APPROVAL |
| License | none chosen; Zama BSD-3-Clause-Clear note; see `docs/license_recommendation.md` | NEEDS APPROVAL |
| Documentation | README, docker, protocol, remediation, license docs updated; stale statements labelled | PASS |
| Git readiness | three groups separable (section 6); nothing committed | PARTIAL (awaiting your approval) |

**Prototype vs production:** this is a research prototype with a local loopback demo of encrypted inference for LR (and verified exact encrypted/clear agreement for LR, MLP, XGBoost in Phase 10). It is not production-ready: no TLS or authentication, container evidence is two MLP requests and one LR request (no load or concurrency testing; XGBoost untested in a container), with the documented error responses and a restart tested, and the sealed test evaluation and dataset-terms question are unresolved.

## 6. Proposed commits (nothing created)

**Commit 1 — Phase 12 evaluation tooling and freeze evidence** (new files only)
`configs/phase12/test_eval.yaml`, `src/analysis/test_eval_export.py`, `src/analysis/test_evaluation.py`, `tests/test_phase12_test_eval.py`, `docs/test_evaluation_protocol.md`, `results/phase12_test_eval/freeze.json`, `results/phase12_test_eval/freeze_partial/` (18), `freeze_partial_superseded_code_v1/` (6), `freeze_partial_superseded_code_v2/` (18).
Message: `feat(eval): add frozen test-evaluation tooling and the verified 18-model freeze` — body: tooling gated by freeze/hash/marker; freeze sha256; validation reproduced within 1e-12; test labels not scored; superseded checkpoint folders kept; the XGBoost chunking and threshold-tolerance fixes. Trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
Concerns: the label-access observation (section 2); hashes assume LF; failure and guard logs stay local (see option below).

**Commit 2 — Docker server image definition** (6 files + 1 new)
`Dockerfile`, `.dockerignore`, `docs/docker.md`, `tests/test_phase11_docker.py`, `docs/architecture.md`, `requirements-server-cpu.txt` (new).
Message: `feat(docker): CPU-only server image, source-built onnxoptimizer, exec tmpfs; LR live test passed` — body: the three fixes found by real builds/tests; LR-only evidence; MLP/XGBoost, TLS, auth not tested. Same trailer.
Concerns: image was built from the Dockerfile before a comment-only edit; `docs/docker.md` keeps HISTORICAL sections on purpose.

**Commit 3 — documentation and compliance** (1 modified + 3 new)
`README.md`, `docs/license_recommendation.md`, `docs/dataset_remediation_plan.md`, `docs/finalization_audit.md` (this file).
Message: `docs: finalization audit, dataset-remediation plan, license recommendation, README Docker status`. Same trailer. Commit after commit 2 (the README refers to `docs/docker.md`).
Concerns: contains the public repository URL (not a secret); the compliance questions are explicitly open, not resolved.

Checks on all three: no secrets, no key/ciphertext/cache/data files, no machine-specific paths other than the public GitHub name in two docs; 52 untracked files in total (including this audit), about 290 KB.
**Optional (needs your decision):** copy the small failure/guard logs (`data/phase12/logs`, 78 KB, and `data/phase11/logs`, 257 KB) into the repository (e.g. `results/phase12_test_eval/attempt_logs/`, `results/phase11_docker/logs/`) so "every failure log preserved" holds inside the repo and not only in the gitignored `data/`.

## 7. Decisions that need you

1. Approve the three commits (and whether to include the optional log copies).
2. Decide how to treat the unexplained label-file access before the sealed `--run`; and when to run `--run` relative to OpenSky.
3. Read the Kaggle IEEE-CIS rules and decide on `split_assignments.csv` (untrack, and whether to purge history).
4. Choose a license.
5. Whether MLP/XGBoost container tests and any TLS/authentication work are in scope.

## 8. Update 2026-10-12 15:09
MLP container validation and failure-handling tests were added and passed (see `docs/docker.md`); sections 1-7 above are otherwise as written at 14:50. `tests/test_phase11_docker.py` therefore changed after the Commit 2 description above (it now holds 12 tests). Freeze hash re-verified unchanged: `3a192a94…2ced`.

## 9. Final completion status (2026-10-12, after the master completion directive)
Verified facts are separate from open questions. Status labels: PASS, PARTIAL, BLOCKED, NOT RUN, NEEDS APPROVAL.

| Item | Status | Evidence |
|---|---|---|
| OpenSky paused, auto-restart controlled | PASS | `data/queue/PAUSED` written in the OpenSky repo; its queue launches no further step; no training or queue process was running afterwards; the idle-watcher had already exited. Resume = delete `OpenSky/data/queue/PAUSED` and run `python scripts/queue_week3.py` (resumable; finished steps are skipped). `gru_seed2` finished before the pause (`state.json`). |
| Phase 12 freeze integrity | PASS | `freeze.json` sha256 `3a192a94...2ced` unchanged before and after the run; the run's own guards re-verified code, config, pinned-input and export hashes and all 18 models before opening labels. |
| Sealed-label integrity concern | PARTIAL | access-time changes traced to metadata probes (identical timestamp on label and feature files, equal to `date`), see protocol section 11; no OS audit log exists, so not proven; no credible evidence of scoring or tuning. |
| Final frozen test evaluation | PASS (run once) | exit 0; `docs/test_evaluation_protocol.md` section 12; results `results/phase12_test_eval/{test_results.json,summary.md}`; every configuration scored below its validation PR-AUC. |
| LR container inference | PASS | live 12/12 re-run with exact assertions (`pytest_live_final_tightened_assertions.log`) and earlier 8/8 (`pytest_live_manual2.log`). |
| MLP container inference | PASS (n = 2 requests; pipeline demo only, `mlp_top20_bits3` failed the Phase 9 quantization-accuracy gate) | `pytest_live_mlp_failure_tests.log`, `mlp_container_result.json`. |
| XGBoost container inference | BLOCKED (hardware) | needs about 5 GB of the shared WSL VM plus about 45 min; the 8 GB PC with a 5 GB VM limit (not changed, no authorization) shares that VM with Docker Desktop; a machine with roughly 16 GB RAM is the practical minimum. Not claimed. |
| API error handling, restart/recovery | PASS | exact status and error codes asserted; restart test in the same 12/12 run. |
| TLS / authentication | NOT RUN (out of scope; required for any remote use) | no code added; loopback research demo only. |
| Hash portability | PASS | `.gitattributes` pins the hashed files to the exact recorded bytes (LF; CRLF for the two Windows-authored JSON pins); no frozen byte changed. |
| Dataset compliance | NEEDS APPROVAL | Kaggle Rules section 7 read (non-commercial; no redistribution to non-participants); `split_assignments.csv` is a verbatim two-column extract and is already public; `docs/dataset_remediation_plan.md`. |
| License | NEEDS APPROVAL | `docs/license_recommendation.md`: Apache-2.0 if reuse is expected, MIT otherwise; none added. |
| Tests | see logs | Windows 24 passed / 6 skipped (`pytest_windows_static_post_run.log`); WSL Phase 11+12 files 25 passed / 5 skipped (`data/phase12/logs/pytest_wsl_phase12_post_run.log`); live Docker 12 passed (log above). Skips are live-Docker or Concrete-ML-only tests, not counted as passes. |
| Pushed to remote | NOT RUN | nothing pushed. |
