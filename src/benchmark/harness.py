"""Phase 7 benchmark harness -- repeated-trial timing/statistics (`docs/plan.md` Phase 7).

This module holds only what Phase 5/6 explicitly deferred as out of scope
("compile/runtime numbers... observations, not benchmarks -- Phase 7 owns
repeated-trial measurement", `docs/fhe_xgboost.md`): running the SAME kind
of plaintext-predict and real encrypt->run->decrypt call Phase 5/6 already
proved correct, N times on a fixed request, to get a real mean/standard
deviation instead of a single anecdotal number.

Deliberately generic and Concrete-ML-import-free at the top level: every
function here takes a plain callable or plain arrays, so the statistics
math (`trial_stats`) is unit-testable on any platform, and only the actual
`circuit.encrypt`/`run`/`decrypt` calls a caller passes in need WSL2 +
Concrete-ML. Nothing here reads a config or a handoff file -- that is
`src/benchmark/run.py`'s job, which reuses (never modifies) the existing
`src/fhe/handoff.py` / `src/fhe/compile/*.py` / `src/data/provenance.py`
infrastructure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def time_repeated(fn: Callable[[], Any], n_trials: int) -> tuple[list[float], list[Any]]:
    """Call `fn()` `n_trials` times, timing each call. No warmup trial is
    discarded: the model/circuit is already built once outside this loop
    (compile time is measured and reported separately, per
    `docs/architecture.md` Sec.16), so every call here is a genuine
    per-request trial. Returns (seconds per trial, that trial's return
    value) so callers can both compute latency statistics and verify the
    outputs were reproducible across trials."""
    seconds: list[float] = []
    outputs: list[Any] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        outputs.append(fn())
        seconds.append(time.perf_counter() - t0)
    return seconds, outputs


def trial_stats(seconds: list[float]) -> dict[str, Any]:
    """Mean/standard deviation/min/max over repeated trials
    (`docs/instructions.md` Benchmarking Rules: "report mean and standard
    deviation, never a single-shot number"). `ddof=1` (sample std) since
    these trials are a sample, not the full population of possible runs;
    with a single trial, std is reported as 0.0 rather than NaN, since a
    single-trial "spread" is not meaningfully undefined for reporting
    purposes -- `n_trials` is always recorded alongside it so a reader
    can see when std is degenerate."""
    arr = np.asarray(seconds, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("trial_stats requires at least one trial")
    return {
        "n_trials": int(arr.size),
        "mean_seconds": float(arr.mean()),
        "std_seconds": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "trials_seconds": [float(x) for x in arr],
    }


def fhe_round_trip_trials(
    circuit: Any, q_row: np.ndarray, n_trials: int,
    checkpoint_dir: Path | None = None, fingerprint: str | None = None,
) -> dict[str, Any]:
    """5 REPEATED EXECUTIONS OF A FIXED REPRESENTATIVE INPUT -- real
    `encrypt -> run -> decrypt`, on the SAME already-quantized row every
    time, using ONE `circuit.keygen()` amortized across all trials
    (matching Phase 5/6's own `_explicit_round_trip`/`explicit_round_trip`
    pattern: keygen is a once-per-client-session cost, not a per-request
    one, so it is measured and reported separately, never folded into the
    per-request latency statistics). These are repeated executions of ONE
    input, not independent samples across different inputs -- see the
    caller's `row_selection` record for exactly which row/seed was used.

    `checkpoint_dir`/`fingerprint`, when given, persist each trial's
    decrypted result + per-step timing to disk immediately after it
    completes (mirroring `src/fhe/xgb_poc.py::explicit_round_trip`'s exact
    pattern) so a run interrupted mid-grid resumes at the next un-executed
    trial rather than restarting -- essential at Phase 8's scale (a single
    XGBoost trial can cost 30-50 minutes; docs/research.md).

    Every trial's decrypted integer output is compared for exact,
    bit-for-bit reproducibility (a deterministic circuit on the same
    input must return the same output every time); a mismatch is recorded
    as `outputs_reproducible: False` rather than raised, since this
    module measures latency, not correctness -- Phase 5/6 already own
    correctness (T0-T3) and are not re-validated here.
    """
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        meta_path = checkpoint_dir / "meta.json"
        if meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            for stale in checkpoint_dir.glob("trial_*"):
                stale.unlink()
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "n_trials": n_trials}))

    t0 = time.perf_counter()
    circuit.keygen()
    keygen_seconds = time.perf_counter() - t0

    encrypt_s, run_s, decrypt_s, total_s, decrypted_outputs = [], [], [], [], []
    for i in range(n_trials):
        result_path = checkpoint_dir / f"trial_{i:03d}.npy" if checkpoint_dir else None
        timing_path = checkpoint_dir / f"trial_{i:03d}_timing.json" if checkpoint_dir else None

        if result_path is not None and result_path.exists():
            decrypted = np.load(result_path, allow_pickle=False)
            if not np.array_equal(decrypted, np.asarray(circuit.simulate(q_row))):
                raise RuntimeError(f"checkpoint {result_path} disagrees with a fresh simulation of this row")
            timing = json.loads(timing_path.read_text())
            encrypt_s.append(timing["encrypt_seconds"])
            run_s.append(timing["run_seconds"])
            decrypt_s.append(timing["decrypt_seconds"])
            total_s.append(timing["total_seconds"])
            decrypted_outputs.append(decrypted)
            continue

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        encrypted = circuit.encrypt(q_row)
        e_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        ran = circuit.run(encrypted)
        r_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        decrypted = np.asarray(circuit.decrypt(ran))
        d_s = time.perf_counter() - t0

        t_s = time.perf_counter() - t_total
        encrypt_s.append(e_s)
        run_s.append(r_s)
        decrypt_s.append(d_s)
        total_s.append(t_s)
        decrypted_outputs.append(decrypted)

        if result_path is not None:
            np.save(result_path, decrypted)
            timing_path.write_text(json.dumps({"encrypt_seconds": e_s, "run_seconds": r_s, "decrypt_seconds": d_s, "total_seconds": t_s}))

    outputs_reproducible = all(np.array_equal(decrypted_outputs[0], o) for o in decrypted_outputs[1:])

    return {
        "keygen_seconds": keygen_seconds,
        "total": trial_stats(total_s),
        "encrypt": trial_stats(encrypt_s),
        "run": trial_stats(run_s),
        "decrypt": trial_stats(decrypt_s),
        "outputs_reproducible": bool(outputs_reproducible),
        "throughput_requests_per_second": float(1.0 / (sum(total_s) / len(total_s))),
    }


def plaintext_latency_trials(predict_fn: Callable[[], Any], n_trials: int) -> dict[str, Any]:
    """Repeated plaintext (non-FHE) prediction trials on the same fixed
    request -- the baseline `docs/architecture.md` Sec.16 asks the runner
    to benchmark alongside the FHE path, so the eventual Pareto-frontier
    analysis (Phase 8) has both ends of the accuracy/cost trade-off from
    the same harness and the same request."""
    seconds, outputs = time_repeated(predict_fn, n_trials)
    stats = trial_stats(seconds)
    stats["outputs_reproducible"] = bool(all(np.array_equal(outputs[0], o) for o in outputs[1:]))
    stats["throughput_requests_per_second"] = float(1.0 / stats["mean_seconds"]) if stats["mean_seconds"] > 0 else float("inf")
    return stats
