"""Memory sampling and abort guards for the Phase 10 supervised client/server runs (`docs/client_server.md`).

Linux/WSL2 only for the sampling itself (it reads `/proc`); `evaluate_guard` is pure and runs everywhere. The WSL2 VM has
~3.8 GiB of RAM, and the XGBoost evaluation keys alone are ~1.9 GB, so a failed attempt must be a recorded result, never a
Linux OOM kill of the VM. `MemWatch` samples system and per-process memory on a thread and, when a pre-registered abort
threshold is crossed, SIGKILLs ONLY the process ids it was told to watch (our own client/server), records why, and keeps
sampling. It never touches anything else.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable


class WatchdogError(RuntimeError):
    """The watchdog could not start (e.g. unreadable /proc, unwritable log). Nothing risky has been started when this is raised."""


def read_meminfo() -> dict[str, int]:
    """`/proc/meminfo` as {field: kB}."""
    fields: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            fields[name.strip()] = int(rest.split()[0])
    return fields


def system_memory_mb() -> dict[str, float]:
    info = read_meminfo()
    return {
        "mem_total_mb": info["MemTotal"] / 1024,
        "mem_available_mb": info["MemAvailable"] / 1024,
        "swap_used_mb": (info["SwapTotal"] - info["SwapFree"]) / 1024,
    }


def process_memory_mb(pid: int | str = "self") -> dict[str, float] | None:
    """VmRSS / VmHWM (peak RSS) / VmSwap of one process in MB, or None if it no longer exists."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as handle:
            text = handle.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        for key in ("VmRSS", "VmHWM", "VmSwap"):
            if line.startswith(key + ":"):
                out[key] = int(line.split()[1]) / 1024
    return {"rss_mb": out.get("VmRSS", 0.0), "peak_rss_mb": out.get("VmHWM", 0.0), "swap_mb": out.get("VmSwap", 0.0)}


def evaluate_guard(mem_available_mb: float, swap_used_mb: float, min_mem_available_mb: float | None, max_swap_used_mb: float | None) -> str | None:
    """The reason a guard is violated, or None when the sample is within bounds."""
    if min_mem_available_mb is not None and mem_available_mb < min_mem_available_mb:
        return f"MemAvailable {mem_available_mb:.0f} MB < {min_mem_available_mb:.0f} MB"
    if max_swap_used_mb is not None and swap_used_mb > max_swap_used_mb:
        return f"swap in use {swap_used_mb:.0f} MB > {max_swap_used_mb:.0f} MB"
    return None


class MemWatch:
    """Samples memory every `interval_seconds` until `stop()`. Usable as a context manager.

    `abort_min_mem_available_mb` / `abort_max_swap_used_mb` (either may be None) arm the watchdog: on the first violating
    sample every watched pid is SIGKILLed once and `aborted` / `abort_reason` are set."""

    def __init__(
        self,
        interval_seconds: float = 0.5,
        log_path: Path | None = None,
        abort_min_mem_available_mb: float | None = None,
        abort_max_swap_used_mb: float | None = None,
        on_abort: Callable[[dict[str, Any], str], None] | None = None,
    ) -> None:
        self.interval = interval_seconds
        self.log_path = log_path
        self.abort_min = abort_min_mem_available_mb
        self.abort_swap = abort_max_swap_used_mb
        self.on_abort = on_abort
        self._pids: dict[str, int] = {}
        self._kill: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._n = 0
        self._min_available = float("inf")
        self._max_swap = 0.0
        self._per_pid: dict[str, dict[str, float]] = {}
        self.aborted = False
        self.abort_reason: str | None = None
        self.abort_sample: dict[str, Any] | None = None

    def watch(self, name: str, pid: int, kill: bool = True) -> None:
        """Sample `pid` under `name`; if `kill`, the abort guard SIGKILLs it (the process may be this one)."""
        with self._lock:
            self._pids[name] = int(pid)
            if kill:
                self._kill.add(name)
            else:
                self._kill.discard(name)
        if kill and self.aborted:  # fail closed: a kill-target registered after a trip is killed at once
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def sample_once(self) -> dict[str, Any]:
        system = system_memory_mb()
        sample: dict[str, Any] = {"t": time.time(), **system, "procs": {}}
        with self._lock:
            pids = dict(self._pids)
        for name, pid in pids.items():
            mem = process_memory_mb(pid)
            if mem is None:
                continue
            sample["procs"][name] = mem
            agg = self._per_pid.setdefault(name, {"max_rss_mb": 0.0, "peak_rss_mb": 0.0, "max_swap_mb": 0.0})
            agg["max_rss_mb"] = max(agg["max_rss_mb"], mem["rss_mb"])
            agg["peak_rss_mb"] = max(agg["peak_rss_mb"], mem["peak_rss_mb"])
            agg["max_swap_mb"] = max(agg["max_swap_mb"], mem["swap_mb"])
        self._n += 1
        self._min_available = min(self._min_available, system["mem_available_mb"])
        self._max_swap = max(self._max_swap, system["swap_used_mb"])
        return sample

    @property
    def armed(self) -> bool:
        return self.abort_min is not None or self.abort_swap is not None

    def _trip(self, sample: dict[str, Any], reason: str) -> None:
        """A guard was violated (or the watchdog itself failed while armed): record the evidence, then SIGKILL the kill-targets."""
        self.aborted, self.abort_reason, self.abort_sample = True, reason, sample
        if self.on_abort:
            try:
                self.on_abort(sample, reason)  # record the evidence BEFORE any process (possibly this one) is killed
            except Exception:  # noqa: BLE001 -- the kill below must still happen
                pass
        with self._lock:
            targets = {name: pid for name, pid in self._pids.items() if name in self._kill}
        for pid in targets.values():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _run(self) -> None:
        log = None
        try:
            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                log = open(self.log_path, "a", encoding="utf-8")
            while not self._stop.is_set():
                sample = self.sample_once()
                self._started.set()
                if log:
                    log.write(json.dumps(sample) + "\n")
                    log.flush()
                reason = None if self.aborted else evaluate_guard(sample["mem_available_mb"], sample["swap_used_mb"], self.abort_min, self.abort_swap)
                if reason:
                    self._trip(sample, reason)
                self._stop.wait(self.interval)
        except Exception as exc:  # noqa: BLE001 -- FAIL CLOSED: a dead watchdog must never leave an armed guard silently off
            self.error = f"{type(exc).__name__}: {exc}"
            if self._started.is_set() and self.armed and not self.aborted:
                self._trip({"t": time.time()}, f"watchdog error after start: {self.error}")
            self._started.set()  # never leave start() waiting
        finally:
            if log:
                log.close()

    def start(self) -> "MemWatch":
        """Start sampling and WAIT for the first good sample; raise `WatchdogError` (having started nothing risky) if that fails."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=10) or self.error or self._n == 0:
            self._stop.set()
            raise WatchdogError(f"the memory watchdog failed to start: {self.error or 'no sample within 10 s'}")
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5 * self.interval + 5)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": self._n,
            "min_mem_available_mb": None if self._n == 0 else self._min_available,
            "max_swap_used_mb": self._max_swap,
            "per_process": {name: dict(agg) for name, agg in self._per_pid.items()},
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "watchdog_error": self.error,
            "guards": {"abort_min_mem_available_mb": self.abort_min, "abort_max_swap_used_mb": self.abort_swap},
        }

    def __enter__(self) -> "MemWatch":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
