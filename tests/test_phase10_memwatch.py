"""Phase 10 memory-watchdog tests: the pure guard logic runs everywhere; sampling and the abort behavior read `/proc`, so they run on
Linux/WSL2 only. The watchdog protects the 3.8 GiB VM during the XGBoost attempt, so it must FAIL CLOSED: a silently dead
watchdog (the first preflight ran with one -- its log directory did not exist) is the exact failure these tests pin down."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from src.benchmark.memwatch import MemWatch, WatchdogError, evaluate_guard

linux_only = pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="reads /proc (Linux/WSL2)")


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def _alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_evaluate_guard_reports_the_violated_bound_or_none():
    assert evaluate_guard(500, 10, 250, 700) is None
    assert "MemAvailable 200 MB < 250 MB" == evaluate_guard(200, 10, 250, 700)
    assert "swap in use 800 MB > 700 MB" == evaluate_guard(500, 800, 250, 700)
    assert evaluate_guard(1, 1e9, None, None) is None  # an unarmed guard never trips


@linux_only
def test_sampling_records_system_and_process_memory_and_creates_a_nested_log_directory(tmp_path):
    log = tmp_path / "does" / "not" / "exist" / "yet" / "watch.jsonl"  # the exact regression: the parent directories are missing
    with MemWatch(interval_seconds=0.05, log_path=log) as watch:
        watch.watch("self", os.getpid(), kill=False)
        assert _wait_until(lambda: watch.summary()["n_samples"] >= 4)
    summary = watch.summary()
    assert summary["watchdog_error"] is None and summary["aborted"] is False
    assert summary["min_mem_available_mb"] > 0 and summary["per_process"]["self"]["peak_rss_mb"] > 0
    assert len(log.read_text().splitlines()) >= 4


@linux_only
def test_abort_records_evidence_first_and_kills_only_the_registered_kill_targets():
    victim, bystander = _sleeper(), _sleeper()
    seen: dict = {}

    def on_abort(sample, reason):
        seen["victim_alive_at_hook"] = _alive(victim)
        seen["reason"] = reason

    try:
        watch = MemWatch(interval_seconds=0.05, abort_min_mem_available_mb=1e12, on_abort=on_abort)  # impossible bound: trips at once
        watch.watch("victim", victim.pid, kill=True)  # registered BEFORE start, as the runner does
        watch.watch("bystander", bystander.pid, kill=False)
        watch.start()
        assert _wait_until(lambda: victim.poll() is not None)
        summary = watch.stop()
        assert victim.returncode == -signal.SIGKILL and _alive(bystander)
        assert seen["victim_alive_at_hook"] is True and "MemAvailable" in seen["reason"]
        assert summary["aborted"] is True and summary["abort_reason"] == seen["reason"]
    finally:
        for proc in (victim, bystander):
            proc.kill()
            proc.wait()


@linux_only
def test_a_kill_target_registered_after_a_trip_is_killed_at_once():
    late = _sleeper()
    try:
        watch = MemWatch(interval_seconds=0.05, abort_min_mem_available_mb=1e12).start()  # trips on the first sample
        assert _wait_until(lambda: watch.aborted)
        watch.watch("late", late.pid, kill=True)
        assert _wait_until(lambda: late.poll() is not None)
        watch.stop()
    finally:
        late.kill()
        late.wait()


@linux_only
def test_an_unarmed_watch_never_kills_anything():
    victim = _sleeper()
    try:
        with MemWatch(interval_seconds=0.05) as watch:
            watch.watch("victim", victim.pid, kill=True)
            time.sleep(0.4)
        assert _alive(victim) and watch.summary()["aborted"] is False
    finally:
        victim.kill()
        victim.wait()


@linux_only
def test_a_watchdog_that_cannot_start_raises_loudly_and_kills_nothing(tmp_path):
    victim = _sleeper()
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so a directory cannot be created beneath it")
    try:
        watch = MemWatch(interval_seconds=0.05, log_path=blocker / "sub" / "watch.jsonl", abort_min_mem_available_mb=1e12)
        watch.watch("victim", victim.pid, kill=True)
        with pytest.raises(WatchdogError, match="failed to start"):
            watch.start()
        assert _alive(victim)  # nothing risky was started, and nothing was killed
    finally:
        victim.kill()
        victim.wait()


@linux_only
def test_a_watchdog_error_after_a_good_start_trips_an_armed_guard(monkeypatch):
    victim = _sleeper()
    try:
        watch = MemWatch(interval_seconds=0.05, abort_min_mem_available_mb=1.0)  # armed, but real memory is far above 1 MB
        watch.watch("victim", victim.pid, kill=True)
        real = watch.sample_once
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError("simulated /proc read failure")
            return real()

        monkeypatch.setattr(watch, "sample_once", flaky)
        watch.start()
        assert _wait_until(lambda: victim.poll() is not None)
        summary = watch.stop()
        assert summary["aborted"] is True and "watchdog error" in summary["abort_reason"] and "simulated" in summary["watchdog_error"]
    finally:
        victim.kill()
        victim.wait()
