"""Provenance stamping for generated artifacts.

Every reported number must trace back to a config, a code revision, and a
timestamp (CLAUDE.md Sec.9/Sec.10). This is deliberately generic (not
EDA-specific) so Phase 7/8 benchmark runs can reuse the same helper.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def get_git_commit(cwd: Path | None = None) -> str:
    """Return the current git commit hash, or "unknown" if unavailable.

    Never raises: a missing git binary or a non-repo cwd must not crash a
    pipeline stage, but the fallback value makes the gap visible rather
    than silently omitting provenance.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def get_library_versions(package_names: list[str]) -> dict[str, str]:
    """Look up installed versions for a list of distribution names."""
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def compute_config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config dict (sorted-key canonical JSON)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_provenance(
    *,
    config: dict[str, Any],
    config_path: Path,
    seed: int,
    raw_file_digests: dict[str, str],
    library_names: list[str],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "config_path": str(config_path),
        "config_hash": compute_config_hash(config),
        "seed": seed,
        "library_versions": get_library_versions(library_names),
        "raw_file_sha256": raw_file_digests,
    }
