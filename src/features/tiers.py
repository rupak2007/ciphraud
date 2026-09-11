"""Feature-count tiers built from an importance ranking (`docs/plan.md` Phase 4).

Tiers are **versioned artifacts** every later FHE phase references by
config (`docs/architecture.md` Sec.4) -- exact, reproducible membership is
the entire point of this module, not an implementation detail.

Per this session's explicit decision, tiers are **pure top-k** by rank:
`docs/eda.md`'s suggestion to use the 15 V-column null-mask blocks as
sampling/grouping units is honored only as a diagnostic report
(`v_block_coverage`), never as something that changes which columns are
*in* a tier -- `CLAUDE.md` Sec.6 requires tiers to come from the
importance ranking, not from convenience-based block sampling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class TierError(Exception):
    """Raised when tier construction or verification fails."""


def build_tiers(ranking: pd.DataFrame, sizes: list[int]) -> dict[str, list[str]]:
    """Build `{"top_<size>": [feature, ...]}` for each size, in rank order.

    Because every tier is a prefix of the same rank-ordered feature list,
    tiers are nested by construction (top_20 subset of top_50 subset of
    top_100) -- there is no separate nesting check to write or that could
    fail; it follows directly from this implementation.
    """
    if len(set(sizes)) != len(sizes) or list(sizes) != sorted(sizes):
        raise TierError(f"tier sizes must be strictly increasing with no duplicates, got {sizes}")
    if not sizes or sizes[0] <= 0:
        raise TierError(f"tier sizes must be positive, got {sizes}")

    ranked = ranking.sort_values("rank")
    n_nonzero = int((ranked["total_gain"] > 0).sum())
    n_total = len(ranked)
    max_size = sizes[-1]
    if max_size > n_total:
        raise TierError(f"tier size {max_size} exceeds the number of ranked features ({n_total})")
    if max_size > n_nonzero:
        raise TierError(
            f"tier size {max_size} exceeds the number of nonzero-gain features "
            f"({n_nonzero}) -- padding a tier with features the model never split "
            f"on would be arbitrary, not importance-driven"
        )

    ordered_features = ranked["feature"].tolist()
    return {f"top_{size}": ordered_features[:size] for size in sizes}


def tier_membership_hash(tiers: dict[str, list[str]]) -> str:
    """Stable SHA-256 over the tiers' canonical JSON -- the "versioned
    artifact" identity `load_tiers` re-verifies on every read."""
    canonical = json.dumps(tiers, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_tiers(path: Path, tiers: dict[str, list[str]], metadata: dict[str, Any]) -> None:
    """Write the tier artifact: membership + hash + provenance metadata.

    `metadata` carries whatever the caller wants recorded alongside the
    tiers (source model digest, Phase 2 feature-list hash, seed, ...) --
    this module doesn't prescribe its shape, only that `tiers` and
    `membership_hash` are always present and mutually consistent.
    """
    payload = {"tiers": tiers, "membership_hash": tier_membership_hash(tiers), **metadata}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_tiers(path: Path) -> dict[str, Any]:
    """Read a tier artifact, re-verifying its membership hash.

    Every later phase (Phase 5+) must read tiers only through this
    function: a hand-edited or corrupted `tiers.json` fails loudly here
    rather than silently feeding a wrong feature set into FHE compilation.
    """
    payload = json.loads(path.read_text())
    recorded_hash = payload["membership_hash"]
    actual_hash = tier_membership_hash(payload["tiers"])
    if recorded_hash != actual_hash:
        raise TierError(
            f"tiers.json membership_hash mismatch (recorded={recorded_hash}, "
            f"actual={actual_hash}) -- the file may have been hand-edited or corrupted"
        )
    return payload


def _base_column(feature_name: str) -> str:
    """Map a Phase 2 engineered column back to its source raw column.

    `V123_was_missing` -> `V123`; every other name (including `V123`
    itself, `has_identity`, and any `*_freq` column, none of which are V
    columns) passes through unchanged.
    """
    suffix = "_was_missing"
    return feature_name[: -len(suffix)] if feature_name.endswith(suffix) else feature_name


def v_block_coverage(tier_features: list[str], blocks_csv: Path) -> list[dict[str, Any]]:
    """Diagnostic-only: how much of each Phase 1 V-column null-mask block
    (`results/phase1_eda/v_null_mask_blocks.csv`) a tier's columns cover.

    Never changes tier membership (see module docstring) -- this exists so
    a report can flag "this tier drops an entire upstream V-column source
    entirely" as a documented limitation, per `docs/eda.md`'s Phase 4
    implication, without silently forcing block representation into the
    ranking itself.
    """
    blocks = pd.read_csv(blocks_csv)
    tier_bases = {_base_column(f) for f in tier_features}
    coverage = []
    for _, row in blocks.iterrows():
        block_columns = set(str(row["columns"]).split(","))
        n_covered = len(block_columns & tier_bases)
        coverage.append(
            {
                "block_id": int(row["block_id"]),
                "n_columns": int(row["n_columns"]),
                "n_covered": n_covered,
                "coverage_fraction": n_covered / int(row["n_columns"]),
            }
        )
    return coverage
