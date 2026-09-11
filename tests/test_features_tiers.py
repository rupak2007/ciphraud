import json

import pandas as pd
import pytest

from src.features.tiers import (
    TierError,
    build_tiers,
    load_tiers,
    save_tiers,
    tier_membership_hash,
    v_block_coverage,
)


def _ranking(n=10) -> pd.DataFrame:
    """6 nonzero-gain features (ranks 1-6), 4 zero-gain (ranks 7-10)."""
    rows = [{"feature": f"f{i}", "total_gain": float(100 - i), "gain": 1.0, "weight": 1.0} for i in range(6)]
    rows += [{"feature": f"f{i}", "total_gain": 0.0, "gain": 0.0, "weight": 0.0} for i in range(6, n)]
    df = pd.DataFrame(rows)
    df["rank"] = range(1, len(df) + 1)
    return df


def test_build_tiers_are_nested_prefixes():
    tiers = build_tiers(_ranking(), [2, 4, 6])
    assert tiers["top_2"] == tiers["top_4"][:2]
    assert tiers["top_4"] == tiers["top_6"][:4]


def test_build_tiers_raises_on_non_increasing_sizes():
    with pytest.raises(TierError):
        build_tiers(_ranking(), [4, 2])


def test_build_tiers_raises_on_duplicate_sizes():
    with pytest.raises(TierError):
        build_tiers(_ranking(), [4, 4])


def test_build_tiers_raises_when_size_exceeds_nonzero_gain_features():
    with pytest.raises(TierError):
        build_tiers(_ranking(), [2, 8])  # only 6 nonzero-gain features


def test_build_tiers_raises_when_size_exceeds_total_features():
    with pytest.raises(TierError):
        build_tiers(_ranking(n=10), [20])


def test_build_tiers_is_deterministic_regardless_of_row_order():
    ranking = _ranking()
    shuffled = ranking.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert build_tiers(ranking, [4]) == build_tiers(shuffled, [4])


def test_tier_membership_hash_stable_and_sensitive_to_content():
    tiers_a = {"top_2": ["f0", "f1"]}
    tiers_b = {"top_2": ["f0", "f1"]}
    tiers_c = {"top_2": ["f0", "f2"]}
    assert tier_membership_hash(tiers_a) == tier_membership_hash(tiers_b)
    assert tier_membership_hash(tiers_a) != tier_membership_hash(tiers_c)


def test_save_and_load_tiers_round_trip(tmp_path):
    tiers = {"top_2": ["f0", "f1"]}
    path = tmp_path / "tiers.json"
    save_tiers(path, tiers, metadata={"seed": 42})
    loaded = load_tiers(path)
    assert loaded["tiers"] == tiers
    assert loaded["seed"] == 42


def test_load_tiers_raises_on_tampered_file(tmp_path):
    tiers = {"top_2": ["f0", "f1"]}
    path = tmp_path / "tiers.json"
    save_tiers(path, tiers, metadata={})
    payload = json.loads(path.read_text())
    payload["tiers"]["top_2"].append("f99")  # tamper without recomputing the hash
    path.write_text(json.dumps(payload))
    with pytest.raises(TierError):
        load_tiers(path)


def test_v_block_coverage_maps_was_missing_suffix_to_base_column(tmp_path):
    blocks_csv = tmp_path / "blocks.csv"
    blocks_csv.write_text('block_id,null_mask_sha256,n_columns,columns\n0,abc,3,"V1,V2,V3"\n')
    coverage = v_block_coverage(["V1_was_missing", "V2", "other_col"], blocks_csv)
    assert coverage[0]["n_covered"] == 2
    assert coverage[0]["coverage_fraction"] == pytest.approx(2 / 3)


def test_v_block_coverage_zero_when_no_overlap(tmp_path):
    blocks_csv = tmp_path / "blocks.csv"
    blocks_csv.write_text('block_id,null_mask_sha256,n_columns,columns\n0,abc,3,"V1,V2,V3"\n')
    coverage = v_block_coverage(["unrelated_col"], blocks_csv)
    assert coverage[0]["n_covered"] == 0
