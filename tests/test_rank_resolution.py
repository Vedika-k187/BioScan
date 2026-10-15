"""Rank-resolution helper: explanatory only, must never invent a resolved rank."""
from app.components.rank_resolution import deepest_concordant_rank


def test_resolved_to_deepest_contiguous_concordant_rank():
    c = {"class_concordant": True, "order_concordant": True, "family_concordant": False,
         "genus_concordant": True, "species_concordant": None}
    # genus agreeing does not count once family disagrees
    assert deepest_concordant_rank(c) == "order"


def test_no_rank_when_class_disagrees():
    c = {"class_concordant": False, "order_concordant": True, "family_concordant": True}
    assert deepest_concordant_rank(c) is None


def test_missing_or_none_concordance_resolves_nothing():
    assert deepest_concordant_rank({}) is None
    assert deepest_concordant_rank({"class_concordant": None}) is None


def test_all_ranks_concordant():
    c = {f"{r}_concordant": True for r in ["class", "order", "family", "genus", "species"]}
    assert deepest_concordant_rank(c) == "species"


def test_resolvable_ranks_cap_excludes_genus_and_species():
    from app.components.rank_resolution import RESOLVABLE_RANKS
    c = {f"{r}_concordant": True for r in ["class", "order", "family", "genus", "species"]}
    assert deepest_concordant_rank(c, RESOLVABLE_RANKS) == "family"
    assert "genus" not in RESOLVABLE_RANKS and "species" not in RESOLVABLE_RANKS
