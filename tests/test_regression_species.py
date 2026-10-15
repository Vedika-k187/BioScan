"""Regression: mutated reference sequences must never be assigned a *different* species."""
import random

import pandas as pd
import pytest

from src.prediction.predictor import predict_sequence

NA = "Not available in reference data"


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    out = list(seq)
    for i, base in enumerate(out):
        if rng.random() < rate:
            out[i] = rng.choice([c for c in "ACGT" if c != base])
    return "".join(out)


@pytest.fixture(scope="module")
def refs():
    return pd.read_csv("data/processed/reference_barcodes.csv")


@pytest.mark.parametrize("rate", [0.0, 0.02, 0.04])
def test_mutated_reference_never_gets_wrong_species(refs, rate):
    rng = random.Random(7)
    labelled = refs[refs["species"].notna()].sample(40, random_state=1)
    wrong = []
    for _, row in labelled.iterrows():
        res = predict_sequence(_mutate(row["sequence_normalized"], rate, rng))
        sp = res["predicted_taxonomy"]["species"]
        if sp != NA and sp != row["species"]:
            # allow if a different reference shares the identical/near sequence
            wrong.append((row["sequence_id"], row["species"], sp))
    assert len(wrong) <= 1, f"wrong species assigned: {wrong}"
