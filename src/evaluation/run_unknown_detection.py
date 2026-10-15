"""
Unknown-detection experiment (real held-out data, no synthetic mutation).

A. Leave-one-genus-out (LOGO): remove every reference of a genus from the library,
   then query those sequences. Their true genus is absent, so a correct system
   must NOT report a confident (close-match) identification.
B. Negative controls: real non-fish vertebrate 12S sequences from NCBI
   (data/external/negative_controls_12S.fasta), windowed to amplicon length.
C. Positive control: known references with 2% substitutions (should stay known).

Status uses the same distance thresholds as the predictor:
    d <= 0.05 -> known_reference_like ; 0.05 < d <= 0.15 -> low_confidence ; d > 0.15 -> out_of_reference
"""
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)
from src.prediction.evidence import THRESHOLD_CLOSE_SIMILARITY as CLOSE, THRESHOLD_OUT_OF_REFERENCE as OOR
from src.similarity.reference_search import normalized_edit_distance

REF = pd.read_csv(os.path.join(BASE, "data/processed/reference_barcodes.csv"))
FASTA = os.path.join(BASE, "data/external/negative_controls_12S.fasta")
AMPLICON = 106


def status(d):
    return "known_reference_like" if d <= CLOSE else ("low_confidence" if d <= OOR else "out_of_reference")


def nearest(q, ref_df):
    return min(normalized_edit_distance(q, s) for s in ref_df["sequence_normalized"])


def read_fasta(path):
    recs, name, seq = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(seq)))
            name, seq = line[1:].strip(), []
        else:
            seq.append(line.strip().upper())
    if name:
        recs.append((name, "".join(seq)))
    return recs


def main():
    rows = []
    # A. leave-one-genus-out
    for genus, grp in REF[REF["genus"].notna()].groupby("genus"):
        lib = REF[REF["genus"] != genus]
        for _, r in grp.iterrows():
            d = nearest(r["sequence_normalized"], lib)
            rows.append(dict(experiment="held-out genus", id=r["sequence_id"], label=genus, distance=d, status=status(d)))
    # B. negative controls
    skip = ("cytochrome", "NADH", "D-loop", "control region")
    for name, seq in read_fasta(FASTA):
        if any(k.lower() in name.lower() for k in skip) or "12S" not in name:
            continue
        for off in (0, 120, 240):
            w = seq[off:off + AMPLICON]
            if len(w) == AMPLICON and set(w) <= set("ACGT"):
                d = nearest(w, REF)
                rows.append(dict(experiment="non-fish negative", id=f"{name.split()[0]}@{off}",
                                 label=" ".join(name.split()[1:3]), distance=d, status=status(d)))
    # C. positive control
    rng = random.Random(11)
    for _, r in REF.sample(60, random_state=3).iterrows():
        s = list(r["sequence_normalized"])
        for i in range(len(s)):
            if rng.random() < 0.02:
                s[i] = rng.choice([c for c in "ACGT" if c != s[i]])
        d = nearest("".join(s), REF)
        rows.append(dict(experiment="known + 2% noise", id=r["sequence_id"], label=r["species"], distance=d, status=status(d)))

    df = pd.DataFrame(rows)
    out = os.path.join(BASE, "reports")
    df.to_csv(os.path.join(out, "unknown_detection_experiment.csv"), index=False)

    summ = (df.groupby("experiment")
              .agg(n=("id", "count"),
                   median_distance=("distance", "median"),
                   pct_flagged_unknown=("status", lambda s: round(100 * (s != "known_reference_like").mean(), 1)),
                   pct_out_of_reference=("status", lambda s: round(100 * (s == "out_of_reference").mean(), 1)))
              .round(3).reset_index())
    summ.to_csv(os.path.join(out, "unknown_detection_summary.csv"), index=False)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"known + 2% noise": "#16a34a", "held-out genus": "#f59e0b", "non-fish negative": "#dc2626"}
    for exp, g in df.groupby("experiment"):
        ax.hist(g["distance"], bins=20, range=(0, 0.7), alpha=0.65, label=f"{exp} (n={len(g)})", color=colors[exp])
    ax.axvline(CLOSE, ls="--", c="grey"); ax.axvline(OOR, ls="--", c="black")
    ax.text(CLOSE + .005, ax.get_ylim()[1] * .9, f"{CLOSE:.2f} close", fontsize=8)
    ax.text(OOR + .005, ax.get_ylim()[1] * .8, f"{OOR:.2f} veto", fontsize=8)
    ax.set_xlabel("Normalised edit distance to nearest reference"); ax.set_ylabel("Sequences")
    ax.set_title("Distance veto: known vs. unknown sequences"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "unknown_detection_distance.png"), dpi=150)


if __name__ == "__main__":
    main()
