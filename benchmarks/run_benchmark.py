"""Run the confbci benchmark on a MOABB dataset and save tidy results.

    python benchmarks/run_benchmark.py --dataset Liu2024 --alpha 0.2
    python benchmarks/run_benchmark.py --dataset BNCI2014_001 --score lac

Writes to ``results/``:
    <dataset>_records.csv     one row per (regime, method, subject[, session, k, order, gamma])
    <dataset>_summary.csv     aggregated over subjects
    <dataset>_screening.csv   per-patient screening inputs
    <dataset>_screening.json  Spearman rho / AUROC with bootstrap intervals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from confbci import LAC, LOSOConformalEvaluation, RiemannianMargin, screening_metrics, summarize
from confbci.data import load_covariances

ARRAY_COLUMNS = ("alpha_trajectory", "rolling_coverage", "covered")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Liu2024")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--score", choices=("lac", "margin"), default="margin")
    p.add_argument("--block-size", type=int, default=20, help="screening block (trials)")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, y, groups, sessions = load_covariances(args.dataset)
    counts = np.unique(groups, return_counts=True)[1]
    print(f"{args.dataset}: {len(counts)} subjects, {X.shape[-1]} channels "
          f"(tangent dimension {X.shape[-1] * (X.shape[-1] + 1) // 2}), "
          f"{counts.min()}-{counts.max()} trials per subject")

    score = LAC() if args.score == "lac" else RiemannianMargin()
    ev = LOSOConformalEvaluation(score=score, alpha=args.alpha, random_state=args.seed)
    records = ev.run(X, y, groups, sessions=sessions, regimes=("R1", "R2", "R3"))
    for r in records:
        r["dataset"] = args.dataset

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / args.dataset
    frame = pd.DataFrame.from_records(records).drop(columns=list(ARRAY_COLUMNS), errors="ignore")
    frame.to_csv(f"{stem}_records.csv", index=False)
    pd.DataFrame.from_records(summarize(records)).to_csv(f"{stem}_summary.csv", index=False)

    screen = ev.screening_records(X, y, groups, block_size=args.block_size)
    pd.DataFrame.from_records(screen).to_csv(f"{stem}_screening.csv", index=False)
    if len(screen) >= 3:
        m = screening_metrics([r["singleton_rate_block"] for r in screen],
                              [r["heldout_accuracy"] for r in screen], random_state=args.seed)
        Path(f"{stem}_screening.json").write_text(json.dumps(m, indent=2))
        print("screening:", json.dumps(m, indent=2))
    print(f"wrote {stem}_records.csv, {stem}_summary.csv, {stem}_screening.csv")


if __name__ == "__main__":
    main()
