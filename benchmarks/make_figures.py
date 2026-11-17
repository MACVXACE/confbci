"""Regenerate the paper figures.

    python benchmarks/make_figures.py --records results/Liu2024_records.csv
    python benchmarks/make_figures.py --demo        # synthetic subjects, no download

Figure 1 (``fig_r2_coverage``)   per-subject coverage against calibration budget k, regime R2
Figure 2 (``fig_beta_validity``) coverage over 1,000 random calibration sets against the
                                 exact Beta(n + 1 - l, l) law, on synthetic SPD matrices

``--records`` takes the CSV written by ``benchmarks/run_benchmark.py`` (or any table
with columns regime, method, target, k, order, coverage, coverage_theory, alpha).
Figures are saved as PDF and PNG in ``--out`` (default ``paper/figures``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from confbci import LAC, LOSOConformalEvaluation, ProbMDM, SplitCP, conformal_rank, empirical_coverage
from confbci.scores import true_label_scores
from confbci.sim import make_class_means, sample_dataset, sample_multisubject

# -- style: two validated categorical hues, neutral ink for everything that is not data
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ORDER_COLORS = {"chronological": BLUE, "shuffled": ORANGE}
METHOD_TITLES = {"mondrian": "Mondrian CP", "mondrian_recentered": "Mondrian CP, re-centered decoder"}


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper", rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.pdf and .png")


# --------------------------------------------------------------------------- #
# Figure 1: coverage against calibration budget (R2)
# --------------------------------------------------------------------------- #
def fig_r2_coverage(records: pd.DataFrame, out: Path, title_suffix: str = "") -> None:
    df = records[records["regime"] == "R2"].copy()
    if df.empty:
        raise SystemExit("No R2 rows in the records table.")
    alpha = float(df["alpha"].iloc[0])
    # One value per subject: average sessions first.
    df = (df.groupby(["method", "order", "k", "target"], as_index=False)
            .agg(coverage=("coverage", "mean"), coverage_theory=("coverage_theory", "mean")))
    df["k"] = df["k"].astype(int)
    ks = sorted(df["k"].unique())
    methods = [m for m in METHOD_TITLES if m in set(df["method"])]
    n_subjects = df["target"].nunique()

    fig, axes = plt.subplots(1, len(methods), figsize=(3.5 * len(methods), 2.9), sharey=True, squeeze=False)
    for ax, method in zip(axes[0], methods):
        d = df[df["method"] == method]
        sns.stripplot(data=d, x="k", y="coverage", hue="order", order=ks,
                      hue_order=list(ORDER_COLORS), palette=ORDER_COLORS, dodge=True, jitter=0.18,
                      size=2.6, alpha=0.35, linewidth=0, legend=False, ax=ax, zorder=2)
        sns.pointplot(data=d, x="k", y="coverage", hue="order", order=ks,
                      hue_order=list(ORDER_COLORS), palette=ORDER_COLORS, dodge=0.4,
                      errorbar=("ci", 95), n_boot=2000, seed=0, markers="o", markersize=4.5,
                      linewidth=1.6, err_kws={"linewidth": 1.2}, legend=False, ax=ax, zorder=4)
        theory = d.groupby("k")["coverage_theory"].mean().reindex(ks).to_numpy()
        ax.plot(range(len(ks)), theory, color=INK, lw=1.0, ls=(0, (1, 2)), marker="_", ms=9, zorder=3)
        ax.axhline(1 - alpha, color=INK_2, lw=0.8, ls="--", zorder=1)
        ax.set_title(METHOD_TITLES[method], loc="left")
        ax.set_xlabel("Calibration trials per class, k")
        ax.set_ylabel("Coverage per subject" if ax is axes[0][0] else "")
        ax.set_ylim(min(0.3, d["coverage"].min() - 0.03), 1.03)

    handles = [plt.Line2D([], [], color=c, marker="o", ms=4.5, lw=1.6, label=f"{name} (mean, 95% CI)")
               for name, c in ORDER_COLORS.items()]
    handles.append(plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 2)), marker="_", ms=9,
                              label="expected if exchangeable"))
    handles.append(plt.Line2D([], [], color=INK_2, lw=0.8, ls="--", label=f"target {1 - alpha:.2f}"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Few-shot calibration (R2): {n_subjects} subjects, one dot per subject{title_suffix}",
                 x=0.01, ha="left", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig_r2_coverage")


# --------------------------------------------------------------------------- #
# Figure 2: Beta validity on synthetic SPD matrices
# --------------------------------------------------------------------------- #
def beta_coverages(n_cal: int, alpha: float, n_splits: int = 1000, n_pool: int = 20_000, seed: int = 20261117):
    """Same protocol as tests/test_conformal.py::test_conditional_coverage_follows_beta."""
    rng = np.random.default_rng(seed)
    means = make_class_means(4, n_classes=2, separation=0.6, rng=rng)
    X_train, y_train = sample_dataset(400, means, 0.6, rng)
    model = ProbMDM(recenter=False, random_state=0).fit(X_train, y_train)
    X, y = sample_dataset(n_pool, means, 0.6, rng)
    S = LAC()(model, X)
    s_true = true_label_scores(S, y)
    cp = SplitCP(model=None, alpha=alpha)
    split_rng = np.random.default_rng(n_cal)
    cov = np.empty(n_splits)
    for t in range(n_splits):
        perm = split_rng.permutation(n_pool)
        cal, test = perm[:n_cal], perm[n_cal:]
        cp.calibrate_from_scores(s_true[cal])
        cov[t] = empirical_coverage(cp.predict_set_from_scores(S[test]), y[test])
    return cov


def fig_beta_validity(out: Path, alpha: float = 0.2, n_cals=(50, 99, 400)) -> None:
    fig, axes = plt.subplots(1, len(n_cals), figsize=(7.0, 2.5), sharey=False)
    for ax, n in zip(axes, n_cals):
        cov = beta_coverages(n, alpha)
        k = conformal_rank(n, alpha)
        law = stats.beta(k, n + 1 - k)
        p = stats.kstest(cov, law.cdf).pvalue
        sns.histplot(x=cov, binwidth=law.std() / 2.5, stat="density", color=BLUE, edgecolor=SURFACE, linewidth=0.8,
                     alpha=1.0, ax=ax, zorder=2)
        x = np.linspace(law.ppf(0.0005), law.ppf(0.9995), 400)
        ax.plot(x, law.pdf(x), color=INK, lw=1.4, zorder=3)
        ax.axvline(1 - alpha, color=INK_2, lw=0.8, ls="--", zorder=1)
        ax.set_title(f"n = {n}:  Beta({k}, {n + 1 - k})", loc="left")
        ax.text(0.03, 0.95, f"KS p = {p:.2f}", transform=ax.transAxes, va="top", color=INK_2, fontsize=7.5)
        ax.set_xlabel("Coverage given the calibration set")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.set_xlim(0.62, 0.95)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="1,000 random calibration sets"),
               plt.Line2D([], [], color=INK, lw=1.4, label="exact Beta law"),
               plt.Line2D([], [], color=INK_2, lw=0.8, ls="--", label=f"target {1 - alpha:.2f}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Split conformal coverage on synthetic SPD matrices follows its exact law",
                 x=0.01, ha="left", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig_beta_validity")


# --------------------------------------------------------------------------- #
def demo_records() -> pd.DataFrame:
    """Synthetic subjects with within-session drift, for trying the script without data."""
    means = make_class_means(4, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(20, 240, means, 0.6, shift=1.0, rng=np.random.default_rng(9), drift=2.0)
    ev = LOSOConformalEvaluation(alpha=0.2, n_shuffles=10)
    return pd.DataFrame.from_records(ev.run(X, y, g, regimes=("R2",)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=Path, help="CSV from benchmarks/run_benchmark.py")
    p.add_argument("--demo", action="store_true", help="use synthetic subjects for Figure 1")
    p.add_argument("--out", type=Path, default=Path("paper/figures"))
    p.add_argument("--skip-beta", action="store_true")
    args = p.parse_args()
    set_style()

    if args.demo:
        fig_r2_coverage(demo_records(), args.out / "demo", title_suffix=" (SYNTHETIC DEMO)")
    elif args.records is not None:
        fig_r2_coverage(pd.read_csv(args.records), args.out)
    else:
        print("No --records given: skipping Figure 1.")
    if not args.skip_beta:
        fig_beta_validity(args.out)


if __name__ == "__main__":
    main()
