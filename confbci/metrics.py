"""Online-coverage and patient-screening metrics.

Patient screening
-----------------
A known share of users cannot drive a motor-imagery BCI above about 70%
accuracy ("BCI inefficiency"). The question here is whether a short block at
the start of a session predicts who they are. The predictor is the *singleton
rate* on that block: the fraction of trials for which the conformal predictor
commits to a single label. It needs no labels from the patient.

``screening_metrics`` reports

- Spearman's rho between block singleton rate and held-out accuracy, and
- the AUROC of the singleton rate for predicting ``accuracy >= 0.70``,

each with a bootstrap confidence interval resampled over patients.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "rolling_coverage",
    "long_run_coverage_error",
    "auroc",
    "screening_metrics",
]


# --------------------------------------------------------------------------- #
# Online metrics
# --------------------------------------------------------------------------- #
def rolling_coverage(covered, window: int = 20) -> np.ndarray:
    """Coverage over a sliding window of trials.

    Parameters
    ----------
    covered : array-like of bool, shape (T,)
        Whether the true label was in the prediction set at each trial.
    window : int

    Returns
    -------
    ndarray, shape (max(T - window + 1, 0),)
        Entry ``i`` is the coverage over trials ``i .. i + window - 1``.
    """
    c = np.asarray(covered, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1.")
    if c.size < window:
        return np.empty(0)
    csum = np.concatenate([[0.0], np.cumsum(c)])
    return (csum[window:] - csum[:-window]) / window


def long_run_coverage_error(covered, alpha: float) -> float:
    """``| (1/T) sum_t err_t - alpha |``: distance of the realised miscoverage from target."""
    c = np.asarray(covered, dtype=float)
    return float(abs((1.0 - c).mean() - alpha))


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def auroc(scores, positive) -> float:
    """Area under the ROC curve (Mann-Whitney form, ties count one half).

    The probability that a randomly chosen positive has a higher score than a
    randomly chosen negative. Returns ``nan`` when either class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)                        # average ranks handle ties
    u = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _spearman(x, y) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def screening_metrics(
    singleton_rate,
    heldout_accuracy,
    accuracy_threshold: float = 0.70,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state=0,
) -> dict:
    """Does the singleton rate on a short block predict who can use the BCI?

    Parameters
    ----------
    singleton_rate : array-like, shape (n_patients,)
        Fraction of singleton prediction sets on each patient's screening block.
    heldout_accuracy : array-like, shape (n_patients,)
        Decoder accuracy on that patient's remaining trials.
    accuracy_threshold : float, default 0.70
        Patients at or above it count as "BCI-efficient".
    n_boot : int
        Bootstrap resamples over patients for the confidence intervals
        (0 disables them).
    ci : float
        Confidence level.

    Returns
    -------
    dict with
        ``n_patients``, ``n_efficient``, ``n_inefficient``,
        ``spearman_rho``, ``spearman_p`` (two-sided, exact null for small n is
        not used; treat p as approximate below about 10 patients),
        ``spearman_ci`` (low, high),
        ``auroc``, ``auroc_ci`` (low, high).
        ``auroc`` is ``nan`` if every patient falls on one side of the threshold.
    """
    x = np.asarray(singleton_rate, dtype=float)
    a = np.asarray(heldout_accuracy, dtype=float)
    if x.shape != a.shape or x.ndim != 1:
        raise ValueError("singleton_rate and heldout_accuracy must be 1-D arrays of equal length.")
    keep = ~(np.isnan(x) | np.isnan(a))
    x, a = x[keep], a[keep]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least three patients.")
    efficient = a >= accuracy_threshold

    if np.ptp(x) == 0 or np.ptp(a) == 0:
        rho, p = float("nan"), float("nan")
    else:
        res = stats.spearmanr(x, a)
        rho, p = float(res[0]), float(res[1])
    out = {
        "n_patients": int(n),
        "n_efficient": int(efficient.sum()),
        "n_inefficient": int((~efficient).sum()),
        "accuracy_threshold": accuracy_threshold,
        "spearman_rho": rho,
        "spearman_p": p,
        "spearman_ci": (float("nan"), float("nan")),
        "auroc": auroc(x, efficient),
        "auroc_ci": (float("nan"), float("nan")),
    }

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        rhos, aucs = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            rhos.append(_spearman(x[idx], a[idx]))
            aucs.append(auroc(x[idx], efficient[idx]))
        lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
        for key, vals in (("spearman_ci", rhos), ("auroc_ci", aucs)):
            vals = np.asarray(vals)
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 0.5 * n_boot:              # skip when most resamples are degenerate
                out[key] = (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))
    return out
