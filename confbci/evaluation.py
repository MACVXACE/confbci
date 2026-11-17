"""Leave-one-subject-out evaluation of conformal predictors.

Two regimes are evaluated for every held-out target subject.

R1, zero calibration
    No labelled target trials. Source subjects are split *by subject* into
    training subjects (fit the decoder) and calibration subjects (fit the
    conformal threshold). Three methods:

    ``pooled``      plain split conformal on the calibration subjects;
    ``weighted``    weighted conformal with density-ratio weights between the
                    calibration trials and the unlabeled target trials;
    ``recentered``  every subject re-centered to the identity (target included,
                    using its unlabeled trials), then plain split conformal.

R2, few-shot
    The first ``k`` trials per class of the target are its calibration block
    and the remaining trials are the test set, with a per-subject (Mondrian)
    threshold. Run with trials in recorded order (``chronological``) and with
    the calibration block drawn at random (``shuffled``). Shuffling makes
    calibration and test exchangeable by construction, so the difference
    between the two orders measures within-session non-stationarity. When
    session labels are given, each session is evaluated separately.

R3, online
    The target's trials are processed in recorded order. After each trial the
    cue is revealed and adaptive conformal inference (``AdaptiveCI``) updates
    its working level. The decoder is trained on the training subjects and the
    predictor starts from the scores of the calibration subjects, as in R1.
    Methods: ``static`` (the R1 threshold, never updated, as a reference),
    ``aci`` and ``aci_recentered`` for each step size ``gamma``.

    ``aci_recentered`` re-centers a session with the mean of *all* its trials,
    including future ones. That is not available in a live session, so these
    rows are an offline upper bound and carry ``causal = False``.

Screening
    ``screening_records`` gives, per patient, the singleton rate on a short
    unlabeled block at the start and the decoder accuracy on the rest; feed it
    to :func:`confbci.metrics.screening_metrics`.

Exchangeability and re-centering
    Whenever a re-centering model is used, all trials of a subject (session)
    that will be compared with each other are scored in one call, so they share
    one re-centering reference. In R2 the target's calibration and test trials
    are scored together and only then split.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone

from .conformal import AdaptiveCI, MondrianCP, SplitCP, WeightedCP, aci_coverage_bound, conformal_rank
from .metrics import long_run_coverage_error, rolling_coverage
from .models.riemann import ProbMDM
from .scores import LAC, NonconformityScore, true_label_scores
from .shift import DomainClassifierDensityRatio

__all__ = ["set_metrics", "LOSOConformalEvaluation", "summarize", "records_to_frame"]


def set_metrics(sets, y_idx) -> dict:
    """Validity, efficiency and clinical metrics of a batch of prediction sets.

    Parameters
    ----------
    sets : bool ndarray, shape (n_trials, n_classes)
    y_idx : int ndarray, shape (n_trials,)
        True class indices.

    Returns
    -------
    dict with
        ``coverage``             P(true label in set)
        ``mean_set_size``
        ``singleton_rate``       P(|set| = 1): trials where feedback would be given
        ``abstain_rate``         P(|set| >= 2)
        ``empty_rate``           P(|set| = 0)
        ``false_feedback_rate``  P(|set| = 1 and the label is wrong)
        ``selective_accuracy``   accuracy on singleton trials (nan if none)
        ``n_test``
    """
    sets = np.asarray(sets, dtype=bool)
    y_idx = np.asarray(y_idx, dtype=int)
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(y_idx)), y_idx]
    single = size == 1
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "singleton_rate": float(single.mean()),
        "abstain_rate": float((size >= 2).mean()),
        "empty_rate": float((size == 0).mean()),
        "false_feedback_rate": float((single & ~covered).mean()),
        "selective_accuracy": float(covered[single].mean()) if single.any() else float("nan"),
        "n_test": int(len(y_idx)),
    }


class LOSOConformalEvaluation:
    """Leave-one-subject-out benchmark of conformal methods under subject shift.

    Parameters
    ----------
    score : NonconformityScore, default ``LAC()``
    alpha : float, default 0.2
    ks : tuple of int, default (2, 4, 8, 16, 32)
        Calibration trials *per class* in regime R2.
    n_shuffles : int, default 20
        Random calibration blocks per (subject, session, k) in the shuffled
        condition; metrics are averaged over them.
    cal_subject_fraction : float, default 0.3
        Fraction of source subjects held out for calibration in R1 (at least 1).
    min_test_trials : int, default 10
        An R2 configuration is skipped when fewer test trials would remain.
    density_ratio : DomainClassifierDensityRatio, optional
        Prototype, cloned for every fold.
    model_params : dict, optional
        Extra keyword arguments for :class:`ProbMDM` (``recenter`` is set by
        the evaluation).
    gammas : tuple of float, default (0.005, 0.01, 0.05)
        Step sizes of adaptive conformal inference in regime R3.
    rolling_window : int, default 20
        Window of the rolling coverage in R3.
    keep_trajectories : bool, default True
        Store ``alpha_trajectory``, ``rolling_coverage`` and ``covered`` arrays
        in each R3 record.
    random_state : int, default 0

    Notes
    -----
    Trials of each subject must be in recording order for the chronological
    condition to mean anything. :func:`confbci.data.load_covariances` keeps the
    order MOABB returns.
    """

    def __init__(
        self,
        score: NonconformityScore | None = None,
        alpha: float = 0.2,
        ks=(2, 4, 8, 16, 32),
        n_shuffles: int = 20,
        cal_subject_fraction: float = 0.3,
        min_test_trials: int = 10,
        density_ratio: DomainClassifierDensityRatio | None = None,
        model_params: dict | None = None,
        gammas=(0.005, 0.01, 0.05),
        rolling_window: int = 20,
        keep_trajectories: bool = True,
        random_state: int = 0,
    ):
        self.score = LAC() if score is None else score
        self.alpha = alpha
        self.ks = tuple(ks)
        self.n_shuffles = n_shuffles
        self.cal_subject_fraction = cal_subject_fraction
        self.min_test_trials = min_test_trials
        self.density_ratio = DomainClassifierDensityRatio() if density_ratio is None else density_ratio
        self.model_params = dict(model_params or {})
        self.gammas = tuple(gammas)
        self.rolling_window = rolling_window
        self.keep_trajectories = keep_trajectories
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _model(self, recenter: bool) -> ProbMDM:
        params = {"random_state": self.random_state, **self.model_params, "recenter": recenter}
        return ProbMDM(**params)

    def _record(self, regime, method, target, sets, y_idx, **extra) -> dict:
        rec = {"regime": regime, "method": method, "target": target, "alpha": self.alpha,
               "k": None, "order": None, "session": None}
        rec.update(set_metrics(sets, y_idx))
        rec.update(extra)
        return rec

    # ------------------------------------------------------------------ #
    def run(self, X, y, groups, sessions=None, regimes=("R1", "R2"), targets=None) -> list:
        """Run the benchmark.

        Parameters
        ----------
        X : ndarray, shape (n_trials, c, c)
        y : array-like, shape (n_trials,)
        groups : array-like, shape (n_trials,)
            Subject of each trial.
        sessions : array-like, shape (n_trials,), optional
            Session of each trial. R2 is then evaluated within each session.
        regimes : subset of ("R1", "R2", "R3")
            Default ``("R1", "R2")``.
        targets : iterable, optional
            Subjects to hold out (default: all).

        Returns
        -------
        records : list of dict
            One row per (regime, method, target[, session, k, order]).
        """
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        classes, y_idx = np.unique(np.asarray(y), return_inverse=True)
        sessions = None if sessions is None else np.asarray(sessions)
        subjects = np.unique(groups)
        if len(subjects) < 3:
            raise ValueError("Need at least three subjects (train, calibration, target).")
        targets = subjects if targets is None else np.asarray(list(targets))

        records = []
        for target in targets:
            # Seed by the subject's position among all subjects, so a subject's result
            # does not depend on which other targets are evaluated.
            i = int(np.flatnonzero(subjects == target)[0])
            tkey = target.item() if hasattr(target, "item") else target
            # R1 and R3 draw the same train/calibration subject split.
            if "R1" in regimes:
                rng = np.random.default_rng([self.random_state, i])
                records += self._r1(X, y_idx, groups, target, tkey, rng)
            if "R2" in regimes:
                rng = np.random.default_rng([self.random_state, i, 2])
                records += self._r2(X, y_idx, groups, sessions, target, tkey, rng)
            if "R3" in regimes:
                rng = np.random.default_rng([self.random_state, i])
                records += self._r3(X, y_idx, groups, sessions, target, tkey, rng)
        self.classes_ = classes
        return records

    def run_moabb(self, dataset: str = "BNCI2014_001", subjects=None, regimes=("R1", "R2"), **load_kwargs) -> list:
        """Load a MOABB dataset with :func:`confbci.data.load_covariances` and run."""
        from .data import load_covariances

        X, y, groups, sessions = load_covariances(dataset, subjects=subjects, **load_kwargs)
        records = self.run(X, y, groups, sessions=sessions, regimes=regimes)
        for r in records:
            r["dataset"] = dataset
        return records

    # ------------------------------------------------------------------ #
    # Regime R1: zero calibration
    # ------------------------------------------------------------------ #
    def _split_sources(self, groups, target, rng):
        src_subjects = np.unique(groups[groups != target])
        n_cal = max(1, int(round(self.cal_subject_fraction * len(src_subjects))))
        n_cal = min(n_cal, len(src_subjects) - 1)
        perm = rng.permutation(src_subjects)
        cal_subjects, train_subjects = perm[:n_cal], perm[n_cal:]
        return np.isin(groups, train_subjects), np.isin(groups, cal_subjects)

    def _r1(self, X, y_idx, groups, target, tkey, rng) -> list:
        tr, cal = self._split_sources(groups, target, rng)
        te = groups == target
        out = []
        common = {"n_cal": int(cal.sum()), "n_cal_subjects": int(len(np.unique(groups[cal])))}

        # -- pooled and weighted share one decoder without re-centering
        model = self._model(recenter=False).fit(X[tr], y_idx[tr], groups=groups[tr])
        S_cal = self.score(model, X[cal])
        S_te = self.score(model, X[te])
        s_cal = true_label_scores(S_cal, y_idx[cal])
        acc = float((np.argmin(model.sq_distances(X[te]), axis=1) == y_idx[te]).mean())

        cp = SplitCP(model, self.score, self.alpha).calibrate_from_scores(s_cal)
        out.append(self._record("R1", "pooled", tkey, cp.predict_set_from_scores(S_te), y_idx[te],
                                accuracy=acc, **common))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # n_eff is recorded instead
            dr = clone(self.density_ratio).fit(X[cal], X[te])
        wcp = WeightedCP(model, self.score, self.alpha).calibrate_from_scores(s_cal, dr.source_weights_)
        sets = wcp.predict_set_from_scores(S_te, dr.weights(X[te]))
        out.append(self._record("R1", "weighted", tkey, sets, y_idx[te], accuracy=acc,
                                n_eff=dr.n_eff_, n_eff_ratio=dr.n_eff_ratio_, inf_mass=dr.inf_mass_,
                                domain_accuracy=dr.domain_accuracy_, **common))

        # -- re-centered: each subject is its own group; the target uses all its trials
        model_rc = self._model(recenter=True).fit(X[tr], y_idx[tr], groups=groups[tr])
        S_cal = self.score(model_rc, X[cal], groups=groups[cal])
        S_te = self.score(model_rc, X[te], groups=groups[te])
        acc_rc = float((np.argmin(model_rc.sq_distances(X[te], groups=groups[te]), axis=1) == y_idx[te]).mean())
        cp = SplitCP(model_rc, self.score, self.alpha).calibrate_from_scores(
            true_label_scores(S_cal, y_idx[cal]))
        out.append(self._record("R1", "recentered", tkey, cp.predict_set_from_scores(S_te), y_idx[te],
                                accuracy=acc_rc, **common))
        return out

    # ------------------------------------------------------------------ #
    # Regime R2: few-shot, per-subject calibration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_k_per_class(y_idx, k, order):
        """Indices (into ``order``) of the first k trials of each class along ``order``."""
        picked = []
        for c in np.unique(y_idx):
            pos = order[y_idx[order] == c]
            if len(pos) < k:
                return None
            picked.append(pos[:k])
        return np.concatenate(picked)

    def _r2(self, X, y_idx, groups, sessions, target, tkey, rng) -> list:
        src = groups != target
        te = np.flatnonzero(groups == target)
        y_t = y_idx[te]
        sess_t = np.zeros(len(te), dtype=int) if sessions is None else sessions[te]
        out = []

        for method, recenter in (("mondrian", False), ("mondrian_recentered", True)):
            model = self._model(recenter=recenter).fit(X[src], y_idx[src], groups=groups[src])
            for sess in np.unique(sess_t):
                in_sess = np.flatnonzero(sess_t == sess)
                skey = sess.item() if hasattr(sess, "item") else sess
                # One scoring call per session: calibration and test trials of this
                # session share a single re-centering reference.
                S = self.score(model, X[te[in_sess]])
                y_s = y_t[in_sess]
                s_true = true_label_scores(S, y_s)
                n = len(y_s)
                acc = float((np.argmin(S, axis=1) == y_s).mean())

                for k in self.ks:
                    orders = {"chronological": [np.arange(n)],
                              "shuffled": [rng.permutation(n) for _ in range(self.n_shuffles)]}
                    for order_name, perms in orders.items():
                        rows = []
                        for perm in perms:
                            cal = self._first_k_per_class(y_s, k, perm)
                            if cal is None:
                                continue
                            test = np.setdiff1d(np.arange(n), cal)
                            if len(test) < self.min_test_trials:
                                continue
                            g = np.full(n, tkey)
                            cp = MondrianCP(model, self.score, self.alpha).calibrate_from_scores(
                                s_true[cal], g[cal])
                            rows.append(set_metrics(cp.predict_set_from_scores(S[test], g[test]), y_s[test]))
                        if not rows:
                            continue
                        mean = {m: float(np.nanmean([r[m] for r in rows])) if not all(
                            np.isnan(r[m]) for r in rows) else float("nan") for m in rows[0]}
                        mean["n_test"] = int(rows[0]["n_test"])
                        n_cal = int(k * len(np.unique(y_s)))
                        rec = {"regime": "R2", "method": method, "target": tkey, "alpha": self.alpha,
                               "k": int(k), "order": order_name,
                               "session": skey if sessions is not None else None,
                               "n_cal": n_cal, "accuracy": acc,
                               # Expected coverage if calibration and test are exchangeable.
                               "coverage_theory": min(1.0, conformal_rank(n_cal, self.alpha) / (n_cal + 1)),
                               "n_splits": len(rows)}
                        rec.update(mean)
                        out.append(rec)
        return out

    # ------------------------------------------------------------------ #
    # Regime R3: online, cue revealed after every trial
    # ------------------------------------------------------------------ #
    def _r3(self, X, y_idx, groups, sessions, target, tkey, rng) -> list:
        tr, cal = self._split_sources(groups, target, rng)
        te = np.flatnonzero(groups == target)
        y_t = y_idx[te]
        sess_t = np.zeros(len(te), dtype=int) if sessions is None else sessions[te]
        out = []

        for recenter in (False, True):
            model = self._model(recenter=recenter).fit(X[tr], y_idx[tr], groups=groups[tr])
            kw = {"groups": groups[cal]} if recenter else {}
            init = true_label_scores(self.score(model, X[cal], **kw), y_idx[cal])
            suffix = "_recentered" if recenter else ""

            for sess in np.unique(sess_t):
                in_sess = np.flatnonzero(sess_t == sess)        # recorded (chronological) order
                skey = (sess.item() if hasattr(sess, "item") else sess) if sessions is not None else None
                # With re-centering this uses the mean of the whole session (not causal).
                S = self.score(model, X[te[in_sess]])
                y_s = y_t[in_sess]
                T = len(y_s)
                acc = float((np.argmin(S, axis=1) == y_s).mean())

                runs = []
                static = SplitCP(model, self.score, self.alpha).calibrate_from_scores(init)
                sets = static.predict_set_from_scores(S)
                runs.append(("static" + suffix, None, sets, np.full(T, self.alpha)))
                for gamma in self.gammas:
                    res = AdaptiveCI(self.alpha, gamma, init_scores=init).run(S, y_s)
                    runs.append(("aci" + suffix, float(gamma), res["sets"], res["alpha_t"]))

                for method, gamma, sets, alpha_traj in runs:
                    covered = sets[np.arange(T), y_s]
                    roll = rolling_coverage(covered, self.rolling_window)
                    rec = {"regime": "R3", "method": method, "target": tkey, "alpha": self.alpha,
                           "k": None, "order": "chronological", "session": skey, "gamma": gamma,
                           "causal": not recenter, "n_cal": int(len(init)), "accuracy": acc}
                    rec.update(set_metrics(sets, y_s))
                    rec.update({
                        "long_run_error": long_run_coverage_error(covered, self.alpha),
                        "aci_bound": aci_coverage_bound(T, gamma, self.alpha) if gamma else float("nan"),
                        "rolling_min": float(roll.min()) if roll.size else float("nan"),
                        "rolling_std": float(roll.std()) if roll.size else float("nan"),
                        "frac_windows_under": float((roll < 1 - self.alpha - 0.1).mean()) if roll.size else float("nan"),
                        "alpha_t_min": float(alpha_traj.min()),
                        "alpha_t_max": float(alpha_traj.max()),
                        "alpha_t_final": float(alpha_traj[-1]),
                    })
                    if self.keep_trajectories:
                        rec.update({"alpha_trajectory": alpha_traj, "rolling_coverage": roll,
                                    "covered": covered})
                    out.append(rec)
        return out

    # ------------------------------------------------------------------ #
    # Patient screening
    # ------------------------------------------------------------------ #
    def screening_records(self, X, y, groups, block_size: int = 20, recenter: bool = True,
                          targets=None) -> list:
        """Per-patient inputs for :func:`confbci.metrics.screening_metrics`.

        For each held-out patient the first ``block_size`` trials (recorded
        order) form the screening block. The conformal threshold comes from
        held-out source subjects, so the block needs **no labels**: the
        singleton rate is the share of block trials for which the predictor
        commits to one label. Accuracy is measured on the remaining trials.

        With ``recenter=True`` the block is re-centered with its own mean and
        the remaining trials with theirs, so nothing from the held-out trials
        leaks into the screening block.

        Returns
        -------
        list of dict with ``target``, ``singleton_rate_block``,
        ``heldout_accuracy``, ``block_size``, ``n_heldout``.
        """
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        _, y_idx = np.unique(np.asarray(y), return_inverse=True)
        subjects = np.unique(groups)
        targets = subjects if targets is None else np.asarray(list(targets))
        out = []
        for target in targets:
            # Seed by the subject's position among all subjects, so a subject's result
            # does not depend on which other targets are evaluated.
            i = int(np.flatnonzero(subjects == target)[0])
            rng = np.random.default_rng([self.random_state, i])
            tr, cal = self._split_sources(groups, target, rng)
            te = np.flatnonzero(groups == target)
            if len(te) <= block_size:
                continue
            block, rest = te[:block_size], te[block_size:]
            model = self._model(recenter=recenter).fit(X[tr], y_idx[tr], groups=groups[tr])
            kw = {"groups": groups[cal]} if recenter else {}
            cp = SplitCP(model, self.score, self.alpha).calibrate_from_scores(
                true_label_scores(self.score(model, X[cal], **kw), y_idx[cal]))
            sets = cp.predict_set_from_scores(self.score(model, X[block]))
            pred = np.argmin(model.sq_distances(X[rest]), axis=1)
            out.append({
                "target": target.item() if hasattr(target, "item") else target,
                "singleton_rate_block": float((sets.sum(axis=1) == 1).mean()),
                "heldout_accuracy": float((pred == y_idx[rest]).mean()),
                "block_size": int(block_size),
                "n_heldout": int(len(rest)),
            })
        return out


# ---------------------------------------------------------------------- #
# Aggregation
# ---------------------------------------------------------------------- #
def summarize(records, tolerance: float = 0.05) -> list:
    """Aggregate per-target records over subjects.

    Sessions of a subject are averaged first, so every subject counts once.

    Returns one dict per (regime, method, k, order, gamma) with
        ``coverage_mean``, ``coverage_min``, ``coverage_worst_decile`` (mean of the
        lowest 10% of subjects), ``frac_undercovered`` (subjects with coverage
        below ``1 - alpha - tolerance``), ``singleton_rate``,
        ``false_feedback_rate``, ``selective_accuracy``, ``accuracy``,
        ``n_eff_ratio`` and ``inf_mass`` (weighted method only), ``coverage_theory`` (R2 only: the
        expected coverage ``k / (n + 1)`` under exchangeability), the R3 fields
        ``long_run_error``, ``aci_bound``, ``rolling_min`` and ``frac_windows_under``
        (share of rolling windows with coverage below ``1 - alpha - 0.1``), and
        ``n_subjects``.

    A single chronological calibration block of ``n`` trials gives a coverage
    that varies between subjects like ``Beta(k, n + 1 - k)`` even when nothing
    is wrong, so for small ``n`` judge R2 by ``coverage_mean`` against
    ``coverage_theory`` and by the chronological-vs-shuffled gap, not by
    ``frac_undercovered``.
    """
    metrics = ("coverage", "singleton_rate", "false_feedback_rate", "selective_accuracy",
               "accuracy", "mean_set_size", "n_eff_ratio", "inf_mass", "coverage_theory",
               "long_run_error", "aci_bound", "rolling_min", "frac_windows_under")
    cells = {}
    for r in records:
        key = (r["regime"], r["method"], r.get("k"), r.get("order"), r["alpha"], r.get("gamma"))
        cells.setdefault(key, {}).setdefault(r["target"], []).append(r)

    out = []
    for (regime, method, k, order, alpha, gamma), by_subject in cells.items():
        per_subj = {m: np.array([np.nanmean([r.get(m, np.nan) for r in rows]) if any(
            not np.isnan(r.get(m, np.nan)) for r in rows) else np.nan
            for rows in by_subject.values()]) for m in metrics}
        cov = per_subj["coverage"]
        n_worst = max(1, int(np.ceil(0.1 * len(cov))))
        row = {
            "regime": regime, "method": method, "k": k, "order": order, "gamma": gamma, "alpha": alpha,
            "n_subjects": len(cov),
            "coverage_mean": float(cov.mean()),
            "coverage_min": float(cov.min()),
            "coverage_worst_decile": float(np.sort(cov)[:n_worst].mean()),
            "frac_undercovered": float((cov < 1 - alpha - tolerance).mean()),
        }
        for m in metrics[1:]:
            v = per_subj[m]
            row[m] = float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan")
        out.append(row)
    return out


def records_to_frame(records):
    """Records (or a summary) as a pandas DataFrame. Requires pandas."""
    import pandas as pd

    return pd.DataFrame.from_records(records)
