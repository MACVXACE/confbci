"""Split conformal prediction.

Guarantee
---------
Let ``s_1, ..., s_n`` be the true-label nonconformity scores of ``n``
calibration trials and let ``qhat`` be their ``k``-th smallest value with

    k = ceil((n + 1) (1 - alpha)).

If calibration and test trials are exchangeable, the set
``C(x) = {y : s(x, y) <= qhat}`` satisfies

    1 - alpha  <=  P(Y in C(X))  <=  1 - alpha + 1 / (n + 1)

(the upper bound needs scores without ties). For i.i.d. data with continuous
scores the coverage *conditional on the calibration set* is a random variable
with an exact distribution:

    P(Y in C(X) | calibration set)  ~  Beta(k, n + 1 - k)
                                     =  Beta(n + 1 - l, l),   l = floor((n + 1) alpha).

``tests/test_conformal.py`` checks this distribution with a Kolmogorov-Smirnov
test.
"""

from __future__ import annotations

import numpy as np

from .scores import LAC, NonconformityScore, true_label_scores

__all__ = [
    "conformal_rank",
    "split_quantile",
    "weighted_quantile",
    "SplitCP",
    "WeightedCP",
    "MondrianCP",
    "AdaptiveCI",
    "aci_coverage_bound",
    "crc_feedback_threshold",
    "feedback_risk",
    "empirical_coverage",
    "set_sizes",
]


def conformal_rank(n: int, alpha: float) -> int:
    """Rank ``k = ceil((n + 1)(1 - alpha))`` of the calibration score used as threshold.

    ``k > n`` means the calibration set is too small for this ``alpha`` and the
    threshold is ``+inf``. The product is rounded to 9 decimals before the
    ceiling so that floating-point noise such as ``100 * (1 - 0.1) =
    90.00000000000001`` cannot push the rank up by one.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    return int(np.ceil(np.round((n + 1) * (1.0 - alpha), 9)))


def split_quantile(scores, alpha: float) -> float:
    """Conformal threshold: the ``ceil((n + 1)(1 - alpha))``-th smallest score.

    Returns ``+inf`` when that rank exceeds ``n`` (every label is then kept, so
    coverage is 1 and the guarantee still holds).

    Note: this is an exact order statistic. ``np.quantile(scores, k / n,
    method="higher")``, which appears in many tutorials, returns the
    ``(k + 1)``-th smallest score for most ``(n, alpha)`` and is therefore
    slightly more conservative than necessary.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if np.isnan(scores).any():
        raise ValueError("Calibration scores contain NaN.")
    n = scores.size
    k = conformal_rank(n, alpha)
    if k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def _label_index(classes: np.ndarray, y) -> np.ndarray:
    y = np.asarray(y)
    idx = np.searchsorted(classes, y)
    idx = np.clip(idx, 0, len(classes) - 1)
    if not np.array_equal(classes[idx], y):
        raise ValueError("y contains labels that the model has not seen.")
    return idx


def empirical_coverage(sets: np.ndarray, y_idx: np.ndarray) -> float:
    """Fraction of trials whose true label (given as class index) is in its set."""
    sets = np.asarray(sets, dtype=bool)
    y_idx = np.asarray(y_idx, dtype=int)
    return float(sets[np.arange(sets.shape[0]), y_idx].mean())


def set_sizes(sets: np.ndarray) -> np.ndarray:
    """Number of labels in each prediction set."""
    return np.asarray(sets, dtype=bool).sum(axis=1)


class SplitCP:
    """Split conformal predictor around a fitted classifier.

    Parameters
    ----------
    model : fitted estimator
        Must expose ``classes_`` and whatever the score needs
        (``predict_proba`` for :class:`LAC`, ``distances`` for
        :class:`RiemannianMargin`). The model must not have been trained on the
        calibration trials.
    score : NonconformityScore, default ``LAC()``
    alpha : float in (0, 1), default 0.2
        Target miscoverage: sets contain the true label with probability
        at least ``1 - alpha``.

    Attributes
    ----------
    qhat_ : float
        Calibrated threshold.
    n_cal_ : int
        Number of calibration trials.

    Examples
    --------
    >>> cp = SplitCP(model, score=RiemannianMargin(), alpha=0.2).calibrate(X_cal, y_cal)
    >>> sets = cp.predict_set(X_test)        # bool array (n_trials, n_classes)
    >>> cp.set_labels(sets)[0]               # e.g. array(['left']) or array(['left', 'right'])
    """

    def __init__(self, model, score: NonconformityScore | None = None, alpha: float = 0.2):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.model = model
        self.score = LAC() if score is None else score
        self.alpha = alpha

    # -- calibration ------------------------------------------------------- #
    def calibrate(self, X, y, **score_kwargs) -> "SplitCP":
        """Compute the threshold from labelled calibration trials."""
        y_idx = _label_index(self.model.classes_, y)
        S = self.score(self.model, X, **score_kwargs)
        return self.calibrate_from_scores(true_label_scores(S, y_idx))

    def calibrate_from_scores(self, cal_scores) -> "SplitCP":
        """Compute the threshold from precomputed true-label scores."""
        cal_scores = np.asarray(cal_scores, dtype=float).ravel()
        self.n_cal_ = int(cal_scores.size)
        self.qhat_ = split_quantile(cal_scores, self.alpha)
        return self

    # -- prediction -------------------------------------------------------- #
    def _check_calibrated(self):
        if not hasattr(self, "qhat_"):
            raise RuntimeError("Call calibrate() before predicting.")

    def predict_set(self, X, **score_kwargs) -> np.ndarray:
        """Prediction sets as a boolean array of shape (n_trials, n_classes).

        ``sets[i, k]`` is True when class ``model.classes_[k]`` is in the set of
        trial ``i``. A row with a single True is a confident decision; a row
        with several is an abstention between those labels.
        """
        self._check_calibrated()
        return self.predict_set_from_scores(self.score(self.model, X, **score_kwargs))

    def predict_set_from_scores(self, S) -> np.ndarray:
        """Prediction sets from a precomputed score array (n_trials, n_classes)."""
        self._check_calibrated()
        return np.asarray(S) <= self.qhat_

    def set_labels(self, sets) -> list:
        """Convert a boolean set array to a list of label arrays."""
        return [self.model.classes_[row] for row in np.asarray(sets, dtype=bool)]

    def coverage(self, sets, y) -> float:
        """Empirical coverage of ``sets`` against true labels ``y``."""
        return empirical_coverage(sets, _label_index(self.model.classes_, y))


# --------------------------------------------------------------------------- #
# Weighted conformal prediction (covariate shift)
# --------------------------------------------------------------------------- #
_CDF_TOL = 1e-12


def weighted_quantile(scores, w_cal, w_test: float, alpha: float) -> float:
    """Weighted conformal threshold of Tibshirani et al. (2019).

    The calibration scores get probabilities ``p_i = w_i / (sum_j w_j + w_test)``
    and a point mass ``w_test / (sum_j w_j + w_test)`` sits at ``+inf``. The
    threshold is the ``(1 - alpha)`` quantile of that distribution: the smallest
    score whose cumulative probability reaches ``1 - alpha``, or ``+inf`` when
    the calibration scores alone cannot reach it.

    With all weights equal to 1 this is exactly :func:`split_quantile`.

    Parameters
    ----------
    scores : array-like, shape (n,)
        True-label calibration scores.
    w_cal : array-like, shape (n,)
        Non-negative density-ratio weights ``w(x_i)`` of the calibration trials.
    w_test : float
        Weight ``w(x)`` of the test trial.
    alpha : float in (0, 1)
    """
    return float(_weighted_thresholds(scores, w_cal, np.atleast_1d(w_test), alpha)[0])


def _weighted_thresholds(scores, w_cal, w_test, alpha: float) -> np.ndarray:
    """Vectorised :func:`weighted_quantile` over many test weights."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    scores = np.asarray(scores, dtype=float).ravel()
    w_cal = np.asarray(w_cal, dtype=float).ravel()
    w_test = np.asarray(w_test, dtype=float).ravel()
    if scores.shape != w_cal.shape:
        raise ValueError("scores and w_cal must have the same length.")
    if np.isnan(scores).any() or np.isnan(w_cal).any() or np.isnan(w_test).any():
        raise ValueError("Scores and weights must not contain NaN.")
    if (w_cal < 0).any() or (w_test < 0).any():
        raise ValueError("Weights must be non-negative.")

    order = np.argsort(scores, kind="stable")
    s_sorted = np.append(scores[order], np.inf)
    cum_w = np.cumsum(w_cal[order])
    total = cum_w[-1] if cum_w.size else 0.0
    # Smallest j with cum_w[j] / (total + w_test) >= 1 - alpha.
    target = (1.0 - alpha) * (total + w_test)
    idx = np.searchsorted(cum_w, target - _CDF_TOL * np.maximum(target, 1.0), side="left")
    return s_sorted[idx]


class WeightedCP:
    """Weighted split conformal predictor for covariate shift.

    The threshold depends on the test trial through its weight, so each test
    trial gets its own ``qhat``. Coverage is guaranteed when the weights equal
    the true density ratio ``p_target(x) / p_source(x)`` and ``P(y | x)`` is the
    same in both domains.

    Parameters
    ----------
    model, score, alpha
        As in :class:`SplitCP`.

    Examples
    --------
    >>> dr = DomainClassifierDensityRatio().fit(X_cal, X_target)
    >>> cp = WeightedCP(model, alpha=0.2).calibrate(X_cal, y_cal, weights=dr.weights(X_cal))
    >>> sets = cp.predict_set(X_target, weights=dr.weights(X_target))
    """

    def __init__(self, model, score: NonconformityScore | None = None, alpha: float = 0.2):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.model = model
        self.score = LAC() if score is None else score
        self.alpha = alpha

    def calibrate(self, X, y, weights, **score_kwargs) -> "WeightedCP":
        y_idx = _label_index(self.model.classes_, y)
        S = self.score(self.model, X, **score_kwargs)
        return self.calibrate_from_scores(true_label_scores(S, y_idx), weights)

    def calibrate_from_scores(self, cal_scores, weights) -> "WeightedCP":
        self.cal_scores_ = np.asarray(cal_scores, dtype=float).ravel()
        self.cal_weights_ = np.asarray(weights, dtype=float).ravel()
        if self.cal_scores_.shape != self.cal_weights_.shape:
            raise ValueError("Need one weight per calibration trial.")
        self.n_cal_ = int(self.cal_scores_.size)
        return self

    def _check_calibrated(self):
        if not hasattr(self, "cal_scores_"):
            raise RuntimeError("Call calibrate() before predicting.")

    def thresholds(self, weights) -> np.ndarray:
        """Per-trial thresholds ``qhat(x)`` for the given test weights."""
        self._check_calibrated()
        return _weighted_thresholds(self.cal_scores_, self.cal_weights_, weights, self.alpha)

    def predict_set(self, X, weights, **score_kwargs) -> np.ndarray:
        self._check_calibrated()
        return self.predict_set_from_scores(self.score(self.model, X, **score_kwargs), weights)

    def predict_set_from_scores(self, S, weights) -> np.ndarray:
        S = np.asarray(S)
        q = self.thresholds(weights)
        if q.shape[0] != S.shape[0]:
            raise ValueError("Need one weight per test trial.")
        return S <= q[:, None]

    def set_labels(self, sets) -> list:
        return [self.model.classes_[row] for row in np.asarray(sets, dtype=bool)]

    def coverage(self, sets, y) -> float:
        return empirical_coverage(sets, _label_index(self.model.classes_, y))


# --------------------------------------------------------------------------- #
# Mondrian (group-conditional) conformal prediction
# --------------------------------------------------------------------------- #
class MondrianCP:
    """Per-group split conformal prediction (one threshold per subject).

    Each group's threshold is computed from that group's calibration block
    only, so the coverage guarantee holds *within each subject*, provided the
    subject's calibration and test trials are exchangeable. This is the
    few-shot regime: a short labelled block from the new patient replaces
    any assumption about how they relate to the source subjects.

    A group with ``n`` calibration trials needs ``n >= ceil(1 / alpha) - 1`` for
    a finite threshold; below that every label is kept (valid but uninformative).

    Re-centering
    ------------
    With a model fitted with ``recenter=True``, a subject's calibration and
    test trials must be re-centered with the same reference. Calling
    ``calibrate`` and ``predict_set`` on separate arrays would re-center the two
    blocks separately and break exchangeability. Compute the scores of the
    whole subject in one call instead, then use the ``*_from_scores`` methods::

        S = score(model, X_subject)                       # one call, one reference
        cp.calibrate_from_scores(S[cal, y[cal]], groups[cal])
        sets = cp.predict_set_from_scores(S[test], groups[test])

    Attributes
    ----------
    qhat_ : dict
        ``{group: threshold}``.
    n_cal_ : dict
        ``{group: number of calibration trials}``.
    """

    def __init__(self, model, score: NonconformityScore | None = None, alpha: float = 0.2):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.model = model
        self.score = LAC() if score is None else score
        self.alpha = alpha

    @staticmethod
    def _key(g):
        return g.item() if hasattr(g, "item") else g

    def calibrate(self, X, y, groups, **score_kwargs) -> "MondrianCP":
        if getattr(self.model, "recenter", False):
            raise ValueError(
                "The model re-centers its inputs. Score each subject's calibration and test "
                "trials in one call and use calibrate_from_scores / predict_set_from_scores."
            )
        y_idx = _label_index(self.model.classes_, y)
        S = self.score(self.model, X, **score_kwargs)
        return self.calibrate_from_scores(true_label_scores(S, y_idx), groups)

    def calibrate_from_scores(self, cal_scores, groups) -> "MondrianCP":
        cal_scores = np.asarray(cal_scores, dtype=float).ravel()
        groups = np.asarray(groups)
        if groups.shape[0] != cal_scores.shape[0]:
            raise ValueError("Need one group label per calibration trial.")
        self.qhat_, self.n_cal_ = {}, {}
        for g in np.unique(groups):
            s = cal_scores[groups == g]
            self.qhat_[self._key(g)] = split_quantile(s, self.alpha)
            self.n_cal_[self._key(g)] = int(s.size)
        return self

    def _check_calibrated(self):
        if not hasattr(self, "qhat_"):
            raise RuntimeError("Call calibrate() before predicting.")

    def predict_set(self, X, groups, **score_kwargs) -> np.ndarray:
        self._check_calibrated()
        if getattr(self.model, "recenter", False):
            raise ValueError(
                "The model re-centers its inputs; use predict_set_from_scores with scores "
                "computed jointly with the calibration trials."
            )
        return self.predict_set_from_scores(self.score(self.model, X, **score_kwargs), groups)

    def predict_set_from_scores(self, S, groups) -> np.ndarray:
        self._check_calibrated()
        S = np.asarray(S)
        groups = np.asarray(groups)
        if groups.shape[0] != S.shape[0]:
            raise ValueError("Need one group label per test trial.")
        q = np.empty(S.shape[0])
        for g in np.unique(groups):
            key = self._key(g)
            if key not in self.qhat_:
                raise ValueError(f"Group {key!r} has no calibration block.")
            q[groups == g] = self.qhat_[key]
        return S <= q[:, None]

    def set_labels(self, sets) -> list:
        return [self.model.classes_[row] for row in np.asarray(sets, dtype=bool)]

    def coverage(self, sets, y) -> float:
        return empirical_coverage(sets, _label_index(self.model.classes_, y))


# --------------------------------------------------------------------------- #
# Adaptive conformal inference (online, cued sessions)
# --------------------------------------------------------------------------- #
def aci_coverage_bound(n_trials: int, gamma: float, alpha_init: float) -> float:
    """Gibbs & Candes (2021) bound on ``| (1/T) sum_t err_t - alpha |``.

    ``(max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)``. It holds for every
    sequence of trials, with no assumption on how the data are generated.
    """
    return (max(alpha_init, 1.0 - alpha_init) + gamma) / (gamma * n_trials)


class AdaptiveCI:
    """Adaptive conformal inference (Gibbs & Candes, 2021) for cued sessions.

    In cued rehabilitation the true label (the cue) is known as soon as the
    trial ends, so the predictor can correct itself online. It keeps a working
    miscoverage level ``alpha_t`` and after each trial sets

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),    err_t = 1{y_t not in C_t}.

    A miss lowers ``alpha_t`` (the next sets get larger); a hit raises it
    slightly (the next sets get smaller). The set at trial ``t`` is
    ``{y : s(x_t, y) <= Q_t}`` where ``Q_t`` is the conformal quantile of the
    stored scores at level ``alpha_t``; ``alpha_t <= 0`` gives the full label
    set and ``alpha_t >= 1`` the empty set.

    Guarantee (no exchangeability needed, any drift, any decoder):

        | (1/T) sum_t err_t - alpha |  <=  (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T).

    This is a long-run average. It does not bound the coverage of any single
    trial or short window; a larger ``gamma`` tracks drift faster but makes
    ``alpha_t``, and therefore the set size, swing more.

    Parameters
    ----------
    alpha : float in (0, 1)
        Target miscoverage.
    gamma : float > 0, default 0.01
        Step size.
    init_scores : array-like, optional
        True-label scores to start from (e.g. from held-out source subjects).
        Empty means the first trials get the full label set until enough
        scores have arrived.
    window : int, optional
        Keep only the most recent ``window`` scores. ``None`` keeps all.

    Attributes
    ----------
    alpha_t : float
        Current working level.
    history_ : dict of lists
        ``alpha_t`` (level used at each trial), ``qhat``, ``err``.
    """

    def __init__(self, alpha: float = 0.2, gamma: float = 0.01, init_scores=(), window: int | None = None):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if window is not None and window < 1:
            raise ValueError("window must be >= 1 or None.")
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.alpha_t = alpha
        self.scores_ = [float(v) for v in np.asarray(init_scores, dtype=float).ravel()]
        if window is not None:
            self.scores_ = self.scores_[-window:]
        self.history_ = {"alpha_t": [], "qhat": [], "err": []}

    # -- one trial --------------------------------------------------------- #
    def threshold(self) -> float:
        """Current score threshold ``Q_t`` (``+inf`` = keep all, ``-inf`` = keep none)."""
        if self.alpha_t <= 0.0:
            return float("inf")
        if self.alpha_t >= 1.0:
            return float("-inf")
        return split_quantile(self.scores_, self.alpha_t)

    def predict_set(self, s_all) -> np.ndarray:
        """Prediction set for one trial from its scores ``s_all`` (n_classes,)."""
        return np.asarray(s_all, dtype=float) <= self.threshold()

    def update(self, set_mask, y_idx: int, s_true: float | None = None) -> float:
        """Reveal the cue for the trial just predicted and adapt.

        Parameters
        ----------
        set_mask : bool array (n_classes,)
            The set returned by :meth:`predict_set` for this trial.
        y_idx : int
            Class index of the true label.
        s_true : float, optional
            True-label score of the trial; appended to the stored scores.

        Returns
        -------
        err : float
            1.0 if the true label was outside the set, else 0.0.
        """
        err = float(not bool(np.asarray(set_mask, dtype=bool)[y_idx]))
        self.history_["alpha_t"].append(self.alpha_t)
        self.history_["err"].append(err)
        self.alpha_t += self.gamma * (self.alpha - err)
        if s_true is not None:
            self.scores_.append(float(s_true))
            if self.window is not None and len(self.scores_) > self.window:
                del self.scores_[: len(self.scores_) - self.window]
        return err

    def step(self, s_all, y_idx: int) -> np.ndarray:
        """Predict one trial, then update with its label. Returns the set."""
        s_all = np.asarray(s_all, dtype=float)
        self.history_["qhat"].append(self.threshold())
        sets = s_all <= self.history_["qhat"][-1]
        self.update(sets, y_idx, s_all[y_idx])
        return sets

    # -- a whole session --------------------------------------------------- #
    def run(self, S, y_idx) -> dict:
        """Process trials in the order given.

        Parameters
        ----------
        S : ndarray, shape (T, n_classes)
            Scores of every candidate label, in chronological order.
        y_idx : int ndarray, shape (T,)

        Returns
        -------
        dict with ``sets`` (T, n_classes) bool, ``covered`` (T,) bool,
        ``alpha_t`` (T,) level used at each trial, ``qhat`` (T,).
        """
        S = np.asarray(S, dtype=float)
        y_idx = np.asarray(y_idx, dtype=int)
        start = len(self.history_["err"])
        sets = np.array([self.step(S[t], y_idx[t]) for t in range(len(y_idx))], dtype=bool).reshape(
            len(y_idx), S.shape[1])
        return {
            "sets": sets,
            "covered": sets[np.arange(len(y_idx)), y_idx],
            "alpha_t": np.asarray(self.history_["alpha_t"][start:]),
            "qhat": np.asarray(self.history_["qhat"][start:]),
        }

    def coverage_bound(self, n_trials: int | None = None) -> float:
        """Bound on the long-run coverage error after ``n_trials`` (default: trials seen)."""
        n = len(self.history_["err"]) if n_trials is None else n_trials
        return aci_coverage_bound(n, self.gamma, self.alpha) if n > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Conformal risk control for feedback
# --------------------------------------------------------------------------- #
def feedback_risk(conf, correct, lam: float) -> float:
    """Empirical joint rate ``mean(conf >= lam and wrong)``."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    return float(np.mean((conf >= lam) & ~correct))


def crc_feedback_threshold(conf, correct, alpha: float, grid=None) -> float:
    """Confidence threshold for giving feedback, by conformal risk control.

    Feedback is given on a trial when ``conf >= lambda``. The loss

        L(lambda) = 1{conf >= lambda and the decoded label is wrong}

    is the event "feedback was given and it was wrong". It lies in [0, 1] and
    does not increase with ``lambda``, so conformal risk control (Angelopoulos
    et al., 2024) applies: with ``n`` calibration trials, return the smallest
    grid value with

        n / (n + 1) * Rhat(lambda) + 1 / (n + 1)  <=  alpha.

    The smallest valid threshold gives feedback as often as the constraint
    allows. If calibration and test trials are exchangeable then

        P(feedback given AND wrong)  <=  alpha

    on a new trial, in expectation over the calibration set.

    This bounds the *joint* probability, per trial attempted. It does not
    bound the error rate among the trials that receive feedback, which is
    larger by a factor ``1 / P(feedback)``.

    Parameters
    ----------
    conf : array-like, shape (n,)
        Decoder confidence of each calibration trial (e.g. the largest class
        probability). Higher means more confident.
    correct : array-like of bool, shape (n,)
        Whether the decoded label of each calibration trial was right.
    alpha : float in (0, 1)
        Tolerated rate of wrong feedback.
    grid : array-like, optional
        Candidate thresholds. Must be chosen without looking at the
        calibration data. Default: 201 evenly spaced values on [0, 1].

    Returns
    -------
    lam : float
        Threshold, or ``+inf`` (never give feedback) when no grid value
        qualifies, which always happens when ``n < 1 / alpha - 1``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    conf = np.asarray(conf, dtype=float).ravel()
    correct = np.asarray(correct, dtype=bool).ravel()
    if conf.shape != correct.shape:
        raise ValueError("conf and correct must have the same length.")
    if np.isnan(conf).any():
        raise ValueError("conf contains NaN.")
    grid = np.linspace(0.0, 1.0, 201) if grid is None else np.sort(np.asarray(grid, dtype=float).ravel())
    n = conf.size
    wrong_conf = np.sort(conf[~correct])
    # Number of wrong trials with conf >= lam, for every lam at once.
    n_bad = wrong_conf.size - np.searchsorted(wrong_conf, grid, side="left")
    ok = (n_bad + 1.0) / (n + 1.0) <= alpha + 1e-12
    return float(grid[np.argmax(ok)]) if ok.any() else float("inf")
