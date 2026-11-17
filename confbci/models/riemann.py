"""Probabilistic Riemannian classifier for EEG covariance matrices.

Everything here uses the affine-invariant Riemannian metric on the manifold of
symmetric positive definite (SPD) matrices:

    delta_R(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F

Contents
--------
- small batched SPD helpers (``sqrtm``, ``invsqrtm``, ``expm``, ``logm``),
- ``sq_distance_to_means``: squared distances from many trials to a few class means,
- ``recenter``: Riemannian re-centering  C~ = Cbar_s^{-1/2} C Cbar_s^{-1/2},
- ``fit_temperature``: temperature fitted by held-out negative log-likelihood,
- ``ProbMDM``: minimum distance to mean with softmax probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

try:  # pyriemann >= 0.9
    from pyriemann.geometry.mean import mean_riemann
except ImportError:  # pyriemann 0.6 - 0.8
    from pyriemann.utils.mean import mean_riemann

__all__ = [
    "sqrtm",
    "invsqrtm",
    "expm",
    "logm",
    "sq_distance_to_means",
    "recenter",
    "fit_temperature",
    "ProbMDM",
]


# --------------------------------------------------------------------------- #
# Batched SPD helpers (operate on arrays of shape (..., c, c))
# --------------------------------------------------------------------------- #
def _sym(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + np.swapaxes(X, -1, -2))


def _eig_apply(X: np.ndarray, fun) -> np.ndarray:
    """Apply ``fun`` to the eigenvalues of symmetric matrices."""
    w, V = np.linalg.eigh(_sym(X))
    return (V * fun(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def sqrtm(X: np.ndarray) -> np.ndarray:
    """Matrix square root of SPD matrices."""
    return _eig_apply(X, np.sqrt)


def invsqrtm(X: np.ndarray) -> np.ndarray:
    """Inverse matrix square root of SPD matrices."""
    return _eig_apply(X, lambda w: 1.0 / np.sqrt(w))


def expm(X: np.ndarray) -> np.ndarray:
    """Matrix exponential of symmetric matrices (maps tangent space -> SPD)."""
    return _eig_apply(X, np.exp)


def logm(X: np.ndarray) -> np.ndarray:
    """Matrix logarithm of SPD matrices (maps SPD -> tangent space)."""
    return _eig_apply(X, np.log)


def _check_covs(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[-1] != X.shape[-2]:
        raise ValueError(
            f"Expected covariances of shape (n_trials, n_channels, n_channels), got {X.shape}."
        )
    return X


def sq_distance_to_means(covs: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Squared affine-invariant distances from each trial to each class mean.

    Parameters
    ----------
    covs : ndarray, shape (n_trials, c, c)
    means : ndarray, shape (n_classes, c, c)

    Returns
    -------
    D2 : ndarray, shape (n_trials, n_classes)
        ``D2[i, k] = delta_R(covs[i], means[k]) ** 2``.
    """
    covs = _check_covs(covs)
    means = _check_covs(means)
    out = np.empty((covs.shape[0], means.shape[0]))
    for k, M in enumerate(means):
        W = invsqrtm(M)
        eig = np.linalg.eigvalsh(_sym(W @ covs @ W))
        if np.any(eig <= 0):
            raise ValueError("Non-positive eigenvalue found: inputs must be SPD.")
        out[:, k] = np.sum(np.log(eig) ** 2, axis=-1)
    return out


def recenter(covs, groups=None, return_references: bool = False):
    """Riemannian re-centering: move each group's mean to the identity.

    For every group ``s`` (a subject or a session) with Riemannian mean ``Cbar_s``,

        C~ = Cbar_s^{-1/2} . C . Cbar_s^{-1/2}

    This is unsupervised (it never looks at labels) and removes the
    "translation" part of between-subject shift. A subject-specific rotation
    can remain; re-centering does not correct that.

    Parameters
    ----------
    covs : ndarray, shape (n_trials, c, c)
    groups : array-like of shape (n_trials,), optional
        Group label of each trial. ``None`` treats all trials as one group.
    return_references : bool
        Also return ``{group: Cbar_s}``.
    """
    covs = _check_covs(covs)
    groups = np.zeros(len(covs), dtype=int) if groups is None else np.asarray(groups)
    if groups.shape[0] != covs.shape[0]:
        raise ValueError("groups must have one entry per trial.")
    out = np.empty_like(covs)
    refs = {}
    for g in np.unique(groups):
        idx = groups == g
        ref = mean_riemann(covs[idx])
        W = invsqrtm(ref)
        out[idx] = _sym(W @ covs[idx] @ W)
        refs[g.item() if hasattr(g, "item") else g] = ref
    return (out, refs) if return_references else out


# --------------------------------------------------------------------------- #
# Temperature
# --------------------------------------------------------------------------- #
def _nll(D2: np.ndarray, y_idx: np.ndarray, tau: float) -> float:
    logits = -D2 / tau
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y_idx)), y_idx]))


def fit_temperature(D2: np.ndarray, y_idx: np.ndarray, span: float = 1e4) -> float:
    """Temperature that minimises held-out negative log-likelihood.

    The model is ``p(y=k | C) = softmax_k(-D2[:, k] / tau)``. The NLL is convex
    in ``1 / tau``, hence unimodal in ``log tau``, so a bounded 1-D search finds
    the global minimum inside the bracket.

    The bracket is ``[m / span, m * span]`` with ``m`` the median squared
    distance. If the optimum sits at the upper end the data carry no class
    information (probabilities go to uniform); at the lower end the classes are
    separable on the held-out split.
    """
    D2 = np.asarray(D2, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    scale = float(np.median(D2))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    lo, hi = np.log(scale / span), np.log(scale * span)
    res = minimize_scalar(
        lambda log_tau: _nll(D2, y_idx, np.exp(log_tau)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(np.exp(res.x))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class ProbMDM(ClassifierMixin, BaseEstimator):
    """Minimum distance to mean with calibrated class probabilities.

    .. math::

        p(y = k \\mid C) = \\frac{\\exp(-\\delta_R^2(C, M_k) / \\tau)}
                              {\\sum_j \\exp(-\\delta_R^2(C, M_j) / \\tau)}

    ``M_k`` are Riemannian class means and ``tau`` is fitted by minimising the
    negative log-likelihood on a split that is held out from the class-mean
    estimation.

    Parameters
    ----------
    recenter : bool, default=True
        Apply Riemannian re-centering per group before fitting and predicting.
        With ``groups=None`` all trials passed in one call are one group.
    tau_holdout : float, default=0.25
        Fraction of the training data held out to fit ``tau``. When ``groups``
        has at least four distinct values, whole groups (subjects) are held
        out; otherwise the split is stratified over trials.
    random_state : int or None
        Seed for the held-out split.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
    covmeans_ : ndarray, shape (n_classes, c, c)
        Class means, estimated on the non-held-out part only.
    tau_ : float
    holdout_nll_ : float
        NLL on the held-out split at ``tau_``.

    Notes
    -----
    Re-centering at prediction time is transductive: it uses the unlabeled
    trials passed to ``predict_proba``. For conformal prediction, calibration
    and test trials of a subject must be re-centered with the *same* reference
    (pass them with the same group label, or in the same call), otherwise their
    scores are not exchangeable.
    """

    def __init__(self, recenter: bool = True, tau_holdout: float = 0.25, random_state=0):
        self.recenter = recenter
        self.tau_holdout = tau_holdout
        self.random_state = random_state

    # -- helpers ----------------------------------------------------------- #
    def _prepare(self, X, groups):
        X = _check_covs(X)
        return recenter(X, groups) if self.recenter else X

    def _holdout_split(self, y_idx, groups):
        n = len(y_idx)
        dummy = np.zeros(n)
        if groups is not None and len(np.unique(groups)) >= 4:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=self.tau_holdout, random_state=self.random_state
            )
            return next(splitter.split(dummy, y_idx, groups))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.tau_holdout, random_state=self.random_state
        )
        return next(splitter.split(dummy, y_idx))

    # -- API --------------------------------------------------------------- #
    def fit(self, X, y, groups=None):
        """Fit class means and temperature.

        Parameters
        ----------
        X : ndarray, shape (n_trials, c, c)
            SPD covariance matrices (e.g. from ``pyriemann.estimation.Covariances``).
        y : array-like, shape (n_trials,)
        groups : array-like, shape (n_trials,), optional
            Subject (or session) of each trial, used for re-centering and for
            the subject-level held-out split.
        """
        if not 0.0 < self.tau_holdout < 1.0:
            raise ValueError("tau_holdout must be in (0, 1).")
        y = np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        Xc = self._prepare(X, groups)
        if len(y) != len(Xc):
            raise ValueError("X and y must have the same number of trials.")

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Need at least two classes.")

        fit_idx, hold_idx = self._holdout_split(y_idx, groups)
        if len(np.unique(y_idx[fit_idx])) < len(self.classes_):
            raise ValueError("A class is missing from the mean-estimation split.")

        self.covmeans_ = np.stack(
            [mean_riemann(Xc[fit_idx][y_idx[fit_idx] == k]) for k in range(len(self.classes_))]
        )
        D2_hold = sq_distance_to_means(Xc[hold_idx], self.covmeans_)
        self.tau_ = fit_temperature(D2_hold, y_idx[hold_idx])
        self.holdout_nll_ = _nll(D2_hold, y_idx[hold_idx], self.tau_)
        self.n_features_in_ = Xc.shape[-1]
        return self

    def sq_distances(self, X, groups=None) -> np.ndarray:
        """Squared Riemannian distances to the class means, shape (n_trials, n_classes)."""
        return sq_distance_to_means(self._prepare(X, groups), self.covmeans_)

    def distances(self, X, groups=None) -> np.ndarray:
        """Riemannian distances to the class means, shape (n_trials, n_classes)."""
        return np.sqrt(self.sq_distances(X, groups))

    def decision_function(self, X, groups=None) -> np.ndarray:
        """Logits ``-delta_R^2 / tau``, shape (n_trials, n_classes)."""
        return -self.sq_distances(X, groups) / self.tau_

    def predict_proba(self, X, groups=None) -> np.ndarray:
        """Class probabilities, shape (n_trials, n_classes), rows sum to 1."""
        return softmax(self.decision_function(X, groups), axis=1)

    def predict(self, X, groups=None) -> np.ndarray:
        """Hard labels (the nearest class mean)."""
        return self.classes_[np.argmin(self.sq_distances(X, groups), axis=1)]

    def nll(self, X, y, groups=None) -> float:
        """Mean negative log-likelihood of labels ``y``."""
        y_idx = np.searchsorted(self.classes_, np.asarray(y))
        return _nll(self.sq_distances(X, groups), y_idx, self.tau_)
