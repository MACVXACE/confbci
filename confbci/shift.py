"""Covariate-shift weights between source subjects and a target subject.

Weighted conformal prediction (Tibshirani et al., 2019) needs the density ratio

    w(x) = p_target(x) / p_source(x).

A probabilistic classifier trained to tell source trials from target trials
gives it without density estimation (Bayes' rule):

    w(x) = (n_s / n_t) * P(target | x) / P(source | x).

Only *unlabeled* target trials are needed. The correction is for covariate
shift alone: if P(y | x) differs between subjects, weighted conformal
prediction can still miss its coverage target.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models.riemann import _check_covs, invsqrtm, logm, mean_riemann

__all__ = ["tangent_vectors", "effective_sample_size", "DomainClassifierDensityRatio"]


def tangent_vectors(covs, reference) -> np.ndarray:
    """Tangent-space coordinates of SPD matrices at ``reference``.

    ``S = log(R^{-1/2} C R^{-1/2})`` is vectorised over its upper triangle with
    off-diagonal entries multiplied by sqrt(2), so that the Euclidean norm of
    the vector equals the Riemannian distance ``delta_R(C, R)``.

    Returns
    -------
    V : ndarray, shape (n_trials, c (c + 1) / 2)
    """
    covs = _check_covs(covs)
    W = invsqrtm(np.asarray(reference, dtype=float))
    S = logm(W @ covs @ W)
    c = S.shape[-1]
    iu = np.triu_indices(c)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return S[:, iu[0], iu[1]] * coef


def effective_sample_size(weights) -> float:
    """Kish effective sample size ``n_eff = (sum w)^2 / sum w^2``.

    Equals ``n`` for constant weights and approaches 1 when a single trial
    carries all the weight.
    """
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w**2)
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


class DomainClassifierDensityRatio(BaseEstimator):
    """Density-ratio weights from a source-vs-target logistic regression.

    Parameters
    ----------
    C : float, default=0.1
        Inverse L2 regularisation of the logistic regression. The tangent
        space has ``c (c + 1) / 2`` dimensions (253 for 22 channels), often
        more than the number of target trials, so an unregularised classifier
        separates the domains perfectly and the weights explode. Lower ``C``
        if ``n_eff_`` collapses.
    clip_percentile : float or None, default=99.0
        Weights are capped at this percentile of the weights of all trials seen
        in ``fit`` (source and target together). The same cap is applied to any
        trial passed to :meth:`weights`. The percentile is taken over both
        domains on purpose: capping target weights at the level of the source
        weights would, when the domains barely overlap, shrink every target
        weight to the size of a source weight and silently turn weighted
        conformal prediction back into the unweighted, invalid one.
    min_neff_ratio : float, default=0.1
        A warning is issued when ``n_eff_ / n_source`` falls below this.
    max_inf_mass : float, default=0.1
        A warning is issued when ``inf_mass_`` exceeds this.
    max_iter : int, default=2000

    Attributes
    ----------
    reference_ : ndarray, shape (c, c)
        Global Riemannian mean of source and target trials together.
    clip_value_ : float
        Cap applied to the weights (``inf`` if clipping is off).
    source_weights_ : ndarray, shape (n_source,)
        Clipped weights of the source trials used in ``fit``.
    n_eff_ : float
        Effective sample size of ``source_weights_``. This is the number of
        calibration trials the weighted quantile effectively rests on.
    n_eff_ratio_ : float
        ``n_eff_ / n_source``.
    inf_mass_ : float
        Median over target trials of ``w(x) / (sum_i w_i + w(x))``: the share of
        the weighted distribution that sits at ``+inf`` for a typical target
        trial. When it exceeds ``alpha`` the threshold is infinite and the
        prediction set is the full label set (valid, but no decision is made).
        Large values mean source and target barely overlap.
    domain_accuracy_ : float
        Training accuracy of the domain classifier (0.5 = no detectable
        shift, 1.0 = domains fully separable, weights unreliable).
    """

    def __init__(self, C: float = 0.1, clip_percentile: float | None = 99.0,
                 min_neff_ratio: float = 0.1, max_inf_mass: float = 0.1, max_iter: int = 2000):
        self.C = C
        self.clip_percentile = clip_percentile
        self.min_neff_ratio = min_neff_ratio
        self.max_inf_mass = max_inf_mass
        self.max_iter = max_iter

    def fit(self, X_source, X_target):
        """Fit on source trials (the conformal calibration set) and unlabeled target trials."""
        X_source, X_target = _check_covs(X_source), _check_covs(X_target)
        self.n_source_, self.n_target_ = len(X_source), len(X_target)
        if self.n_source_ == 0 or self.n_target_ == 0:
            raise ValueError("Need at least one source and one target trial.")

        X_all = np.concatenate([X_source, X_target])
        self.reference_ = mean_riemann(X_all)
        V = tangent_vectors(X_all, self.reference_)
        domain = np.r_[np.zeros(self.n_source_, dtype=int), np.ones(self.n_target_, dtype=int)]
        self.classifier_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter)
        ).fit(V, domain)
        self.domain_accuracy_ = float(self.classifier_.score(V, domain))

        raw_s, raw_t = self._raw_weights(X_source), self._raw_weights(X_target)
        self.clip_value_ = (
            float(np.percentile(np.concatenate([raw_s, raw_t]), self.clip_percentile))
            if self.clip_percentile is not None else float("inf")
        )
        self.source_weights_ = np.minimum(raw_s, self.clip_value_)
        w_t = np.minimum(raw_t, self.clip_value_)
        self.n_eff_ = effective_sample_size(self.source_weights_)
        self.n_eff_ratio_ = self.n_eff_ / self.n_source_
        self.inf_mass_ = float(np.median(w_t / (self.source_weights_.sum() + w_t)))
        if self.n_eff_ratio_ < self.min_neff_ratio or self.inf_mass_ > self.max_inf_mass:
            warnings.warn(
                f"Density-ratio weights are unstable: n_eff = {self.n_eff_:.1f} of "
                f"{self.n_source_} source trials, and a typical target trial puts "
                f"{self.inf_mass_:.0%} of the weighted distribution at +inf (domain classifier "
                f"accuracy {self.domain_accuracy_:.2f}). Source and target barely overlap, so "
                "weighted conformal sets will be wide or uninformative.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def _raw_weights(self, X) -> np.ndarray:
        V = tangent_vectors(X, self.reference_)
        log_odds = self.classifier_.decision_function(V)          # log P(target|x) / P(source|x)
        return (self.n_source_ / self.n_target_) * np.exp(np.clip(log_odds, -50.0, 50.0))

    def weights(self, X) -> np.ndarray:
        """``w(x) = (n_s / n_t) P(target | x) / P(source | x)``, clipped at ``clip_value_``."""
        return np.minimum(self._raw_weights(X), self.clip_value_)
