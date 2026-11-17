"""Nonconformity scores.

A score maps a fitted model and a batch of trials to an array ``S`` of shape
``(n_trials, n_classes)`` where ``S[i, k]`` is how badly label ``k`` fits trial
``i`` (larger = worse). The conformal calibrator only ever needs:

- the true-label scores ``S[i, y_i]`` on the calibration set, and
- the full array ``S`` on test trials, to build ``{k : S[i, k] <= qhat}``.

Extra keyword arguments (e.g. ``groups=``) are forwarded to the model.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NonconformityScore", "LAC", "RiemannianMargin", "true_label_scores"]


class NonconformityScore:
    """Base class: ``score(model, X, **kwargs) -> ndarray (n_trials, n_classes)``."""

    def __call__(self, model, X, **kwargs) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LAC(NonconformityScore):
    """Least-ambiguous classifier score: ``s(x, y) = 1 - p_hat(y | x)``.

    Works with any model exposing ``predict_proba``.
    """

    def __call__(self, model, X, **kwargs) -> np.ndarray:
        return 1.0 - np.asarray(model.predict_proba(X, **kwargs))


class RiemannianMargin(NonconformityScore):
    """Geometric margin on the SPD manifold.

    .. math:: s(C, y) = \\delta_R(C, M_y) - \\min_{k \\neq y} \\delta_R(C, M_k)

    Negative when ``y`` is the nearest class mean. It needs no probability
    model and no temperature, only a model exposing
    ``distances(X) -> (n_trials, n_classes)`` (e.g. :class:`confbci.ProbMDM`).
    """

    def __call__(self, model, X, **kwargs) -> np.ndarray:
        D = np.asarray(model.distances(X, **kwargs))
        n_classes = D.shape[1]
        if n_classes < 2:
            raise ValueError("RiemannianMargin needs at least two classes.")
        out = np.empty_like(D)
        for k in range(n_classes):
            out[:, k] = D[:, k] - np.delete(D, k, axis=1).min(axis=1)
        return out


def true_label_scores(S: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Pick ``S[i, y_idx[i]]`` for every trial."""
    S = np.asarray(S)
    y_idx = np.asarray(y_idx, dtype=int)
    return S[np.arange(S.shape[0]), y_idx]
