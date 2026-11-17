"""Synthetic SPD matrices that stand in for EEG trial covariances.

Trials are drawn from a Gaussian defined on the SPD manifold around a known
class mean ``M``:

    S ~ N(0, sigma^2 I)   in the tangent space at the identity,
    C = M^{1/2} expm(S) M^{1/2}.

"Gaussian in the tangent space" is taken with respect to the affine-invariant
metric: the coordinates of ``S`` in an orthonormal basis of the symmetric
matrices are i.i.d. ``N(0, sigma^2)``. Consequences used by the tests:

- the Riemannian (Frechet) mean of the distribution is exactly ``M``;
- ``delta_R(C, M) = ||S||_F``, so ``delta_R^2 / sigma^2`` is chi-squared with
  ``c (c + 1) / 2`` degrees of freedom and
  ``E[delta_R^2(C, M)] = sigma^2 c (c + 1) / 2``.

This is the *wrapped* (log-normal) Gaussian. It is not the maximum-entropy
Riemannian Gaussian of Said et al. (2017), whose density is proportional to
``exp(-delta_R^2 / (2 sigma^2))`` with respect to the Riemannian volume; the two
agree for small ``sigma``. The wrapped form is used because it can be sampled
exactly. Conformal validity is distribution-free, so the choice does not
affect the coverage tests.
"""

from __future__ import annotations

import numpy as np

from .models.riemann import expm, sqrtm

__all__ = [
    "random_tangent",
    "sample_riemannian_gaussian",
    "make_class_means",
    "sample_dataset",
    "sample_multisubject",
]


def _rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def random_tangent(n_matrices: int, n_channels: int, sigma: float = 1.0, rng=None) -> np.ndarray:
    """Symmetric matrices with i.i.d. ``N(0, sigma^2)`` orthonormal coordinates.

    Diagonal entries have variance ``sigma^2`` and off-diagonal entries
    ``sigma^2 / 2``, so that ``||S||_F^2 / sigma^2 ~ chi2(c (c + 1) / 2)``.
    """
    rng = _rng(rng)
    A = rng.normal(scale=sigma, size=(n_matrices, n_channels, n_channels))
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def sample_riemannian_gaussian(n_matrices: int, mean: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    """Draw SPD matrices from a wrapped Gaussian centred on ``mean``.

    Parameters
    ----------
    n_matrices : int
    mean : ndarray, shape (c, c)
        SPD centre of mass.
    sigma : float
        Dispersion (standard deviation per tangent-space coordinate).

    Returns
    -------
    covs : ndarray, shape (n_matrices, c, c)
    """
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 2 or mean.shape[0] != mean.shape[1]:
        raise ValueError("mean must be a square matrix.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    S = random_tangent(n_matrices, mean.shape[0], sigma, rng)
    M_half = sqrtm(mean)
    C = M_half @ expm(S) @ M_half
    return 0.5 * (C + np.swapaxes(C, -1, -2))


def make_class_means(
    n_channels: int = 4,
    n_classes: int = 2,
    separation: float = 0.6,
    base_dispersion: float = 0.3,
    rng=None,
) -> np.ndarray:
    """Class means placed at Riemannian distance ``separation`` from a common base.

    A random SPD base ``B`` is drawn, then ``M_k = B^{1/2} expm(separation V_k) B^{1/2}``
    with ``V_k`` a random symmetric direction of unit Frobenius norm, so that
    ``delta_R(M_k, B) = separation`` for every class.

    Returns
    -------
    means : ndarray, shape (n_classes, c, c)
    """
    rng = _rng(rng)
    B = expm(random_tangent(1, n_channels, base_dispersion, rng))[0]
    B_half = sqrtm(B)
    V = random_tangent(n_classes, n_channels, 1.0, rng)
    V /= np.linalg.norm(V, axis=(-2, -1), keepdims=True)
    M = B_half @ expm(separation * V) @ B_half
    return 0.5 * (M + np.swapaxes(M, -1, -2))


def sample_dataset(n_matrices: int, means: np.ndarray, sigma: float, rng=None, priors=None):
    """I.i.d. labelled trials: ``y ~ Categorical(priors)``, ``C | y ~ Gaussian(M_y, sigma)``.

    Returns
    -------
    X : ndarray, shape (n_matrices, c, c)
    y : ndarray of int, shape (n_matrices,)
    """
    rng = _rng(rng)
    means = np.asarray(means, dtype=float)
    n_classes = means.shape[0]
    y = rng.choice(n_classes, size=n_matrices, p=priors)
    X = np.empty((n_matrices,) + means.shape[1:])
    for k in range(n_classes):
        idx = np.flatnonzero(y == k)
        X[idx] = sample_riemannian_gaussian(len(idx), means[k], sigma, rng)
    return X, y


def sample_multisubject(
    n_subjects: int,
    n_per_subject: int,
    means: np.ndarray,
    sigma: float,
    shift: float = 0.5,
    rng=None,
    drift: float = 0.0,
):
    """Trials from several "subjects" that differ by a congruence transform.

    Subject ``s`` observes ``A_s C A_s`` with ``A_s = expm(shift * H_s)`` and
    ``H_s`` a random symmetric matrix of unit Frobenius norm. A congruence
    preserves affine-invariant distances, so each subject has the same class
    geometry in a different place on the manifold. Re-centering removes this
    shift up to a subject-specific rotation.

    ``drift > 0`` adds within-subject non-stationarity: trial ``t`` of ``T`` is
    further transformed by ``expm(drift * (t / T) * D_s)`` with ``D_s`` a random
    unit-norm symmetric direction, so later trials move steadily away from
    earlier ones. Trials of each subject are returned in chronological order.

    Returns
    -------
    X : ndarray, shape (n_subjects * n_per_subject, c, c)
    y : ndarray of int
    groups : ndarray of int
        Subject index of each trial.
    """
    rng = _rng(rng)
    means = np.asarray(means, dtype=float)
    c = means.shape[-1]
    Xs, ys, gs = [], [], []
    for s in range(n_subjects):
        H = random_tangent(1, c, 1.0, rng)[0]
        A = expm(shift * H / np.linalg.norm(H))
        X, y = sample_dataset(n_per_subject, means, sigma, rng)
        if drift > 0:
            Dm = random_tangent(1, c, 1.0, rng)[0]
            Dm /= np.linalg.norm(Dm)
            t = (np.arange(n_per_subject) / n_per_subject)[:, None, None]
            G = expm(drift * t * Dm)
            X = G @ X @ G
        X = A @ X @ A
        Xs.append(0.5 * (X + np.swapaxes(X, -1, -2)))
        ys.append(y)
        gs.append(np.full(n_per_subject, s))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
