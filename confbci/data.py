"""Loading motor-imagery datasets from MOABB as per-subject covariance matrices.

MOABB (and MNE underneath it) is imported lazily so that the rest of the
package, and the test suite, work without it.

Each subject is cached to ``<cache_dir>/<dataset>_c<n_classes>_<fmin>-<fmax>_<estimator>/subject_<id>.npz``
so that leave-one-subject-out experiments never re-filter raw EEG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["covariances_from_epochs", "load_subject", "load_covariances"]

DEFAULT_CACHE = Path.home() / ".cache" / "confbci"


def covariances_from_epochs(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Trial covariance matrices from epochs of shape (n_trials, n_channels, n_times).

    ``estimator`` is passed to :class:`pyriemann.estimation.Covariances`;
    shrinkage estimators (``"oas"``, ``"lwf"``) keep the matrices well
    conditioned for short epochs.
    """
    from pyriemann.estimation import Covariances

    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected epochs of shape (n_trials, n_channels, n_times), got {X.shape}.")
    return Covariances(estimator=estimator).fit_transform(X)


def _cache_file(cache_dir, dataset: str, n_classes: int, fmin: float, fmax: float, estimator: str, subject: int) -> Path:
    tag = f"{dataset}_c{n_classes}_{fmin:g}-{fmax:g}_{estimator}"
    return Path(cache_dir).expanduser() / tag / f"subject_{subject}.npz"


def _moabb_objects(dataset: str, n_classes: int, fmin: float, fmax: float):
    import moabb.datasets
    from moabb.paradigms import LeftRightImagery, MotorImagery

    if not hasattr(moabb.datasets, dataset):
        raise ValueError(f"Unknown MOABB dataset {dataset!r}.")
    ds = getattr(moabb.datasets, dataset)()
    if n_classes == 2:
        paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)
    else:
        paradigm = MotorImagery(n_classes=n_classes, fmin=fmin, fmax=fmax)
    return ds, paradigm


def load_subject(
    dataset: str,
    subject: int,
    n_classes: int = 2,
    fmin: float = 8.0,
    fmax: float = 32.0,
    estimator: str = "oas",
    cache_dir=DEFAULT_CACHE,
    overwrite: bool = False,
) -> dict:
    """Covariances, labels and session ids of one subject (cached on disk).

    Returns a dict with keys ``covs`` (n_trials, c, c), ``y`` (n_trials,) of
    str, and ``session`` (n_trials,) of str, in the order MOABB returns them.
    """
    path = _cache_file(cache_dir, dataset, n_classes, fmin, fmax, estimator, subject)
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in ("covs", "y", "session")}

    ds, paradigm = _moabb_objects(dataset, n_classes, fmin, fmax)
    X, labels, meta = paradigm.get_data(dataset=ds, subjects=[subject])
    out = {
        "covs": covariances_from_epochs(X, estimator),
        "y": np.asarray(labels).astype(str),
        "session": meta["session"].to_numpy().astype(str),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def load_covariances(
    dataset: str = "BNCI2014_001",
    subjects=None,
    n_classes: int = 2,
    fmin: float = 8.0,
    fmax: float = 32.0,
    estimator: str = "oas",
    cache_dir=DEFAULT_CACHE,
    overwrite: bool = False,
):
    """Covariances for several subjects of a MOABB motor-imagery dataset.

    Parameters
    ----------
    dataset : str
        Name of a class in ``moabb.datasets``, e.g. ``"BNCI2014_001"``,
        ``"Liu2024"`` (50 acute stroke patients) or ``"Lee2019_MI"``.
    subjects : iterable of int, optional
        Defaults to every subject in the dataset.
    n_classes : int
        2 selects left- vs right-hand imagery; larger values use MOABB's
        generic ``MotorImagery`` paradigm.

    Returns
    -------
    X : ndarray, shape (n_trials, c, c)
    y : ndarray of str, shape (n_trials,)
    groups : ndarray of int, shape (n_trials,)
        Subject id of each trial, ready to pass as ``groups=`` to
        :class:`confbci.ProbMDM`.
    sessions : ndarray of str, shape (n_trials,)
    """
    if subjects is None:
        ds, _ = _moabb_objects(dataset, n_classes, fmin, fmax)
        subjects = ds.subject_list
    Xs, ys, gs, ss = [], [], [], []
    for subject in subjects:
        d = load_subject(dataset, int(subject), n_classes, fmin, fmax, estimator, cache_dir, overwrite)
        Xs.append(d["covs"])
        ys.append(d["y"])
        gs.append(np.full(len(d["y"]), int(subject)))
        ss.append(d["session"])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ss)
