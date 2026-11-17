"""Validity tests for confbci (Week 1).

The central test is ``test_conditional_coverage_follows_beta``.

Theory. Fix a model. Draw ``n`` i.i.d. calibration trials, compute their
true-label scores and take the ``k``-th smallest as threshold ``qhat``, with
``k = ceil((n + 1)(1 - alpha))``. If the score distribution is continuous,
the coverage conditional on that calibration set,

    P(Y in C(X) | calibration set) = F(qhat),      F = CDF of the true-label score,

is the ``k``-th order statistic of ``n`` uniforms, i.e. it follows
``Beta(k, n + 1 - k) = Beta(n + 1 - l, l)`` with ``l = floor((n + 1) alpha)``.

Protocol. A classifier is fitted once on its own training data. A pool of
``N_POOL`` fresh synthetic trials is then split 1,000 times at random into
``n`` calibration trials and ``N_POOL - n`` test trials. Each split gives one
coverage value; the 1,000 values are compared with the Beta distribution by a
Kolmogorov-Smirnov test.

Two approximations are involved, both negligible at this pool size: test-set
coverage estimates the conditional coverage with binomial noise of variance
``< 0.25 / (N_POOL - n)``, and splitting a finite pool samples without
replacement. Together they inflate the standard deviation of the coverage by
well under 1% relative to the Beta distribution.

All random seeds are fixed, so the suite is deterministic. Under a correct
implementation each KS assertion would fail for about 1 seed in 1,000
(``KS_LEVEL``). The negative controls show the test has power: two plausible
but wrong quantile rules are rejected with p-values below 1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from confbci import LAC, ProbMDM, RiemannianMargin, SplitCP, conformal_rank, recenter, split_quantile
from confbci.conformal import empirical_coverage, set_sizes
from confbci.models.riemann import fit_temperature, sq_distance_to_means
from confbci.scores import true_label_scores
from confbci.sim import (
    make_class_means,
    random_tangent,
    sample_dataset,
    sample_multisubject,
    sample_riemannian_gaussian,
)

try:
    from pyriemann.geometry.mean import mean_riemann
except ImportError:  # pyriemann < 0.9
    from pyriemann.utils.mean import mean_riemann

N_CHANNELS = 4
SIGMA = 0.6          # dispersion chosen so the two classes overlap (accuracy ~ 0.8)
SEPARATION = 0.6
N_TRAIN = 400
N_POOL = 20_000
N_SPLITS = 1_000
KS_LEVEL = 1e-3      # significance level of each KS assertion

SCORES = {"lac": LAC(), "riemannian_margin": RiemannianMargin()}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def world():
    """Class means, a fitted model, and a pool of fresh trials with their scores."""
    rng = np.random.default_rng(20261117)
    means = make_class_means(N_CHANNELS, n_classes=2, separation=SEPARATION, rng=rng)
    X_train, y_train = sample_dataset(N_TRAIN, means, SIGMA, rng)
    model = ProbMDM(recenter=False, random_state=0).fit(X_train, y_train)
    X_pool, y_pool = sample_dataset(N_POOL, means, SIGMA, rng)  # independent of training
    S = {name: score(model, X_pool) for name, score in SCORES.items()}
    return {"means": means, "model": model, "X": X_pool, "y": y_pool, "S": S,
            "X_train": X_train, "y_train": y_train}


def _coverages_over_splits(S, y, n_cal, alpha, seed, threshold_fn=None):
    """Coverage on the rest of the pool for N_SPLITS random calibration sets."""
    rng = np.random.default_rng(seed)
    s_true = true_label_scores(S, y)
    cp = SplitCP(model=None, alpha=alpha)
    out = np.empty(N_SPLITS)
    for t in range(N_SPLITS):
        perm = rng.permutation(len(y))
        cal, test = perm[:n_cal], perm[n_cal:]
        if threshold_fn is None:
            cp.calibrate_from_scores(s_true[cal])
            sets = cp.predict_set_from_scores(S[test])
        else:
            sets = S[test] <= threshold_fn(s_true[cal], alpha)
        out[t] = empirical_coverage(sets, y[test])
    return out


def _beta(n, alpha):
    """Beta(n + 1 - l, l) with l = floor((n + 1) alpha)."""
    l = int(np.floor(np.round((n + 1) * alpha, 9)))
    assert l >= 1, "calibration set too small for this alpha"
    return stats.beta(n + 1 - l, l)


# --------------------------------------------------------------------------- #
# 1. split_quantile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 9, 10, 19, 50, 99, 100, 101, 500, 1000])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.8, 0.9])
def test_rank_matches_floor_form(n, alpha):
    """k = ceil((n+1)(1-alpha)) equals n + 1 - floor((n+1) alpha), without float drift."""
    from fractions import Fraction

    a = Fraction(str(alpha))
    exact_l = ((n + 1) * a).numerator // ((n + 1) * a).denominator
    assert conformal_rank(n, alpha) == n + 1 - exact_l


def test_split_quantile_is_exact_order_statistic():
    rng = np.random.default_rng(0)
    for n, alpha in [(10, 0.2), (99, 0.1), (100, 0.2), (37, 0.33), (500, 0.05)]:
        scores = rng.normal(size=n)
        k = conformal_rank(n, alpha)
        assert split_quantile(scores, alpha) == np.sort(scores)[k - 1]
        assert split_quantile(rng.permutation(scores), alpha) == np.sort(scores)[k - 1]


def test_split_quantile_known_values():
    scores = np.arange(1, 10)                      # n = 9
    assert split_quantile(scores, 0.1) == 9        # k = ceil(10 * 0.9) = 9
    assert split_quantile(scores, 0.2) == 8        # k = 8
    assert split_quantile(scores, 0.5) == 5        # k = 5


def test_split_quantile_infinite_when_calibration_too_small():
    assert split_quantile(np.arange(8.0), 0.1) == np.inf     # k = 9 > n = 8
    assert split_quantile(np.array([]), 0.2) == np.inf


def test_split_quantile_rejects_bad_input():
    with pytest.raises(ValueError):
        split_quantile([0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        split_quantile([0.1, 0.2], 1.0)
    with pytest.raises(ValueError):
        split_quantile([0.1, np.nan], 0.2)


# --------------------------------------------------------------------------- #
# 2. The coverage guarantee
# --------------------------------------------------------------------------- #
def test_scores_have_no_ties(world):
    """The Beta law needs continuous scores; check the pool has no ties."""
    for name, S in world["S"].items():
        s_true = true_label_scores(S, world["y"])
        assert len(np.unique(s_true)) == len(s_true), f"ties in {name} scores"


@pytest.mark.parametrize("score_name", list(SCORES))
@pytest.mark.parametrize("n_cal", [50, 99, 400])
@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_conditional_coverage_follows_beta(world, score_name, n_cal, alpha):
    """Coverage given the calibration set ~ Beta(n + 1 - l, l), l = floor((n + 1) alpha)."""
    cov = _coverages_over_splits(world["S"][score_name], world["y"], n_cal, alpha, seed=n_cal)
    target = _beta(n_cal, alpha)

    ks = stats.kstest(cov, target.cdf)
    assert ks.pvalue > KS_LEVEL, (
        f"coverage is not Beta({target.args[0]}, {target.args[1]}): "
        f"KS D={ks.statistic:.4f}, p={ks.pvalue:.2e}"
    )

    # Marginal coverage: E[Beta] = k / (n + 1), which lies in [1 - alpha, 1 - alpha + 1/(n+1)].
    se = target.std() / np.sqrt(N_SPLITS)
    assert abs(cov.mean() - target.mean()) < 4 * se
    assert 1 - alpha <= target.mean() <= 1 - alpha + 1 / (n_cal + 1) + 1e-12
    # Spread: the standard deviation across calibration sets matches the Beta law.
    assert cov.std(ddof=1) == pytest.approx(target.std(), rel=0.1)


def test_negative_control_uncorrected_quantile_is_rejected(world):
    """np.quantile(scores, 1 - alpha), with no finite-sample correction, under-covers."""
    n_cal, alpha = 100, 0.2
    naive = lambda s, a: np.quantile(s, 1 - a)
    cov = _coverages_over_splits(world["S"]["lac"], world["y"], n_cal, alpha, seed=1, threshold_fn=naive)
    assert stats.kstest(cov, _beta(n_cal, alpha).cdf).pvalue < 1e-6
    assert cov.mean() < 1 - alpha


def test_negative_control_higher_interpolation_is_off_by_one(world):
    """np.quantile(scores, k / n, method='higher') picks the (k + 1)-th order statistic.

    It is still valid (it over-covers) but it does not have the exact Beta law;
    it follows Beta(k + 1, n - k) instead.
    """
    n_cal, alpha = 100, 0.2
    k = conformal_rank(n_cal, alpha)

    def higher(s, a):
        return np.quantile(s, conformal_rank(len(s), a) / len(s), method="higher")

    cov = _coverages_over_splits(world["S"]["lac"], world["y"], n_cal, alpha, seed=2, threshold_fn=higher)
    assert stats.kstest(cov, stats.beta(k, n_cal + 1 - k).cdf).pvalue < 1e-6
    assert stats.kstest(cov, stats.beta(k + 1, n_cal - k).cdf).pvalue > KS_LEVEL


def test_api_path_matches_score_path(world):
    """SplitCP.calibrate/predict_set on matrices equals the precomputed-score path."""
    model, X, y = world["model"], world["X"], world["y"]
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(y))
    cal, test = perm[:150], perm[150:2150]
    for name, score in SCORES.items():
        cp = SplitCP(model, score=score, alpha=0.2).calibrate(X[cal], y[cal])
        sets = cp.predict_set(X[test])

        S = world["S"][name]
        ref = SplitCP(model, score=score, alpha=0.2).calibrate_from_scores(true_label_scores(S[cal], y[cal]))
        assert cp.qhat_ == pytest.approx(ref.qhat_, rel=1e-12, abs=1e-12)
        assert np.array_equal(sets, ref.predict_set_from_scores(S[test]))

        assert sets.shape == (len(test), 2) and sets.dtype == bool
        assert cp.coverage(sets, y[test]) == empirical_coverage(sets, y[test])
        assert set(np.unique(set_sizes(sets))) <= {0, 1, 2}
        labels = cp.set_labels(sets)
        assert all(len(lab) == size for lab, size in zip(labels, set_sizes(sets)))


def test_end_to_end_coverage_on_fresh_data(world):
    """Full pipeline with freshly sampled matrices (no shared pool): marginal coverage holds."""
    model, means = world["model"], world["means"]
    rng = np.random.default_rng(4)
    n_cal, alpha, n_test, n_rep = 80, 0.2, 400, 150
    hits = 0
    for _ in range(n_rep):
        X_cal, y_cal = sample_dataset(n_cal, means, SIGMA, rng)
        X_test, y_test = sample_dataset(n_test, means, SIGMA, rng)
        cp = SplitCP(model, score=RiemannianMargin(), alpha=alpha).calibrate(X_cal, y_cal)
        sets = cp.predict_set(X_test)
        hits += sets[np.arange(n_test), y_test].sum()
    cov = hits / (n_rep * n_test)
    target = _beta(n_cal, alpha)
    # Variance across replicates: Beta variance + binomial noise, averaged over n_rep.
    se = np.sqrt((target.var() + 0.25 / n_test) / n_rep)
    assert abs(cov - target.mean()) < 4 * se


def test_tiny_calibration_set_returns_full_sets(world):
    model, X, y = world["model"], world["X"], world["y"]
    cp = SplitCP(model, alpha=0.1).calibrate(X[:5], y[:5])   # k = 6 > n = 5
    assert cp.qhat_ == np.inf
    assert cp.predict_set(X[5:50]).all()


def test_string_labels_and_unknown_label(world):
    means = world["means"]
    rng = np.random.default_rng(5)
    X, y = sample_dataset(300, means, SIGMA, rng)
    names = np.array(["left_hand", "right_hand"])[y]
    model = ProbMDM(recenter=False).fit(X[:200], names[:200])
    cp = SplitCP(model, alpha=0.2).calibrate(X[200:260], names[200:260])
    sets = cp.predict_set(X[260:])
    assert 0.0 <= cp.coverage(sets, names[260:]) <= 1.0
    with pytest.raises(ValueError):
        cp.calibrate(X[200:260], np.array(["feet"] * 60))
    with pytest.raises(RuntimeError):
        SplitCP(model).predict_set(X[:3])


# --------------------------------------------------------------------------- #
# 3. ProbMDM
# --------------------------------------------------------------------------- #
def test_probmdm_probabilities(world):
    model, X = world["model"], world["X"][:500]
    P = model.predict_proba(X)
    assert P.shape == (500, 2)
    assert np.all(P >= 0) and np.allclose(P.sum(axis=1), 1.0)
    # Formula check: p_k = exp(-d_k^2 / tau) / sum_j exp(-d_j^2 / tau)
    D2 = sq_distance_to_means(X, model.covmeans_)
    ref = np.exp(-D2 / model.tau_)
    ref /= ref.sum(axis=1, keepdims=True)
    assert np.allclose(P, ref)
    assert np.array_equal(model.predict(X), model.classes_[P.argmax(axis=1)])


def test_probmdm_accuracy_and_temperature_scale(world):
    model, X, y = world["model"], world["X"], world["y"]
    acc = (model.predict(X) == y).mean()
    assert 0.70 < acc < 0.95          # overlapping classes, far from chance and from perfect
    # For this simulator the log-likelihood is -delta^2 / (2 sigma^2) up to a
    # Jacobian term, so the fitted temperature should be near 2 sigma^2.
    assert 0.5 * 2 * SIGMA**2 < model.tau_ < 2.0 * 2 * SIGMA**2


def test_temperature_minimises_heldout_nll():
    rng = np.random.default_rng(6)
    means = make_class_means(N_CHANNELS, 2, SEPARATION, rng=rng)
    X_fit, y_fit = sample_dataset(300, means, SIGMA, rng)
    X_hold, y_hold = sample_dataset(300, means, SIGMA, rng)
    class_means = np.stack([mean_riemann(X_fit[y_fit == k]) for k in range(2)])
    D2 = sq_distance_to_means(X_hold, class_means)
    tau = fit_temperature(D2, y_hold)

    def nll(t):
        logits = -D2 / t
        logits = logits - logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y_hold)), y_hold].mean()

    grid = tau * np.logspace(-2, 2, 81)
    assert all(nll(tau) <= nll(t) + 1e-9 for t in grid)
    assert nll(tau) < nll(1e-3) and nll(tau) < np.log(2)     # better than overconfident and than uniform


def test_temperature_calibration_improves_on_fresh_data(world):
    """Fitted tau gives lower NLL on unseen trials than a 10x sharper or 10x flatter one."""
    model, X, y = world["model"], world["X"], world["y"]
    D2 = model.sq_distances(X)

    def nll(t):
        logits = -D2 / t
        logits = logits - logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    assert nll(model.tau_) < nll(model.tau_ / 10)
    assert nll(model.tau_) < nll(model.tau_ * 10)
    assert model.nll(X, y) == pytest.approx(nll(model.tau_))


def test_recenter_moves_group_means_to_identity():
    rng = np.random.default_rng(7)
    means = make_class_means(N_CHANNELS, 2, SEPARATION, rng=rng)
    X, _, groups = sample_multisubject(3, 120, means, 0.4, shift=1.0, rng=rng)
    Xc, refs = recenter(X, groups, return_references=True)
    assert set(refs) == {0, 1, 2}
    for g in range(3):
        assert np.allclose(mean_riemann(Xc[groups == g]), np.eye(N_CHANNELS), atol=1e-6)
    assert np.all(np.linalg.eigvalsh(Xc) > 0)


def test_recentering_recovers_cross_subject_accuracy():
    """Leave-one-subject-out on shifted subjects: re-centering beats no re-centering."""
    means = make_class_means(N_CHANNELS, 2, SEPARATION, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(8, 200, means, SIGMA, shift=1.5, rng=np.random.default_rng(9))
    acc = {False: [], True: []}
    for s in range(8):
        tr, te = g != s, g == s
        for flag in (False, True):
            m = ProbMDM(recenter=flag, random_state=0).fit(X[tr], y[tr], groups=g[tr])
            acc[flag].append((m.predict(X[te], groups=g[te]) == y[te]).mean())
    assert np.mean(acc[True]) > np.mean(acc[False]) + 0.03
    assert np.mean(acc[True]) > 0.75


# --------------------------------------------------------------------------- #
# 4. Simulator
# --------------------------------------------------------------------------- #
def test_samples_are_spd():
    rng = np.random.default_rng(10)
    means = make_class_means(6, 3, 0.8, rng=rng)
    X, y = sample_dataset(500, means, 0.7, rng)
    assert X.shape == (500, 6, 6) and set(np.unique(y)) == {0, 1, 2}
    assert np.allclose(X, np.swapaxes(X, -1, -2))
    assert np.all(np.linalg.eigvalsh(X) > 0)


def test_class_means_are_at_requested_distance():
    rng = np.random.default_rng(11)
    c, sep = 5, 0.9
    # base_dispersion=0 makes the common base the identity.
    means = make_class_means(c, 4, sep, base_dispersion=0.0, rng=rng)
    d = np.sqrt(sq_distance_to_means(means, np.eye(c)[None]))[:, 0]
    assert np.allclose(d, sep)
    # Affine invariance of the distance helper.
    A = rng.normal(size=(c, c))
    base = A @ A.T + c * np.eye(c)
    X = sample_riemannian_gaussian(20, base, 0.3, rng)
    W = rng.normal(size=(c, c)) + 3 * np.eye(c)
    d1 = sq_distance_to_means(X, base[None])
    d2 = sq_distance_to_means(W @ X @ W.T, (W @ base @ W.T)[None])
    assert np.allclose(d1, d2)


def test_gaussian_distance_law():
    """delta_R^2(C, M) / sigma^2 ~ chi2(c (c + 1) / 2) for the wrapped Gaussian."""
    rng = np.random.default_rng(12)
    c, sigma = 4, 0.5
    M = make_class_means(c, 1, 0.0, rng=rng)[0]
    X = sample_riemannian_gaussian(5000, M, sigma, rng)
    d2 = sq_distance_to_means(X, M[None])[:, 0]
    dof = c * (c + 1) // 2
    assert stats.kstest(d2 / sigma**2, stats.chi2(dof).cdf).pvalue > KS_LEVEL
    assert d2.mean() == pytest.approx(sigma**2 * dof, rel=0.05)


def test_gaussian_frechet_mean_is_the_centre():
    rng = np.random.default_rng(13)
    M = make_class_means(4, 1, 0.0, rng=rng)[0]
    X = sample_riemannian_gaussian(20_000, M, 0.4, rng)
    est = mean_riemann(X)
    dist = np.sqrt(sq_distance_to_means(est[None], M[None]))[0, 0]
    assert dist < 0.02


def test_random_tangent_variances():
    S = random_tangent(50_000, 3, sigma=2.0, rng=np.random.default_rng(14))
    assert np.allclose(S, np.swapaxes(S, -1, -2))
    assert S[:, 0, 0].var() == pytest.approx(4.0, rel=0.05)      # diagonal: sigma^2
    assert S[:, 0, 1].var() == pytest.approx(2.0, rel=0.05)      # off-diagonal: sigma^2 / 2
