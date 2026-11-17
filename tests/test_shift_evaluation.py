"""Tests for Week 2: density-ratio weights, weighted and Mondrian conformal
prediction, and the leave-one-subject-out evaluation regimes.

All data are synthetic (``confbci.sim``), so every expected value is known.
Seeds are fixed; the suite is deterministic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from confbci import (
    LAC,
    DomainClassifierDensityRatio,
    LOSOConformalEvaluation,
    MondrianCP,
    ProbMDM,
    RiemannianMargin,
    SplitCP,
    WeightedCP,
    conformal_rank,
    effective_sample_size,
    set_metrics,
    split_quantile,
    summarize,
    weighted_quantile,
)
from confbci.models.riemann import sq_distance_to_means
from confbci.scores import true_label_scores
from confbci.shift import tangent_vectors
from confbci.sim import make_class_means, sample_dataset, sample_multisubject

C, SIGMA, ALPHA = 4, 0.6, 0.2


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(2026)
    means = make_class_means(C, 2, 0.6, rng=rng)
    X_train, y_train = sample_dataset(400, means, SIGMA, rng)
    model = ProbMDM(recenter=False).fit(X_train, y_train)
    X, y = sample_dataset(60_000, means, SIGMA, rng)
    S = LAC()(model, X)
    return {"means": means, "model": model, "X": X, "y": y, "S": S,
            "s_true": true_label_scores(S, y), "V": tangent_vectors(X, np.eye(C))}


# --------------------------------------------------------------------------- #
# weighted_quantile
# --------------------------------------------------------------------------- #
def test_weighted_quantile_reduces_to_split_quantile():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 300))
        alpha = float(rng.choice([0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.8]))
        s = rng.normal(size=n)
        assert weighted_quantile(s, np.ones(n), 1.0, alpha) == split_quantile(s, alpha)


def test_weighted_quantile_hand_example():
    # total mass 1 + 1 + 2 + 1 (test) = 5 -> cdf at scores 1, 2, 3 is 0.2, 0.4, 0.8; 0.2 at +inf
    s, w = [1.0, 2.0, 3.0], [1.0, 1.0, 2.0]
    assert weighted_quantile(s, w, 1.0, alpha=0.7) == 2.0     # needs cdf >= 0.3
    assert weighted_quantile(s, w, 1.0, alpha=0.6) == 2.0     # needs cdf >= 0.4 (boundary)
    assert weighted_quantile(s, w, 1.0, alpha=0.4) == 3.0     # needs cdf >= 0.6
    assert weighted_quantile(s, w, 1.0, alpha=0.1) == np.inf  # needs 0.9 > 0.8
    # a heavier test point pushes the threshold up
    assert weighted_quantile(s, w, 6.0, alpha=0.4) == np.inf
    # order of the calibration trials is irrelevant
    assert weighted_quantile(s[::-1], w[::-1], 1.0, alpha=0.4) == 3.0


def test_weighted_quantile_scale_invariant_and_validates():
    rng = np.random.default_rng(1)
    s, w = rng.normal(size=50), rng.uniform(0.1, 3, size=50)
    for alpha in (0.1, 0.2, 0.5):
        assert weighted_quantile(s, w, 1.3, alpha) == weighted_quantile(s, 7 * w, 7 * 1.3, alpha)
    with pytest.raises(ValueError):
        weighted_quantile(s, -w, 1.0, 0.2)
    with pytest.raises(ValueError):
        weighted_quantile(s, w[:10], 1.0, 0.2)
    with pytest.raises(ValueError):
        weighted_quantile(s, w, 1.0, 1.5)


def test_weighted_cp_matches_scalar_quantile(world):
    rng = np.random.default_rng(2)
    s = world["s_true"][:200]
    w_cal = rng.uniform(0.2, 4, size=200)
    w_test = rng.uniform(0.2, 4, size=50)
    cp = WeightedCP(world["model"], alpha=ALPHA).calibrate_from_scores(s, w_cal)
    q = cp.thresholds(w_test)
    assert np.array_equal(q, [weighted_quantile(s, w_cal, wt, ALPHA) for wt in w_test])
    sets = cp.predict_set_from_scores(world["S"][200:250], w_test)
    assert np.array_equal(sets, world["S"][200:250] <= q[:, None])


# --------------------------------------------------------------------------- #
# Covariate shift: weighted CP restores coverage, plain split CP does not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sign", [+1, -1])
def test_weighted_cp_restores_coverage_under_covariate_shift(world, sign):
    """Target trials are resampled with probability proportional to exp(theta . v(x)).

    P(y | x) is untouched, so this is pure covariate shift with a known density
    ratio. The shift direction joins the two class means, which changes how
    hard the target trials are.
    """
    model, X, y, S, s_true, V = (world[k] for k in ("model", "X", "y", "S", "s_true", "V"))
    d = tangent_vectors(world["means"], np.eye(C))
    theta = sign * 1.5 * (d[0] - d[1]) / np.linalg.norm(d[0] - d[1])
    w = np.exp(V @ theta)

    rng = np.random.default_rng(10 + sign)
    n_cal = n_tgt = 300
    cov = {"plain": [], "oracle": [], "estimated": []}
    for _ in range(60):
        idx = rng.permutation(len(y))
        cal, rest = idx[:n_cal], idx[n_cal:]
        tgt = rng.choice(rest, size=n_tgt, replace=False, p=w[rest] / w[rest].sum())
        rows = np.arange(n_tgt)

        plain = SplitCP(model, alpha=ALPHA).calibrate_from_scores(s_true[cal])
        cov["plain"].append(plain.predict_set_from_scores(S[tgt])[rows, y[tgt]].mean())

        oracle = WeightedCP(model, alpha=ALPHA).calibrate_from_scores(s_true[cal], w[cal])
        cov["oracle"].append(oracle.predict_set_from_scores(S[tgt], w[tgt])[rows, y[tgt]].mean())

        dr = DomainClassifierDensityRatio(C=1.0).fit(X[cal], X[tgt])
        est = WeightedCP(model, alpha=ALPHA).calibrate_from_scores(s_true[cal], dr.source_weights_)
        cov["estimated"].append(est.predict_set_from_scores(S[tgt], dr.weights(X[tgt]))[rows, y[tgt]].mean())

    mean = {k: float(np.mean(v)) for k, v in cov.items()}
    assert abs(mean["plain"] - (1 - ALPHA)) > 0.03, mean        # unweighted CP is off target
    assert abs(mean["oracle"] - (1 - ALPHA)) < 0.02, mean       # true weights fix it
    assert mean["oracle"] >= 1 - ALPHA - 0.01, mean
    assert abs(mean["estimated"] - (1 - ALPHA)) < 0.03, mean    # so do estimated weights
    assert abs(mean["estimated"] - (1 - ALPHA)) < abs(mean["plain"] - (1 - ALPHA)), mean


# --------------------------------------------------------------------------- #
# DomainClassifierDensityRatio
# --------------------------------------------------------------------------- #
def test_tangent_vector_norm_is_riemannian_distance(world):
    X = world["X"][:200]
    ref = world["means"][0]
    V = tangent_vectors(X, ref)
    assert V.shape == (200, C * (C + 1) // 2)
    assert np.allclose(np.linalg.norm(V, axis=1) ** 2, sq_distance_to_means(X, ref[None])[:, 0])


def test_effective_sample_size():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size([1.0, 0, 0, 0]) == pytest.approx(1)
    assert effective_sample_size([2.0, 2.0, 0, 0]) == pytest.approx(2)
    assert effective_sample_size(np.zeros(5)) == 0.0


def test_density_ratio_is_flat_without_shift(world):
    X = world["X"]
    dr = DomainClassifierDensityRatio().fit(X[:600], X[600:900])
    w = dr.weights(X[900:1500])
    assert w.mean() == pytest.approx(1.0, abs=0.1)
    assert dr.n_eff_ratio_ > 0.85 and dr.inf_mass_ < 0.01
    assert dr.domain_accuracy_ < 0.72            # close to the majority-class rate of 2/3


def test_density_ratio_recovers_known_log_linear_ratio(world):
    X, V = world["X"], world["V"]
    theta = np.zeros(V.shape[1]); theta[0] = 1.0
    w_true = np.exp(V @ theta)
    rng = np.random.default_rng(3)
    src = rng.choice(len(X), 3000, replace=False)
    rest = np.setdiff1d(np.arange(len(X)), src)
    tgt = rng.choice(rest, 3000, replace=False, p=w_true[rest] / w_true[rest].sum())
    dr = DomainClassifierDensityRatio(C=1.0, clip_percentile=None).fit(X[src], X[tgt])
    probe = rng.choice(len(X), 2000, replace=False)
    r = np.corrcoef(np.log(dr.weights(X[probe])), np.log(w_true[probe]))[0, 1]
    assert r > 0.95


def test_density_ratio_clipping_and_instability_warning(world):
    means = world["means"]
    X, _, g = sample_multisubject(2, 300, means, 0.3, shift=3.0, rng=np.random.default_rng(4))
    with pytest.warns(RuntimeWarning, match="barely overlap"):
        dr = DomainClassifierDensityRatio(C=100.0).fit(X[g == 0], X[g == 1])   # disjoint domains
    assert dr.domain_accuracy_ > 0.99 and dr.inf_mass_ > 0.5
    assert dr.source_weights_.max() <= dr.clip_value_
    assert dr.weights(X[g == 1]).max() <= dr.clip_value_      # same cap on target trials
    # No overlap: weighted CP must abstain on (nearly) every target trial rather than
    # quietly fall back to the unweighted threshold.
    cp = WeightedCP(world["model"], alpha=ALPHA).calibrate_from_scores(
        np.linspace(0, 1, 300), dr.source_weights_)
    assert np.mean(np.isinf(cp.thresholds(dr.weights(X[g == 1])))) > 0.9
    assert dr.n_eff_ == pytest.approx(effective_sample_size(dr.source_weights_))

    unclipped = DomainClassifierDensityRatio(C=1.0, clip_percentile=None, min_neff_ratio=0.0)
    Xs = world["X"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unclipped.fit(Xs[:500], Xs[500:1000])
    assert unclipped.clip_value_ == np.inf


# --------------------------------------------------------------------------- #
# MondrianCP
# --------------------------------------------------------------------------- #
def test_mondrian_thresholds_are_per_group(world):
    rng = np.random.default_rng(5)
    model, S, y = world["model"], world["S"][:600], world["y"][:600]
    groups = np.repeat(["a", "b", "c"], 200)
    S = S + np.array([0.0, 0.3, -0.2])[np.repeat([0, 1, 2], 200)][:, None]   # group-specific score shift
    s_true = true_label_scores(S, y)
    cal = rng.random(600) < 0.3

    cp = MondrianCP(model, alpha=ALPHA).calibrate_from_scores(s_true[cal], groups[cal])
    for g in "abc":
        m = cal & (groups == g)
        assert cp.qhat_[g] == split_quantile(s_true[m], ALPHA)
        assert cp.n_cal_[g] == m.sum()
    sets = cp.predict_set_from_scores(S[~cal], groups[~cal])
    for g in "abc":
        m = groups[~cal] == g
        assert np.array_equal(sets[m], S[~cal][m] <= cp.qhat_[g])
    with pytest.raises(ValueError, match="no calibration block"):
        cp.predict_set_from_scores(S[:3], np.array(["a", "b", "zzz"]))


def test_mondrian_gives_per_subject_coverage_where_pooling_fails(world):
    """Two 'subjects' whose scores differ by an offset: pooled CP over-covers one and
    under-covers the other; Mondrian CP is on target for both."""
    model, y = world["model"], world["y"]
    S0 = world["S"]
    rng = np.random.default_rng(6)
    n = 20_000
    groups = np.repeat([0, 1], n)
    S = np.concatenate([S0[:n], S0[n:2 * n] + 0.25])
    yy = y[: 2 * n]
    s_true = true_label_scores(S, yy)
    cov_pool, cov_mond = np.zeros(2), np.zeros(2)
    reps = 200
    for _ in range(reps):
        cal = np.concatenate([rng.choice(n, 100, replace=False), n + rng.choice(n, 100, replace=False)])
        test = np.setdiff1d(np.arange(2 * n), cal)
        pooled = SplitCP(model, alpha=ALPHA).calibrate_from_scores(s_true[cal]).predict_set_from_scores(S[test])
        mond = MondrianCP(model, alpha=ALPHA).calibrate_from_scores(s_true[cal], groups[cal]) \
            .predict_set_from_scores(S[test], groups[test])
        for g in (0, 1):
            m = groups[test] == g
            cov_pool[g] += pooled[m][np.arange(m.sum()), yy[test][m]].mean() / reps
            cov_mond[g] += mond[m][np.arange(m.sum()), yy[test][m]].mean() / reps
    theory = conformal_rank(100, ALPHA) / 101
    assert np.allclose(cov_mond, theory, atol=0.015), cov_mond
    assert cov_pool[1] < 1 - ALPHA - 0.05 < 1 - ALPHA + 0.05 < cov_pool[0], cov_pool


def test_mondrian_matrix_api_and_recentering_guard(world):
    model, X, y = world["model"], world["X"], world["y"]
    groups = np.repeat([1, 2], 150)
    cp = MondrianCP(model, score=RiemannianMargin(), alpha=ALPHA).calibrate(X[:300], y[:300], groups)
    sets = cp.predict_set(X[300:600], groups)
    assert sets.shape == (300, 2) and 0.6 < cp.coverage(sets, y[300:600]) <= 1.0

    Xm, ym, gm = sample_multisubject(4, 120, world["means"], SIGMA, shift=1.0, rng=np.random.default_rng(7))
    rc = ProbMDM(recenter=True).fit(Xm, ym, groups=gm)
    # Separate calls re-center calibration and test blocks with different references ...
    one_subject = Xm[gm == 0]
    joint = LAC()(rc, one_subject)
    separate = np.concatenate([LAC()(rc, one_subject[:20]), LAC()(rc, one_subject[20:])])
    assert not np.allclose(joint, separate)
    # ... so the matrix API refuses re-centering models.
    with pytest.raises(ValueError, match="re-centers"):
        MondrianCP(rc, alpha=ALPHA).calibrate(one_subject[:20], ym[gm == 0][:20], gm[gm == 0][:20])
    with pytest.raises(RuntimeError):
        MondrianCP(model).predict_set_from_scores(world["S"][:2], [1, 1])


# --------------------------------------------------------------------------- #
# Evaluation regimes
# --------------------------------------------------------------------------- #
def test_set_metrics():
    sets = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=bool)
    y = np.array([0, 0, 1, 1, 0])
    m = set_metrics(sets, y)
    assert m["coverage"] == pytest.approx(3 / 5)
    assert m["singleton_rate"] == pytest.approx(3 / 5)
    assert m["abstain_rate"] == pytest.approx(1 / 5)
    assert m["empty_rate"] == pytest.approx(1 / 5)
    assert m["false_feedback_rate"] == pytest.approx(1 / 5)
    assert m["selective_accuracy"] == pytest.approx(2 / 3)
    assert m["mean_set_size"] == pytest.approx(1.0) and m["n_test"] == 5
    assert np.isnan(set_metrics(np.ones((3, 2), bool), [0, 1, 0])["selective_accuracy"])


@pytest.fixture(scope="module")
def stationary_run():
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(12, 240, means, SIGMA, shift=1.0, rng=np.random.default_rng(9))
    labels = np.array(["left_hand", "right_hand"])[y]
    ev = LOSOConformalEvaluation(alpha=ALPHA, ks=(2, 4, 8, 16, 32), n_shuffles=20, random_state=0)
    records = ev.run(X, labels, g + 1)
    return records, summarize(records)


def _row(summary, **kw):
    hits = [r for r in summary if all(r[k] == v for k, v in kw.items())]
    assert len(hits) == 1, (kw, len(hits))
    return hits[0]


def test_evaluation_record_structure(stationary_run):
    records, summary = stationary_run
    r1 = [r for r in records if r["regime"] == "R1"]
    r2 = [r for r in records if r["regime"] == "R2"]
    assert len(r1) == 12 * 3 and {r["method"] for r in r1} == {"pooled", "weighted", "recentered"}
    assert len(r2) == 12 * 2 * 5 * 2                       # subjects x models x k x orders
    assert {r["order"] for r in r2} == {"chronological", "shuffled"}
    assert {r["k"] for r in r2} == {2, 4, 8, 16, 32}
    assert all(r["n_cal"] == 2 * r["k"] for r in r2)
    assert all(r["n_splits"] == (1 if r["order"] == "chronological" else 20) for r in r2)
    assert all(0 < r["n_eff_ratio"] <= 1 for r in r1 if r["method"] == "weighted")
    assert all(r["n_cal_subjects"] == 3 for r in r1)       # round(0.3 * 11)
    assert {r["target"] for r in records} == set(range(1, 13))
    assert len(summary) == 3 + 2 * 5 * 2 and all(s["n_subjects"] == 12 for s in summary)


def test_r1_recentering_fixes_what_pooling_gets_wrong(stationary_run):
    _, summary = stationary_run
    pooled = _row(summary, regime="R1", method="pooled")
    rc = _row(summary, regime="R1", method="recentered")
    assert abs(rc["coverage_mean"] - (1 - ALPHA)) < 0.03
    assert rc["frac_undercovered"] == 0.0
    assert pooled["frac_undercovered"] >= 0.25              # several subjects badly under-covered
    assert rc["coverage_min"] > pooled["coverage_min"] + 0.1
    assert rc["accuracy"] > pooled["accuracy"]


def test_r2_shuffled_coverage_matches_theory(stationary_run):
    _, summary = stationary_run
    for method in ("mondrian", "mondrian_recentered"):
        for k in (2, 4, 8, 16, 32):
            row = _row(summary, regime="R2", method=method, k=k, order="shuffled")
            assert row["coverage_theory"] == pytest.approx(conformal_rank(2 * k, ALPHA) / (2 * k + 1))
            assert abs(row["coverage_mean"] - row["coverage_theory"]) < 0.02, (method, k, row)
            chrono = _row(summary, regime="R2", method=method, k=k, order="chronological")
            # Stationary subjects: one chronological block per subject is unbiased, with
            # between-subject spread given by Beta(r, n + 1 - r). Allow 3.5 standard errors.
            n, r = 2 * k, conformal_rank(2 * k, ALPHA)
            se = stats.beta(r, n + 1 - r).std() / np.sqrt(12)
            assert abs(chrono["coverage_mean"] - row["coverage_theory"]) < 3.5 * se, (method, k, chrono)


def test_r2_small_block_gives_full_sets_at_strict_alpha():
    """alpha = 0.1 needs n >= 9 calibration trials; with k = 2 (n = 4) every label is kept."""
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(4, 80, means, SIGMA, shift=0.5, rng=np.random.default_rng(1))
    ev = LOSOConformalEvaluation(alpha=0.1, ks=(2,), n_shuffles=3)
    for r in ev.run(X, y, g, regimes=("R2",)):
        assert r["coverage"] == 1.0 and r["singleton_rate"] == 0.0 and r["coverage_theory"] == 1.0


def test_chronological_vs_shuffled_detects_within_session_drift():
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(20, 240, means, SIGMA, shift=1.0, rng=np.random.default_rng(9), drift=3.0)
    ev = LOSOConformalEvaluation(alpha=ALPHA, ks=(32,), n_shuffles=10)
    summary = summarize(ev.run(X, y, g, regimes=("R2",)))
    chrono = _row(summary, method="mondrian", order="chronological")
    shuffled = _row(summary, method="mondrian", order="shuffled")
    assert abs(shuffled["coverage_mean"] - shuffled["coverage_theory"]) < 0.02
    assert chrono["coverage_mean"] < shuffled["coverage_mean"] - 0.08


def test_sessions_are_evaluated_separately_and_small_blocks_skipped():
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(4, 80, means, SIGMA, shift=0.5, rng=np.random.default_rng(2))
    sessions = np.tile(np.repeat(["0train", "1test"], 40), 4)
    ev = LOSOConformalEvaluation(alpha=ALPHA, ks=(4, 16, 32), n_shuffles=2, min_test_trials=10)
    records = ev.run(X, y, g, sessions=sessions, regimes=("R2",))
    assert {r["session"] for r in records} == {"0train", "1test"}
    assert {r["k"] for r in records} <= {4, 16}            # k = 32 needs 64 of 40 trials
    assert all(r["n_test"] >= 10 for r in records)
    assert all(s["n_subjects"] == 4 for s in summarize(records))
    with pytest.raises(ValueError):
        ev.run(X[g < 2], y[g < 2], g[g < 2])


def test_drift_free_simulator_is_unchanged_by_new_argument():
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    a = sample_multisubject(3, 20, means, SIGMA, shift=0.5, rng=np.random.default_rng(0))
    b = sample_multisubject(3, 20, means, SIGMA, shift=0.5, rng=np.random.default_rng(0), drift=0.0)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
