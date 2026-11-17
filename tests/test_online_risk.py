"""Tests for Week 3: adaptive conformal inference, conformal risk control for
feedback, the online regime R3, and patient-screening metrics.

Synthetic data only; seeds are fixed, so the suite is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from confbci import (
    AdaptiveCI,
    LOSOConformalEvaluation,
    ProbMDM,
    SplitCP,
    aci_coverage_bound,
    auroc,
    crc_feedback_threshold,
    feedback_risk,
    rolling_coverage,
    screening_metrics,
    split_quantile,
    summarize,
)
from confbci.metrics import long_run_coverage_error
from confbci.sim import make_class_means, sample_dataset, sample_multisubject

C, SIGMA, ALPHA = 4, 0.6, 0.2


# --------------------------------------------------------------------------- #
# AdaptiveCI
# --------------------------------------------------------------------------- #
def test_aci_update_rule_and_extreme_levels():
    aci = AdaptiveCI(alpha=0.2, gamma=0.1, init_scores=np.linspace(0, 1, 50))
    s = np.array([0.1, 0.99])
    sets = aci.predict_set(s)
    assert sets.tolist() == [True, False]
    assert aci.update(sets, y_idx=0, s_true=0.1) == 0.0           # hit -> alpha_t rises (smaller sets)
    assert aci.alpha_t == pytest.approx(0.2 + 0.1 * 0.2)
    assert aci.update(sets, y_idx=1, s_true=0.99) == 1.0          # miss -> alpha_t falls (larger sets)
    assert aci.alpha_t == pytest.approx(0.22 + 0.1 * (0.2 - 1))
    assert len(aci.scores_) == 52

    aci.alpha_t = -0.05                                            # full label set
    assert aci.threshold() == np.inf and aci.predict_set(s).all()
    aci.alpha_t = 1.3                                              # empty set
    assert aci.threshold() == -np.inf and not aci.predict_set(s).any()
    aci.alpha_t = 0.3
    assert aci.threshold() == split_quantile(aci.scores_, 0.3)


def test_aci_cold_start_gives_full_sets_and_window_is_respected():
    aci = AdaptiveCI(alpha=0.2, gamma=0.01, window=30)
    assert aci.predict_set([0.4, 0.9]).all()                       # no scores yet
    rng = np.random.default_rng(0)
    S = rng.random((200, 2))
    aci.run(S, rng.integers(0, 2, 200))
    assert len(aci.scores_) == 30
    with pytest.raises(ValueError):
        AdaptiveCI(alpha=0.2, gamma=0.0)
    with pytest.raises(ValueError):
        AdaptiveCI(alpha=1.2)


@pytest.mark.parametrize("gamma", [0.005, 0.01, 0.05, 0.2])
@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("scenario", ["iid", "drift", "jump", "adversarial", "cold"])
def test_aci_long_run_bound_holds_for_any_sequence(gamma, alpha, scenario):
    """|mean(err) - alpha| <= (max(alpha, 1 - alpha) + gamma) / (gamma T), whatever the data."""
    rng = np.random.default_rng([len(scenario), int(gamma * 1000), int(alpha * 10)])
    T = 600
    t = np.arange(T)[:, None]
    S = rng.random((T, 2))
    y = rng.integers(0, 2, T)
    init = rng.random(100) * 0.5
    if scenario == "drift":
        S = S + 2.0 * t / T                                        # scores grow steadily
    elif scenario == "jump":
        S[T // 2:] += 5.0                                          # decoder breaks mid-session
    elif scenario == "adversarial":
        S[np.arange(T), y] = 1e6                                   # true label always looks worst
    elif scenario == "cold":
        init = ()
    aci = AdaptiveCI(alpha=alpha, gamma=gamma, init_scores=init)
    res = aci.run(S, y)
    err = long_run_coverage_error(res["covered"], alpha)
    assert err <= aci_coverage_bound(T, gamma, alpha) + 1e-12
    assert aci.coverage_bound() == pytest.approx(aci_coverage_bound(T, gamma, alpha))
    assert res["sets"].shape == (T, 2) and res["alpha_t"].shape == (T,) and res["alpha_t"][0] == alpha
    # the theory also says alpha_t stays in [-gamma, 1 + gamma]
    assert res["alpha_t"].min() >= -gamma - 1e-12 and res["alpha_t"].max() <= 1 + gamma + 1e-12


def test_aci_recovers_after_shift_where_static_threshold_fails():
    rng = np.random.default_rng(1)
    T = 1000
    s_true = rng.random(T)
    s_true[T // 2:] += 0.6                                         # true-label scores jump up
    S = np.stack([s_true, np.full(T, 10.0)], axis=1)               # other label never included
    y = np.zeros(T, dtype=int)
    init = rng.random(200)

    static = S[:, 0] <= split_quantile(init, ALPHA)
    res = AdaptiveCI(alpha=ALPHA, gamma=0.05, init_scores=init, window=100).run(S, y)

    assert static[: T // 2].mean() == pytest.approx(1 - ALPHA, abs=0.06)
    assert static[T // 2:].mean() < 0.35                           # static CP collapses after the jump
    assert res["covered"][T // 2 + 100:].mean() == pytest.approx(1 - ALPHA, abs=0.05)
    assert res["covered"].mean() == pytest.approx(1 - ALPHA, abs=0.03)


def test_aci_on_exchangeable_decoder_scores():
    rng = np.random.default_rng(2)
    means = make_class_means(C, 2, 0.6, rng=rng)
    X_train, y_train = sample_dataset(400, means, SIGMA, rng)
    model = ProbMDM(recenter=False).fit(X_train, y_train)
    X_cal, y_cal = sample_dataset(200, means, SIGMA, rng)
    X, y = sample_dataset(1500, means, SIGMA, rng)
    S = 1 - model.predict_proba(X)
    init = (1 - model.predict_proba(X_cal))[np.arange(200), y_cal]
    res = AdaptiveCI(alpha=ALPHA, gamma=0.01, init_scores=init).run(S, y)
    assert res["covered"].mean() == pytest.approx(1 - ALPHA, abs=0.02)
    # stationary data: the working level stays near the target
    assert abs(res["alpha_t"].mean() - ALPHA) < 0.08


# --------------------------------------------------------------------------- #
# Conformal risk control
# --------------------------------------------------------------------------- #
def test_crc_hand_example():
    conf = np.array([0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.65, 0.85])   # n = 9
    correct = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0], dtype=bool)               # wrong at .55 .60 .80 .85
    grid = np.linspace(0, 1, 101)
    # need (n_bad + 1) / 10 <= alpha
    assert crc_feedback_threshold(conf, correct, 0.5, grid) == 0.0                  # 4 wrong allowed
    assert crc_feedback_threshold(conf, correct, 0.3, grid) == pytest.approx(0.61)  # 2 wrong: > 0.60
    assert crc_feedback_threshold(conf, correct, 0.2, grid) == pytest.approx(0.81)  # 1 wrong: > 0.80
    assert crc_feedback_threshold(conf, correct, 0.1, grid) == pytest.approx(0.86)  # 0 wrong: > 0.85
    assert crc_feedback_threshold(conf, correct, 0.05, grid) == np.inf              # 1/10 > 0.05
    assert feedback_risk(conf, correct, 0.81) == pytest.approx(1 / 9)
    assert feedback_risk(conf, correct, np.inf) == 0.0


def test_crc_matches_brute_force_and_is_monotone_in_alpha():
    rng = np.random.default_rng(3)
    grid = np.linspace(0.5, 1.0, 51)
    for _ in range(200):
        n = int(rng.integers(5, 150))
        conf = rng.uniform(0.5, 1, n)
        correct = rng.random(n) < conf
        prev = -np.inf
        for alpha in (0.4, 0.2, 0.1, 0.05):
            lam = crc_feedback_threshold(conf, correct, alpha, grid)
            ok = [g for g in grid
                  if n / (n + 1) * np.mean((conf >= g) & ~correct) + 1 / (n + 1) <= alpha + 1e-12]
            assert lam == (min(ok) if ok else np.inf)
            assert lam >= prev                                    # stricter alpha -> higher threshold
            prev = lam


def test_crc_validates_input():
    with pytest.raises(ValueError):
        crc_feedback_threshold([0.9, 0.8], [True], 0.2)
    with pytest.raises(ValueError):
        crc_feedback_threshold([0.9, np.nan], [True, False], 0.2)
    with pytest.raises(ValueError):
        crc_feedback_threshold([0.9, 0.8], [True, False], 0.0)


@pytest.mark.parametrize("n_cal", [20, 50, 200])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_crc_bounds_joint_false_feedback_rate(n_cal, alpha):
    """E[P(feedback AND wrong)] <= alpha over calibration draws, on real decoder outputs."""
    rng = np.random.default_rng(100 + n_cal)
    means = make_class_means(C, 2, 0.6, rng=rng)
    X_train, y_train = sample_dataset(400, means, SIGMA, rng)
    model = ProbMDM(recenter=False).fit(X_train, y_train)
    X, y = sample_dataset(40_000, means, SIGMA, rng)
    P = model.predict_proba(X)
    conf, correct = P.max(axis=1), P.argmax(axis=1) == y
    base_error = 1 - correct.mean()
    assert base_error > alpha                                     # otherwise the problem is trivial

    risks, rates = [], []
    for _ in range(2000):
        idx = rng.permutation(len(y))
        cal, test = idx[:n_cal], idx[n_cal:n_cal + 5000]
        lam = crc_feedback_threshold(conf[cal], correct[cal], alpha)
        risks.append(feedback_risk(conf[test], correct[test], lam))
        rates.append(np.mean(conf[test] >= lam))
    risk = float(np.mean(risks))
    se = float(np.std(risks) / np.sqrt(len(risks)))
    assert risk <= alpha + 3 * se, (risk, se)                     # the guarantee
    assert risk >= alpha - 2.0 / (n_cal + 1) - 0.01 - 3 * se, risk  # and it is not needlessly conservative
    if n_cal >= 50:
        assert np.mean(rates) > 0.2                               # feedback is still given


def test_crc_never_gives_feedback_when_calibration_is_too_small():
    conf, correct = np.full(8, 0.99), np.ones(8, dtype=bool)
    assert crc_feedback_threshold(conf, correct, alpha=0.1) == np.inf   # 1/9 > 0.1
    assert crc_feedback_threshold(np.full(9, 0.99), np.ones(9, bool), alpha=0.1) == 0.0


# --------------------------------------------------------------------------- #
# metrics.py
# --------------------------------------------------------------------------- #
def test_rolling_coverage():
    c = np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    assert np.allclose(rolling_coverage(c, 4), [0.75, 0.5, 0.25, 0.5, 0.5])
    assert rolling_coverage(c, 8).tolist() == [0.625]
    assert rolling_coverage(c, 20).size == 0
    assert long_run_coverage_error(c, 0.2) == pytest.approx(abs(3 / 8 - 0.2))


def test_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(6, 60))
        scores = np.round(rng.random(n), 1)                       # many ties
        pos = rng.random(n) < 0.5
        if pos.all() or not pos.any():
            continue
        assert auroc(scores, pos) == pytest.approx(roc_auc_score(pos, scores))
    assert np.isnan(auroc([0.1, 0.2], [True, True]))


def test_screening_metrics_known_cases():
    acc = np.array([0.52, 0.55, 0.61, 0.66, 0.72, 0.78, 0.85, 0.91])
    perfect = screening_metrics(acc - 0.3, acc, n_boot=200)
    assert perfect["spearman_rho"] == pytest.approx(1.0) and perfect["auroc"] == 1.0
    assert perfect["n_efficient"] == 4 and perfect["n_inefficient"] == 4
    reversed_ = screening_metrics(1 - acc, acc, n_boot=0)
    assert reversed_["spearman_rho"] == pytest.approx(-1.0) and reversed_["auroc"] == 0.0
    assert np.isnan(reversed_["auroc_ci"][0])

    all_low = screening_metrics(np.linspace(0, 1, 8), acc * 0.5, n_boot=100)   # nobody reaches 0.70
    assert np.isnan(all_low["auroc"]) and all_low["n_efficient"] == 0
    flat = screening_metrics(np.zeros(8), acc, n_boot=0)                        # no abstention signal
    assert np.isnan(flat["spearman_rho"]) and flat["auroc"] == 0.5
    with pytest.raises(ValueError):
        screening_metrics([0.1, 0.2], [0.5, 0.9])

    rng = np.random.default_rng(5)
    x = rng.random(40)
    m = screening_metrics(x, 0.5 + 0.4 * x + rng.normal(0, 0.05, 40), n_boot=500)
    assert m["spearman_ci"][0] < m["spearman_rho"] < m["spearman_ci"][1]
    assert m["auroc_ci"][0] <= m["auroc"] <= m["auroc_ci"][1] and m["auroc"] > 0.8


# --------------------------------------------------------------------------- #
# Regime R3 and screening records
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def drift_run():
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(10, 240, means, SIGMA, shift=1.0, rng=np.random.default_rng(9), drift=3.0)
    ev = LOSOConformalEvaluation(alpha=ALPHA, gammas=(0.005, 0.01, 0.05), random_state=0)
    records = ev.run(X, y, g, regimes=("R1", "R3"))
    return ev, records, (X, y, g)


def test_r3_records(drift_run):
    _, records, _ = drift_run
    r3 = [r for r in records if r["regime"] == "R3"]
    assert len(r3) == 10 * 2 * (1 + 3)                             # subjects x models x (static + 3 gammas)
    assert {r["method"] for r in r3} == {"static", "aci", "static_recentered", "aci_recentered"}
    for r in r3:
        assert r["order"] == "chronological" and r["n_test"] == 240
        assert r["causal"] == (not r["method"].endswith("recentered"))
        assert r["rolling_coverage"].shape == (240 - 20 + 1,)
        assert r["alpha_trajectory"].shape == (240,) and r["covered"].shape == (240,)
        assert r["rolling_min"] == pytest.approx(r["rolling_coverage"].min())
        assert r["long_run_error"] == pytest.approx(abs(1 - r["covered"].mean() - ALPHA))
        if r["gamma"] is not None:
            assert r["long_run_error"] <= r["aci_bound"] + 1e-12   # Gibbs & Candes guarantee
            assert r["aci_bound"] == pytest.approx(aci_coverage_bound(240, r["gamma"], ALPHA))
        else:
            assert np.all(r["alpha_trajectory"] == ALPHA)


def test_r3_static_equals_r1_pooled(drift_run):
    """Same subject split and threshold: R3 'static' must reproduce the R1 result."""
    _, records, _ = drift_run
    r1 = {r["target"]: r for r in records if r["regime"] == "R1" and r["method"] == "pooled"}
    r1rc = {r["target"]: r for r in records if r["regime"] == "R1" and r["method"] == "recentered"}
    for r in records:
        if r["regime"] == "R3" and r["method"] == "static":
            assert r["coverage"] == pytest.approx(r1[r["target"]]["coverage"])
        if r["regime"] == "R3" and r["method"] == "static_recentered":
            assert r["coverage"] == pytest.approx(r1rc[r["target"]]["coverage"])


def test_r3_aci_tightens_long_run_coverage_under_drift(drift_run):
    _, records, _ = drift_run
    summary = {(s["method"], s["gamma"]): s for s in summarize(records) if s["regime"] == "R3"}
    static, fast = summary[("static", None)], summary[("aci", 0.05)]
    assert static["long_run_error"] > 0.08                         # per-subject coverage is far off
    assert fast["long_run_error"] < 0.03
    assert fast["coverage_min"] > static["coverage_min"] + 0.1
    assert fast["frac_windows_under"] < static["frac_windows_under"]
    errors = [summary[("aci", g)]["long_run_error"] for g in (0.005, 0.01, 0.05)]
    assert errors[0] > errors[1] > errors[2]                       # larger step tracks drift faster


def test_r3_per_session_and_no_trajectories():
    means = make_class_means(C, 2, 0.6, rng=np.random.default_rng(8))
    X, y, g = sample_multisubject(4, 80, means, SIGMA, shift=0.5, rng=np.random.default_rng(2))
    sessions = np.tile(np.repeat(["0train", "1test"], 40), 4)
    ev = LOSOConformalEvaluation(alpha=ALPHA, gammas=(0.05,), keep_trajectories=False)
    records = ev.run(X, y, g, sessions=sessions, regimes=("R3",))
    assert len(records) == 4 * 2 * 2 * 2
    assert {r["session"] for r in records} == {"0train", "1test"}
    assert all(r["n_test"] == 40 and "alpha_trajectory" not in r for r in records)
    assert all(s["n_subjects"] == 4 for s in summarize(records))


def test_screening_block_uses_no_heldout_trials_and_tracks_separability():
    """Patients differ in class separability; the block singleton rate should rank them."""
    rng = np.random.default_rng(6)
    Xs, ys, gs = [], [], []
    seps = np.linspace(0.05, 1.2, 16)
    for s, sep in enumerate(seps):
        means = make_class_means(C, 2, sep, base_dispersion=0.0, rng=np.random.default_rng(77))
        X, y = sample_dataset(200, means, 0.35, rng)
        Xs.append(X); ys.append(y); gs.append(np.full(200, s))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)

    ev = LOSOConformalEvaluation(alpha=ALPHA)
    recs = ev.screening_records(X, y, g, block_size=40)
    assert len(recs) == 16 and all(r["n_heldout"] == 160 for r in recs)
    m = screening_metrics([r["singleton_rate_block"] for r in recs],
                          [r["heldout_accuracy"] for r in recs], n_boot=300)
    assert 0 < m["n_efficient"] < 16
    assert m["spearman_rho"] > 0.5 and m["auroc"] > 0.8, m

    # Causality: altering one patient's held-out trials leaves that patient's block result untouched.
    X_alt = X.copy()
    rest = np.flatnonzero(g == 9)[40:]
    X_alt[rest] = X_alt[rest][::-1] * 3.0
    alt = ev.screening_records(X_alt, y, g, block_size=40, targets=[9])
    assert alt[0]["target"] == 9
    assert alt[0]["singleton_rate_block"] == recs[9]["singleton_rate_block"]
