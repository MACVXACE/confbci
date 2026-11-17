# confbci

Conformal prediction for motor-imagery brain-computer interfaces (BCIs): prediction
sets with finite-sample coverage guarantees, abstention, and a bound on wrong
feedback, for decoders used in stroke rehabilitation.

## Why

In motor-imagery neurofeedback for stroke rehabilitation, the decoder decides when
the patient is told "that attempt worked". The therapeutic premise is that feedback
contingent on the intended sensorimotor activity reinforces it, so feedback given
on a wrong decode is worse than no feedback. Standard decoders nevertheless return
a hard label on every trial, including when cross-subject accuracy on patients is
near chance. `confbci` wraps a decoder so that it returns a *prediction set*
(`{left}`, `{right}`, or `{left, right}` = abstain). If calibration and test trials
are exchangeable, the true label is in the set with probability at least
`1 - alpha`, for any decoder and any data distribution. The toolkit also provides
the corrections needed when exchangeability fails across subjects and within a
session, and a benchmark that measures when it fails.

## Install

```bash
pip install -e .
pytest                      # 237 tests, synthetic data only, about a minute
```

## Quickstart

Any scikit-learn-compatible classifier with `predict_proba` and `classes_` works,
including any pyRiemann pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from confbci import SplitCP

clf = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression())
clf.fit(X_train, y_train)                             # epochs: (trials, channels, samples)
cp = SplitCP(clf, alpha=0.2).calibrate(X_cal, y_cal)  # trials the decoder never saw
sets = cp.predict_set(X_new)                          # bool array (trials, classes)
give_feedback = sets.sum(axis=1) == 1                 # commit only when one label remains
```

`cp.coverage(sets, y_new)` returns the empirical coverage. The calibration trials
must not have been used to fit `clf`.

## What is in the box

| Object | Purpose | Guarantee, and what it assumes |
|---|---|---|
| `ProbMDM` | Minimum distance to mean with probabilities `softmax(-d_R^2 / tau)`; optional Riemannian re-centering per subject | none (a decoder) |
| `LAC`, `RiemannianMargin` | Nonconformity scores: `1 - p(y)` and `d_R(C, M_y) - min_{k != y} d_R(C, M_k)` | none |
| `SplitCP` | Split conformal prediction | `1 - alpha <= P(Y in C(X)) <= 1 - alpha + 1/(n+1)`; exchangeable calibration and test trials |
| `WeightedCP` + `DomainClassifierDensityRatio` | Weighted conformal prediction for covariate shift | coverage `>= 1 - alpha` if the weights equal the true density ratio and `P(y \| x)` is shared |
| `MondrianCP` | Per-subject threshold from a short labelled block | as `SplitCP`, within each subject |
| `AdaptiveCI` | Online update of the working level after each cue | long-run miscoverage within `(max(alpha, 1-alpha) + gamma) / (gamma T)` of `alpha`; no assumption on the data |
| `crc_feedback_threshold` | Confidence threshold for giving feedback | `P(feedback given AND wrong) <= alpha`; exchangeable trials |
| `screening_metrics` | Spearman rho and AUROC between block singleton rate and held-out accuracy | none (a statistic) |

## Evaluation regimes

`LOSOConformalEvaluation` holds out one subject at a time and evaluates three
regimes, which differ in what is known about the new patient.

**R1, zero calibration.** No labelled trial from the target. Source subjects are
split *by subject* into training subjects (fit the decoder) and calibration
subjects (fit the threshold). Methods: `pooled` (plain split CP), `weighted`
(density-ratio weights between calibration trials and the target's unlabeled
trials), `recentered` (every subject's Riemannian mean moved to the identity, the
target's from its own unlabeled trials, then split CP). Validity here rests on the
target being exchangeable with the calibration subjects after the correction.

**R2, few-shot.** The first `k` trials per class of the target, `k` in
{2, 4, 8, 16, 32}, form a labelled calibration block with a per-subject (Mondrian)
threshold; the remaining trials are the test set. Each configuration is run in
recorded order (`chronological`) and with a randomly drawn block (`shuffled`).
Shuffling makes calibration and test exchangeable by construction, so the gap
between the two orders measures within-session non-stationarity. With `n = 2k`
calibration trials the expected coverage is `ceil((n+1)(1-alpha)) / (n+1)`, and
`n >= 1/alpha - 1` is needed for a finite threshold.

**R3, online.** Trials are processed in recorded order; after each one the cue is
revealed and `AdaptiveCI` moves its working level
`alpha_{t+1} = alpha_t + gamma (alpha - err_t)`. The predictor starts from the
calibration subjects' scores. Reported per session: rolling 20-trial coverage,
long-run coverage error, and the trajectory of `alpha_t`, for `static` (never
updated), `aci`, and `aci_recentered`.

```python
from confbci import LOSOConformalEvaluation, RiemannianMargin, summarize
ev = LOSOConformalEvaluation(score=RiemannianMargin(), alpha=0.2)
records = ev.run_moabb("Liu2024", regimes=("R1", "R2", "R3"))
summary = summarize(records)
```

## Reproducing the benchmark and figures

```bash
python benchmarks/run_benchmark.py --dataset Liu2024      # writes results/*.csv
python benchmarks/make_figures.py --records results/Liu2024_records.csv
python benchmarks/make_figures.py --demo                  # synthetic subjects, no download
```

## Limits you should know before using this

- **Coverage is marginal.** It holds on average over trials, not for each trial or
  each class. At chance-level accuracy a valid predictor abstains on most trials;
  that is the correct output, not a failure.
- **The online bound is a long-run average and is vacuous for short sessions.** For
  `T = 40` trials and `gamma = 0.05` it equals 0.43. It says nothing about any
  20-trial window.
- **Risk control bounds a joint rate.** `P(feedback AND wrong) <= alpha` is per
  trial attempted. The error rate among trials that receive feedback is larger by
  the factor `1 / P(feedback)`.
- **`aci_recentered` and `static_recentered` are not causal.** They re-center a
  session with the mean of all its trials, including future ones, and are an
  offline upper bound (`causal = False` in the records).
- **Weighted CP needs overlapping domains.** In the tangent space
  (`c (c + 1) / 2` dimensions) a domain classifier can separate two subjects
  perfectly. The weights then put nearly all mass at `+inf` (`inf_mass_` near 1) and
  the method returns the full label set: valid, and uninformative.
- **Re-centering and exchangeability.** With `recenter=True`, a subject's
  calibration and test trials must be re-centered with the same reference: score
  them in one call and use the `*_from_scores` methods.
- **The simulator** draws a wrapped (log-normal) Gaussian on the SPD manifold, not
  the maximum-entropy Riemannian Gaussian.

## Citation

A tool paper is in `paper/confbci.tex`. Until it has a DOI, cite the repository.

## License

BSD-3-Clause.
