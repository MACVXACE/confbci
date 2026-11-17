"""confbci: conformal prediction for motor-imagery brain-computer interfaces."""

from .conformal import (
    AdaptiveCI,
    aci_coverage_bound,
    crc_feedback_threshold,
    feedback_risk,
    MondrianCP,
    SplitCP,
    WeightedCP,
    conformal_rank,
    empirical_coverage,
    split_quantile,
    weighted_quantile,
)
from .metrics import auroc, rolling_coverage, screening_metrics
from .evaluation import LOSOConformalEvaluation, set_metrics, summarize
from .models.riemann import ProbMDM, recenter
from .scores import LAC, RiemannianMargin
from .shift import DomainClassifierDensityRatio, effective_sample_size

__version__ = "0.0.1"

__all__ = [
    "ProbMDM",
    "recenter",
    "LAC",
    "RiemannianMargin",
    "SplitCP",
    "AdaptiveCI",
    "aci_coverage_bound",
    "crc_feedback_threshold",
    "feedback_risk",
    "screening_metrics",
    "auroc",
    "rolling_coverage",
    "WeightedCP",
    "MondrianCP",
    "weighted_quantile",
    "DomainClassifierDensityRatio",
    "effective_sample_size",
    "LOSOConformalEvaluation",
    "set_metrics",
    "summarize",
    "split_quantile",
    "conformal_rank",
    "empirical_coverage",
]
