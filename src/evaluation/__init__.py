"""Evaluate fixed forecasts without fitting or selecting models.

prediction: apply a selected head.
metrics: WAPE, seasonal MASE and coherence nMAE.
statistics: paired comparisons, rank agreement and moving-block intervals.
base_checks: distinct frozen forecasts and canonical-nine coverage.

Import metrics/statistics directly for CPU analysis without loading PyTorch.
"""
