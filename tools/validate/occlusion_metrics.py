"""Historical import compatibility; active callers use densekp.core.evaluation."""
from densekp.core.evaluation import (
    evaluation_masks,
    pair_masks,
    point_errors,
    prediction_retention,
    summarize,
)

__all__ = ['evaluation_masks', 'pair_masks', 'point_errors', 'prediction_retention', 'summarize']
