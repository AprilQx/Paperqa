from logging import getLogger
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    PARTIAL,
    Metric,
    SampleScore,
    Value,
    ValueToFloat,
    metric,
    value_to_float,
)

logger = getLogger(__name__)

# Reuse the precision_value_to_float function from the example
def precision_value_to_float(
    correct: Value = CORRECT,
    incorrect: Value = INCORRECT,
    partial: Value = PARTIAL,
    noanswer: Value = NOANSWER,
) -> ValueToFloat:
    """Create a ValueToFloat function."""

    def to_float(value: Value) -> float:
        if isinstance(value, int | float | bool):
            return float(value)
        elif value == noanswer:
            return -1
        else:
            return value_to_float(
                correct=correct, incorrect=incorrect, partial=partial, noanswer=noanswer
            )(value)
    return to_float

@metric
def precision(to_float: ValueToFloat = precision_value_to_float()) -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        answered = [item for item in scores if to_float(item.score.value) != -1]
        if len(answered) == 0:
            return 0.0

        total = 0.0
        for item in answered:
            total += to_float(item.score.value)
        return total / float(len(answered))  # Divide by ANSWERED questions

    return metric

@metric
def coverage(to_float: ValueToFloat = precision_value_to_float()) -> Metric:
    r"""Compute proportion of answered questions to total questions.

    Args:
       to_float: Function for mapping `Value` to float for computing
          metrics. The default `value_to_float()` maps CORRECT ("C") to 1.0,
          INCORRECT ("I") to 0, PARTIAL ("P") to 0.5, and NOANSWER ("N") to -1,
          casts numeric values to float directly, and prints a warning and returns
          0 if the Value is a complex object (list or dict).

          Note that this value_to_float must return -1 for NOANSWER in order for
          the precision metric to be computed correctly.

    Returns:
       Coverage metric
    """

    def metric(scores: list[SampleScore]) -> float:
        if len(scores) == 0:
            return 0.0
        # Filter to only answered questions
        answered = [item for item in scores if to_float(item.score.value) != -1]
        return float(len(answered)) / float(len(scores))

    return metric

@metric
def accuracy(to_float: ValueToFloat = precision_value_to_float()) -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        if len(scores) == 0:
            return 0.0
            
        # Filter to find answered questions (excluding NOANSWER)
        answered = [item for item in scores if to_float(item.score.value) != -1]
        
        correct = 0.0
        for item in answered:
            correct += to_float(item.score.value)
            
        # Divide by TOTAL questions
        return correct / float(len(scores))  # This is the key difference!

    return metric