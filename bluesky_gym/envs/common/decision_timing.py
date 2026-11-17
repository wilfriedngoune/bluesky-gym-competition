"""Decision timing without changing simulation or metric sampling intervals."""

import math


def validate_decision_seconds(value):
    """Keep complete decisions inside the fixed 3000-second episode budget.

    The inherited step loop advances whole decision intervals. Restricting
    these settings to divisors prevents overshooting the official time limit.
    The upper bound also fits the independent lookahead predictor's segments.
    """
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 1 <= value <= 120 or 3000 % value):
        raise ValueError("decision_seconds must be an integer divisor of 3000 in [1, 120], e.g. 5, 10, 15, 20 or 30")
    return value


def resolve_decision_seconds(mdp, saved=None, override=None):
    default = 10 if mdp == "base" else 5
    return validate_decision_seconds(override if override is not None else saved if saved is not None else default)


def rescale_discount(gamma, previous_seconds, decision_seconds):
    """Keep the same discount over simulated time when decision spacing changes."""
    validate_decision_seconds(previous_seconds)
    validate_decision_seconds(decision_seconds)
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    return gamma if previous_seconds == decision_seconds else gamma ** (decision_seconds / previous_seconds)


def forecast_segment_seconds(requested_seconds, decision_seconds):
    """Search cannot change manoeuvres earlier than the real next decision."""
    validate_decision_seconds(decision_seconds)
    if not math.isfinite(requested_seconds) or not 0 < requested_seconds <= 120:
        raise ValueError("Forecast duration must be in (0, 120] seconds")
    intervals = min(max(1, math.ceil(requested_seconds / decision_seconds)), 120 // decision_seconds)
    return float(intervals * decision_seconds)
