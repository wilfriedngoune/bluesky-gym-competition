"""Optional root-candidate separation screen; no simulator or scoring changes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeparationForecast:
    minimum_nm: float
    intrusion_deficit_nm_seconds: float
    buffer_deficit_nm_seconds: float
    duration_seconds: float


def select_separation_candidate(forecasts, visits, values, priors, threshold_nm):
    """Keep the RL search ranking within the safe set, or least-severity set.

    Severity first minimizes integrated penetration below the official minimum,
    then integrated penetration below the buffered target. Distances are summed
    over neighbours and time. This is an approximate screen, not a guarantee.
    """
    safe = [i for i, forecast in enumerate(forecasts) if forecast.minimum_nm >= threshold_nm]
    eligible = safe or list(range(len(forecasts)))
    def ranking(i):
        learned = (visits[i], values[i], priors[i])
        if safe:
            return learned
        forecast = forecasts[i]
        return (-forecast.intrusion_deficit_nm_seconds,
                -forecast.buffer_deficit_nm_seconds, *learned)
    return max(eligible, key=ranking)
