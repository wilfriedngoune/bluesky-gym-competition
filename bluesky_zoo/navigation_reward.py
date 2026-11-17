"""Configurable per-aircraft rewards; independent of competition scoring.

Rates use simulated seconds and the official metric deltas. Distance shaping
uses PPO's discount, and only a true goal termination has zero potential.
Time-limit truncations keep their potential for SB3's value bootstrapping.
"""

from dataclasses import dataclass, fields, replace
import math

import numpy as np


REWARD_COMPONENT_KEYS = (
    "goal", "progress", "intrusion", "restricted_area", "sector",
    "predicted_conflict", "flight_time", "action_smoothness",
    "sector_buffer", "aircraft_buffer",
)


@dataclass(frozen=True)
class NavigationRewardConfig:
    gamma: float = 0.99
    goal_bonus: float = 20.0
    progress_weight: float = 10.0
    distance_scale_km: float = 100.0
    intrusion_event_cost: float = 5.0
    intrusion_second_cost: float = 0.1
    restricted_event_cost: float = 5.0
    restricted_second_cost: float = 0.05
    sector_event_cost: float = 5.0
    sector_second_cost: float = 0.05
    conflict_risk_second_cost: float = 0.02
    conflict_horizon_seconds: float = 120.0
    conflict_buffer_nm: float = 8.0
    flight_second_cost: float = 0.002
    action_change_cost: float = 0.005
    sector_buffer_second_cost: float = 0.0
    sector_buffer_nm: float = 5.0
    aircraft_buffer_second_cost: float = 0.0
    aircraft_buffer_nm: float = 8.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        for name in ("distance_scale_km", "conflict_horizon_seconds", "conflict_buffer_nm", "sector_buffer_nm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.aircraft_buffer_nm <= 5.0:
            raise ValueError("aircraft_buffer_nm must exceed the fixed 5 NM separation minimum")
        if self.conflict_horizon_seconds > 600:
            raise ValueError("conflict_horizon_seconds must be <= 600 (the CPA helper's receding sentinel)")


def make_reward_config(mdp, *, gamma=None, conflict_horizon_seconds=None,
                       conflict_risk_second_cost=None, conflict_buffer_nm=None,
                       sector_buffer_second_cost=None, sector_buffer_nm=None,
                       aircraft_buffer_second_cost=None, aircraft_buffer_nm=None,
                       saved=None):
    """Versioned defaults, with explicit overrides recorded in the checkpoint."""
    overrides = {name: value for name, value in {
        "gamma": gamma, "conflict_horizon_seconds": conflict_horizon_seconds,
        "conflict_risk_second_cost": conflict_risk_second_cost,
        "conflict_buffer_nm": conflict_buffer_nm,
        "sector_buffer_second_cost": sector_buffer_second_cost,
        "sector_buffer_nm": sector_buffer_nm,
        "aircraft_buffer_second_cost": aircraft_buffer_second_cost,
        "aircraft_buffer_nm": aircraft_buffer_nm,
    }.items() if value is not None}
    if mdp not in {"navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5"}:
        if any(key != "gamma" for key in overrides):
            raise ValueError("Reward warning options require navigation-v2 or navigation-v3 or navigation-v4 or navigation-v5")
        return None
    config = NavigationRewardConfig()
    if mdp in {"navigation-v3", "navigation-v4", "navigation-v5"}:
        config = replace(config, gamma=0.998, conflict_horizon_seconds=180.0,
                         conflict_risk_second_cost=0.05)
    if mdp == "navigation-v5":
        config = replace(config, aircraft_buffer_second_cost=0.05)
    if saved is not None:
        # Dataclass defaults keep pre-bubble checkpoints disabled. Explicit
        # training overrides can enable the new reward without changing inputs.
        config = saved if isinstance(saved, NavigationRewardConfig) else NavigationRewardConfig(**saved)
    return replace(config, **overrides)


def distance_potential(distance_nm, capture_radius_km, config):
    remaining_km = max(float(distance_nm) * 1.852 - capture_radius_km, 0.0)
    return -remaining_km / config.distance_scale_km


def predicted_conflict_risk(tcpa, dcpa_nm, distance_nm, separation_nm, config):
    """Bounded warning before a future conflict, excluding actual intrusions."""
    if (distance_nm < separation_nm or not 0 < tcpa < config.conflict_horizon_seconds
            or dcpa_nm >= config.conflict_buffer_nm):
        return 0.0
    return float(np.clip(
        (1.0 - tcpa / config.conflict_horizon_seconds)
        * (1.0 - dcpa_nm / config.conflict_buffer_nm), 0.0, 1.0,
    ))


def sector_buffer_risk(clearance_nm, inside, config):
    """Smooth proximity cost inside; saturated outside to avoid an escape incentive.

    Clearance is unsigned distance to the existing sector polygon in NM. This
    warning region affects the learning signal only, not the scored boundary.
    """
    if not inside:
        return 1.0
    return float((1.0 - np.clip(clearance_nm / config.sector_buffer_nm, 0.0, 1.0)) ** 2)


def aircraft_buffer_risk(distance_nm, separation_nm, config):
    """Per-neighbour proximity risk, independent of bearing or closing speed.

    Zero at the warning radius, rising smoothly to one at actual separation
    loss, and saturated inside it. The radius is a centre-to-centre distance,
    not the sum of two aircraft's radii. Existing intrusion penalties remain.
    """
    if config.aircraft_buffer_nm <= separation_nm:
        raise ValueError("Aircraft warning radius must exceed the separation minimum")
    return float(np.clip((config.aircraft_buffer_nm - float(distance_nm))
                         / (config.aircraft_buffer_nm - separation_nm), 0.0, 1.0) ** 2)


def reward_components(config, *, previous_distance_nm, distance_nm,
                      capture_radius_km, metrics, baseline, risk,
                      action, previous_action, sector_risk=0.0, aircraft_risk=0.0):
    """Pure reward calculation; never mutates metrics or episode lifecycle.

    Baseline is the fixed metric snapshot at the start of this decision.
    Durations already include intermediate one-second safety measurements.
    """
    def delta(key):
        return float(metrics[key] - baseline[key])

    elapsed = delta("flight_time")
    reached = delta("waypoint_reached")
    progress = 0.0
    if previous_distance_nm is not None:
        before = distance_potential(previous_distance_nm, capture_radius_km, config)
        after = (0.0 if metrics["waypoint_reached"] else
                 distance_potential(distance_nm, capture_radius_km, config))
        progress = config.progress_weight * (config.gamma * after - before)
    action_change = np.asarray(action) - np.asarray(previous_action)
    return {
        "goal": config.goal_bonus * reached,
        "progress": progress,
        "intrusion": -(
            config.intrusion_event_cost * delta("intrusion_events")
            + config.intrusion_second_cost * delta("intrusion_time")
        ),
        "restricted_area": -(
            config.restricted_event_cost * delta("restricted_area_events")
            + config.restricted_second_cost * delta("time_in_restricted_area")
        ),
        "sector": -(
            config.sector_event_cost * delta("sector_exit_events")
            + config.sector_second_cost * delta("time_outside_sector")
        ),
        "predicted_conflict": -config.conflict_risk_second_cost * float(risk) * elapsed,
        "flight_time": -config.flight_second_cost * elapsed,
        "action_smoothness": -config.action_change_cost * float(np.dot(action_change, action_change)),
        "sector_buffer": -config.sector_buffer_second_cost * float(sector_risk) * elapsed,
        "aircraft_buffer": -config.aircraft_buffer_second_cost * float(aircraft_risk) * elapsed,
    }
