"""Navigation v5: waypoint-centred shrinking zones used only by the RL MDP.

The circles are observations and bounded reward costs, never simulator areas.
World geometry, A320 dynamics, official metrics and episode lifecycle remain
inherited from the competition environment. All distances in this module are
nautical miles except the competition's explicitly named capture radius.
"""

from dataclasses import asdict, dataclass, fields, replace
import math

import bluesky as bs
import numpy as np
from gymnasium import spaces

from bluesky_zoo.navigation_reward import REWARD_COMPONENT_KEYS, make_reward_config
from bluesky_zoo.passage_navigation import MultiAgentPassageEnv, _PassageNavigation
from core.tools import kwikqdrdist


@dataclass(frozen=True)
class ZoneConfig:
    """Training incentives; none of these settings changes a world rule.

    The initial goal distance plus ``detour_allowance_nm`` sets the first
    radius. Shrinking begins after a grace period and reaches the existing
    capture radius at a distance-dependent deadline, capped by the episode's
    original time limit. Outside cost saturates at ``outside_second_cost``;
    its default is smaller than the existing occupied-safety-region rates.
    """

    grace_seconds: float = 120.0
    detour_allowance_nm: float = 10.0
    closing_speed_kt: float = 150.0
    arrival_time_factor: float = 1.5
    minimum_shrink_seconds: float = 300.0
    outside_second_cost: float = 0.01
    excess_scale_nm: float = 10.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        for name in ("closing_speed_kt", "minimum_shrink_seconds", "excess_scale_nm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.arrival_time_factor < 1:
            raise ValueError("arrival_time_factor must be >= 1 to allow nominal travel time")


def resolve_zone_config(saved=None, **overrides):
    """Reconstruct a persisted profile, applying only explicit overrides."""
    config = saved if isinstance(saved, ZoneConfig) else ZoneConfig(**(saved or {}))
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


@dataclass(frozen=True)
class ZoneSchedule:
    """An aircraft-local monotonic schedule, fixed once at episode reset."""

    initial_radius_nm: float
    capture_radius_nm: float
    grace_seconds: float
    end_seconds: float

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("Zone schedule values must be finite")
        if not 0 < self.capture_radius_nm <= self.initial_radius_nm:
            raise ValueError("Initial zone radius must include the positive capture radius")
        if not 0 <= self.grace_seconds < self.end_seconds:
            raise ValueError("Zone deadline must follow a nonnegative grace period")

    def radius_nm(self, sim_time):
        fraction = float(np.clip(
            (float(sim_time) - self.grace_seconds) / (self.end_seconds - self.grace_seconds),
            0.0, 1.0,
        ))
        return max(self.capture_radius_nm,
                   self.initial_radius_nm + fraction * (self.capture_radius_nm - self.initial_radius_nm))

    def observation(self, distance_nm, sim_time, episode_time_limit):
        radius = self.radius_nm(sim_time)
        return np.array([
            np.clip(radius / 100.0, 0.0, 3.0),
            np.clip((radius - float(distance_nm)) / 100.0, -3.0, 3.0),
            np.clip((self.end_seconds - float(sim_time)) / episode_time_limit, 0.0, 1.0),
        ], dtype=np.float64)


def make_zone_schedule(initial_distance_nm, capture_radius_km, episode_time_limit, config):
    """Build a schedule without changing or extending the real episode budget."""
    for name, value in (("initial_distance_nm", initial_distance_nm),
                        ("capture_radius_km", capture_radius_km),
                        ("episode_time_limit", episode_time_limit)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if capture_radius_km <= 0 or episode_time_limit <= 0:
        raise ValueError("capture_radius_km and episode_time_limit must be positive")
    capture_nm = float(capture_radius_km) / 1.852
    radius = max(capture_nm, float(initial_distance_nm) + config.detour_allowance_nm)
    # A shortened test episode still has a positive shrink interval. Official
    # evaluation retains the inherited 3000-second budget and the full grace.
    grace = min(config.grace_seconds, max(float(episode_time_limit) - 1.0, 0.0))
    nominal_travel = max(float(initial_distance_nm) - capture_nm, 0.0) * 3600.0 / config.closing_speed_kt
    duration = max(config.minimum_shrink_seconds, nominal_travel * config.arrival_time_factor)
    deadline = min(float(episode_time_limit), grace + duration)
    return ZoneSchedule(radius, capture_nm, grace, deadline)


def zone_penalty(distance_nm, radius_nm, elapsed_seconds, config):
    """Smooth nonpositive cost: no entry/exit bonuses, no growth past the cap.

    Endpoint geometry is sampled once per decision, like the optional sector
    buffer shaping. Duration is the aircraft's actual flown seconds, including
    an early-arrival decision that lasts less than the usual action interval.
    """
    excess = max(float(distance_nm) - float(radius_nm), 0.0)
    risk = float(np.clip(excess / config.excess_scale_nm, 0.0, 1.0)) ** 2
    return -config.outside_second_cost * max(float(elapsed_seconds), 0.0) * risk


class _StrategicNavigation(_PassageNavigation):
    def __init__(self, env, ac_id):
        super().__init__(env, ac_id)
        self.zone_schedule = None
        self.reward_component_totals["zone"] = 0.0
        self.last_reward_components["zone"] = 0.0
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "zone": spaces.Box(np.array([0.0, -3.0, 0.0]),
                               np.array([3.0, 3.0, 1.0]), dtype=np.float64),
        })

    def _reset_navigation(self):
        super()._reset_navigation()
        self.zone_schedule = None
        self.reward_component_totals["zone"] = 0.0
        self.last_reward_components["zone"] = 0.0

    def _goal_distance_nm(self):
        idx = bs.traf.id2idx(self.ac_id)
        _, distance = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], self.goal_lat, self.goal_lon)
        return float(distance)

    def _ensure_zone(self):
        if self.zone_schedule is None:
            self.zone_schedule = make_zone_schedule(
                self._goal_distance_nm(), self.env.distance_margin,
                self.env.episode_time_limit, self.env.zone_config,
            )
        return self.zone_schedule

    def _get_obs(self, ac_id):
        observation = super()._get_obs(ac_id)
        observation["zone"] = self._ensure_zone().observation(
            self._goal_distance_nm(), self.env.sim_time, self.env.episode_time_limit,
        )
        return observation

    def _get_reward(self, ac_id):
        schedule = self._ensure_zone()
        reward = super()._get_reward(ac_id)
        elapsed = self.metrics[ac_id]["flight_time"] - self._reward_baseline["flight_time"]
        cost = zone_penalty(self.previous_goal_distance, schedule.radius_nm(self.env.sim_time),
                            elapsed, self.env.zone_config)
        self.last_reward_components["zone"] = cost
        self.reward_component_totals["zone"] += cost
        return reward + cost


class MultiAgentStrategicEnv(MultiAgentPassageEnv):
    """V4 horizontal navigation plus a soft waypoint-centred shrinking zone.

    Inherits the actual simulator, scenario generator, metrics, step and goal
    termination. Leaving the virtual zone only changes the learning signal.
    New observations require v5-compatible weights; older checkpoints remain
    usable with their original MDPs.
    """

    navigation_class = _StrategicNavigation
    metadata = {**MultiAgentPassageEnv.metadata, "name": "navigation_v5"}
    reward_component_keys = (*REWARD_COMPONENT_KEYS, "zone")

    def __init__(self, zone_config=None, passage_config=None, reward_config=None, **kwargs):
        self.zone_config = resolve_zone_config(zone_config)
        super().__init__(passage_config=passage_config,
                         reward_config=reward_config or make_reward_config("navigation-v5"), **kwargs)

    def get_zone_config(self):
        return asdict(self.zone_config)
