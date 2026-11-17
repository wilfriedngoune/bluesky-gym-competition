"""Navigation-v4: polygon look-ahead and policy access to all heading directions."""

from dataclasses import dataclass, replace
import math

import bluesky as bs
import numpy as np
from gymnasium import spaces

from bluesky_gym.envs.common import functions as fn
from bluesky_zoo.multi_agent_traffic_env import MultiAgentTrafficEnv, _TrafficNavigation
from bluesky_zoo.navigation_reward import make_reward_config


RAY_OFFSETS_DEG = np.arange(-180, 180, 15, dtype=np.float64)


@dataclass(frozen=True)
class PassageNavigationConfig:
    max_heading_offset_deg: float = 180.0
    obstacle_buffer_nm: float = 0.5
    obstacle_release_nm: float = 1.0
    lookahead_nm: float = 20.0

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.max_heading_offset_deg <= 180 or self.lookahead_nm <= 0:
            raise ValueError("Use heading offset in (0, 180] and positive lookahead")
        if self.obstacle_release_nm < self.obstacle_buffer_nm:
            raise ValueError("Obstacle release clearance must be >= activation buffer")


def resolve_passage_config(saved=None, **overrides):
    config = saved if isinstance(saved, PassageNavigationConfig) else PassageNavigationConfig(**(saved or {}))
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


def ray_distances(origin, headings_deg, starts, ends, horizon):
    """First polygon-edge intersection in a local north/east NM plane.

    These are geometric observations, not a planner or flight-feasibility test.
    Collinear contact with an edge is conservatively reported as zero.
    """
    angles = np.radians(headings_deg)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    if not len(starts):
        return np.full(len(directions), horizon)
    edges = ends - starts
    relative = starts - origin
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    denominator = cross(directions[:, None], edges[None])
    nonparallel = np.abs(denominator) > 1e-10
    distance = np.divide(cross(relative, edges)[None], denominator,
                         out=np.full_like(denominator, np.inf), where=nonparallel)
    fraction = np.divide(cross(relative[None], directions[:, None]), denominator,
                         out=np.full_like(denominator, np.inf), where=nonparallel)
    valid = nonparallel & (distance >= -1e-8) & (fraction >= -1e-8) & (fraction <= 1 + 1e-8)
    hits = np.where(valid, np.maximum(distance, 0), horizon)
    # Deal with rays lying along a segment (including the segment endpoints).
    collinear = ~nonparallel & (np.abs(cross(relative[None], directions[:, None])) < 1e-8)
    start_projection = np.sum(relative[None] * directions[:, None], axis=2)
    end_projection = np.sum((ends - origin)[None] * directions[:, None], axis=2)
    near = np.minimum(start_projection, end_projection)
    far = np.maximum(start_projection, end_projection)
    hits = np.where(collinear & (far >= 0), np.maximum(near, 0), hits)
    return np.clip(hits.min(axis=1), 0, horizon)


class _PassageNavigation(_TrafficNavigation):
    def __init__(self, env, ac_id):
        super().__init__(env, ac_id)
        config = env.passage_config
        self.max_heading_offset = config.max_heading_offset_deg
        self.restricted_buffer_nm = config.obstacle_buffer_nm
        self.restricted_buffer_release_nm = config.obstacle_release_nm
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "clearance_rays": spaces.Box(0.0, 1.0, shape=(len(RAY_OFFSETS_DEG),), dtype=np.float64),
        })

    def _get_obs(self, ac_id):
        observation = super()._get_obs(ac_id)
        if observation["sector"][3] == 0 or np.any(observation["obstacles"][:, 6] > 0):
            rays = np.zeros(len(RAY_OFFSETS_DEG))
        else:
            idx = bs.traf.id2idx(ac_id)
            starts, ends = self.env.polygon_edges()
            origin = fn.latlong_to_nm(np.array(self.scenario.center),
                                      np.array([bs.traf.lat[idx], bs.traf.lon[idx]]))
            rays = ray_distances(origin, bs.traf.hdg[idx] + RAY_OFFSETS_DEG,
                                 starts, ends, self.env.passage_config.lookahead_nm)
        observation["clearance_rays"] = rays / self.env.passage_config.lookahead_nm
        return observation


class MultiAgentPassageEnv(MultiAgentTrafficEnv):
    """Original world/scoring, with expanded heading mapping and polygon rays."""

    navigation_class = _PassageNavigation
    metadata = {**MultiAgentTrafficEnv.metadata, "name": "navigation_v4"}

    def __init__(self, passage_config=None, reward_config=None, **kwargs):
        self.passage_config = resolve_passage_config(passage_config)
        self._edge_scenario = None
        self._polygon_edges = None
        super().__init__(reward_config=reward_config or make_reward_config("navigation-v4"), **kwargs)

    def reset(self, seed=None, options=None):
        self._edge_scenario = None
        self._polygon_edges = None
        return super().reset(seed=seed, options=options)

    def polygon_edges(self):
        if self._edge_scenario is not self.scenario:
            polygons = [self.scenario.sector, *(obstacle.vertices for obstacle in self.scenario.obstacles)]
            points = [np.array([fn.latlong_to_nm(np.array(self.scenario.center), np.array(vertex))
                                for vertex in polygon]) for polygon in polygons]
            self._polygon_edges = (np.concatenate(points),
                                   np.concatenate([np.roll(point, -1, axis=0) for point in points]))
            self._edge_scenario = self.scenario
        return self._polygon_edges
