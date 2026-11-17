"""Reward-only MA experiment: same observations, actions, scenarios and scoring."""

from dataclasses import asdict

import bluesky as bs

from bluesky_zoo.multi_agent_env import MultiAgentEnv, _AircraftNavigation
from bluesky_zoo.navigation_reward import (
    NavigationRewardConfig, REWARD_COMPONENT_KEYS, predicted_conflict_risk,
    reward_components, sector_buffer_risk, aircraft_buffer_risk,
)
from core.tools import kwikqdrdist


class _RewardNavigation(_AircraftNavigation):
    def __init__(self, env, ac_id):
        super().__init__(env, ac_id)
        self.reward_component_totals = dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0)
        self.last_reward_components = dict(self.reward_component_totals)

    def _reset_navigation(self):
        super()._reset_navigation()
        self.reward_component_totals = dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0)
        self.last_reward_components = dict(self.reward_component_totals)

    def _get_reward(self, ac_id):
        config = self.env.reward_config
        idx = bs.traf.id2idx(ac_id)
        _, distance_nm = kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx], self.goal_lat, self.goal_lon,
        )
        risk = 0.0
        aircraft_risk = 0.0
        for other in self.intruder_ids:
            other_idx = bs.traf.id2idx(other)
            if other_idx < 0:
                continue
            tcpa, dcpa, distance = self._compute_cpa(idx, other_idx)
            if config.aircraft_buffer_second_cost:
                aircraft_risk += aircraft_buffer_risk(distance, self.env.intrusion_distance, config)
            risk = max(risk, predicted_conflict_risk(
                tcpa, dcpa, distance, self.env.intrusion_distance, config,
            ))
        sector_risk = 0.0
        if config.sector_buffer_second_cost:
            sector = self._get_sector_state(idx)
            sector_risk = sector_buffer_risk(sector["clearance"], sector["inside"], config)
        components = reward_components(
            config, previous_distance_nm=self.previous_goal_distance,
            distance_nm=distance_nm, capture_radius_km=self.env.distance_margin,
            metrics=self.metrics[ac_id], baseline=self._reward_baseline,
            risk=risk, action=self.last_action, previous_action=self.previous_action,
            sector_risk=sector_risk, aircraft_risk=aircraft_risk,
        )
        self.previous_goal_distance = distance_nm
        self.last_reward_components = components
        for key, value in components.items():
            self.reward_component_totals[key] += value
        return float(sum(components.values()))


class MultiAgentRewardEnv(MultiAgentEnv):
    """Opt-in ``navigation-v2`` reward; compatible with existing policy weights.

    No scoring/info/step overrides. Component totals are exposed separately to
    the training adapter; the official nine-metric info remains unchanged.
    """

    navigation_class = _RewardNavigation

    def __init__(self, reward_config=None, **kwargs):
        if isinstance(reward_config, dict):
            reward_config = NavigationRewardConfig(**reward_config)
        self.reward_config = reward_config or NavigationRewardConfig()
        super().__init__(**kwargs)

    def get_reward_components(self, agent):
        return dict(self.navigation_states[agent].reward_component_totals)

    def get_reward_config(self):
        return asdict(self.reward_config)
