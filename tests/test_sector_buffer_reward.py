"""Optional early sector warnings must preserve scoring and old policies."""

from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bluesky_zoo.multi_agent_traffic_env import MultiAgentTrafficEnv
from bluesky_zoo.navigation_reward import (
    NavigationRewardConfig, make_reward_config, reward_components,
    sector_buffer_risk,
)
from scripts.evaluate_competition import policy_reward_kwargs
from test_navigation_reward import metrics


class SectorBufferRewardTests(unittest.TestCase):
    def components(self, *, config=None, elapsed=5, risk=1, **kwargs):
        args = dict(
            previous_distance_nm=50, distance_nm=49, capture_radius_km=5,
            metrics=metrics(flight_time=elapsed), baseline=metrics(), risk=0,
            sector_risk=risk, action=np.zeros(2), previous_action=np.zeros(2),
        )
        args.update(kwargs)
        config = config or NavigationRewardConfig(
            gamma=0.998, sector_buffer_second_cost=0.01,
        )
        return reward_components(config, **args)

    def test_warning_is_local_monotonic_and_continuous_at_boundary(self):
        config = NavigationRewardConfig(sector_buffer_nm=5)
        risks = [sector_buffer_risk(distance, True, config)
                 for distance in (0, 1, 2.5, 4, 5, 20)]
        self.assertTrue(all(left >= right for left, right in zip(risks, risks[1:])))
        self.assertGreater(risks[1], risks[2])
        self.assertGreater(risks[2], risks[3])
        self.assertEqual(risks[-2:], [0, 0])
        self.assertAlmostEqual(sector_buffer_risk(1e-8, True, config),
                               sector_buffer_risk(0, False, config))

    def test_flying_farther_outside_never_removes_the_warning(self):
        config = NavigationRewardConfig()
        for clearance in (0, 0.1, 5, 1000):
            self.assertEqual(sector_buffer_risk(clearance, False, config), 1)

    def test_default_config_keeps_existing_reward_even_when_near_boundary(self):
        config = make_reward_config("navigation-v3")
        safe = self.components(config=config, risk=0)
        near_boundary = self.components(config=config, risk=1)
        self.assertEqual(config.sector_buffer_second_cost, 0)
        self.assertEqual(safe, near_boundary)
        self.assertEqual(near_boundary["sector_buffer"], 0)

    def test_actual_seconds_set_cost_and_returning_inside_pays_no_bonus(self):
        one_second = self.components(elapsed=1, risk=0.5)["sector_buffer"]
        five_seconds = self.components(elapsed=5, risk=0.5)["sector_buffer"]
        self.assertLess(one_second, 0)
        self.assertAlmostEqual(five_seconds, 5 * one_second)
        # Entry, return to the safe interior, and another entry cannot farm a
        # positive warning reward; metric event penalties remain separate.
        crossing_rewards = [self.components(risk=risk)["sector_buffer"]
                            for risk in (1, 0, 1, 0)]
        self.assertEqual(crossing_rewards[1], 0)
        self.assertTrue(all(reward <= 0 for reward in crossing_rewards))
        self.assertLess(sum(crossing_rewards), 0)

    def test_goal_and_timeout_charge_only_actual_elapsed_risk(self):
        ordinary = self.components(elapsed=3, risk=0.5)
        arrival = self.components(
            risk=0.5, metrics=metrics(flight_time=103, waypoint_reached=1),
            baseline=metrics(flight_time=100),
        )
        timeout = self.components(
            risk=0.5, metrics=metrics(flight_time=3000),
            baseline=metrics(flight_time=2997),
        )
        self.assertEqual(arrival["sector_buffer"], ordinary["sector_buffer"])
        self.assertEqual(timeout["sector_buffer"], ordinary["sector_buffer"])
        self.assertGreater(arrival["goal"], 0)
        self.assertEqual(timeout["goal"], 0)

    def test_old_saved_config_loads_disabled_and_explicit_settings_round_trip(self):
        legacy = {key: value for key, value in asdict(make_reward_config("navigation-v3")).items()
                  if key not in {"sector_buffer_second_cost", "sector_buffer_nm"}}
        restored = NavigationRewardConfig(**legacy)
        self.assertEqual(restored.sector_buffer_second_cost, 0)
        self.assertEqual(restored.sector_buffer_nm, 5)
        configured = make_reward_config(
            "navigation-v3", sector_buffer_second_cost=0.01, sector_buffer_nm=7,
        )
        for saved in (legacy, asdict(configured)):
            act = lambda observation: np.zeros(2)
            act.model = SimpleNamespace(gamma=0.998, competition_reward_config=saved)
            env = MultiAgentTrafficEnv(n_agents=2, **policy_reward_kwargs(act, "navigation-v3"))
            try:
                self.assertEqual(env.reward_config, NavigationRewardConfig(**saved))
            finally:
                env.close()

    def test_invalid_costs_and_buffer_width_are_rejected(self):
        for options in ({"sector_buffer_second_cost": -0.01},
                        {"sector_buffer_second_cost": float("nan")},
                        {"sector_buffer_nm": 0}, {"sector_buffer_nm": -5}):
            with self.assertRaises(ValueError):
                NavigationRewardConfig(**options)
        with self.assertRaises(ValueError):
            make_reward_config("base", sector_buffer_second_cost=0.01)


class SectorBufferEnvironmentTests(unittest.TestCase):
    def test_configured_warning_changes_only_reward_for_identical_actions(self):
        snapshots, spaces, rewards = [], [], []
        for cost in (0, 0.01):
            # A large buffer makes this test exercise the enabled reward for
            # every aircraft, without altering the generated world or flight.
            config = make_reward_config("navigation-v3", sector_buffer_second_cost=cost,
                                        sector_buffer_nm=10000)
            env = MultiAgentTrafficEnv(n_agents=4, reward_config=config)
            try:
                env.reset(seed=815)
                agent = env.agents[0]
                spaces.append((env.observation_space(agent), env.action_space(agent)))
                for _ in range(3):
                    env.step({aircraft: np.zeros(2) for aircraft in env.agents})
                snapshots.append({aircraft: {key: value for key, value in values.items()
                                             if key != "total_reward"}
                                  for aircraft, values in env.metrics.items()})
                rewards.append({aircraft: values["total_reward"]
                                for aircraft, values in env.metrics.items()})
                for aircraft in env.agents:
                    components = env.get_reward_components(aircraft)
                    self.assertAlmostEqual(sum(components.values()), rewards[-1][aircraft])
                    if cost:
                        self.assertLess(components["sector_buffer"], 0)
                env.reset(seed=816)
                self.assertTrue(all(env.get_reward_components(aircraft)["sector_buffer"] == 0
                                    for aircraft in env.agents))
            finally:
                env.close()
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(spaces[0], spaces[1])
        for aircraft in rewards[0]:
            self.assertLess(rewards[1][aircraft], rewards[0][aircraft])

    def test_disabled_reward_does_not_query_extra_geometry(self):
        env = MultiAgentTrafficEnv(n_agents=2)
        try:
            env.reset(seed=817)
            aircraft = env.agents[0]
            state = env.navigation_states[aircraft]
            env._reward_baseline[aircraft] = dict(env.metrics[aircraft])
            # Observation and action mapping already query sector geometry;
            # this checks only the additional reward path.
            with patch.object(state, "_get_sector_state", side_effect=AssertionError("extra geometry")):
                state._get_reward(aircraft)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
