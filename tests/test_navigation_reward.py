"""Reward incentives, simulator scoring isolation and per-aircraft diagnostics."""

from copy import deepcopy
import unittest

import numpy as np

from bluesky_zoo.multi_agent_env import MultiAgentEnv
from bluesky_zoo.multi_agent_reward_env import MultiAgentRewardEnv
from bluesky_zoo.navigation_reward import (
    NavigationRewardConfig, REWARD_COMPONENT_KEYS, distance_potential,
    predicted_conflict_risk, reward_components,
)
from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv


def metrics(**updates):
    result = dict.fromkeys((
        "waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
        "restricted_area_events", "time_in_restricted_area", "sector_exit_events",
        "time_outside_sector", "total_reward",
    ), 0.0)
    result.update(updates)
    return result


class NavigationRewardTests(unittest.TestCase):
    def components(self, before=50.0, after=49.0, config=None, **kwargs):
        args = dict(
            previous_distance_nm=before, distance_nm=after, capture_radius_km=5,
            metrics=metrics(flight_time=5), baseline=metrics(), risk=0,
            action=np.zeros(2), previous_action=np.zeros(2),
        )
        args.update(kwargs)
        return reward_components(config or NavigationRewardConfig(), **args)

    def test_closed_loop_has_no_extra_discounted_shaping_gain(self):
        cfg = NavigationRewardConfig(gamma=0.998)
        # Visit both a closer point and a farther point, ending where we started.
        loop = sum(cfg.gamma ** i * self.components(a, b, cfg)["progress"]
                   for i, (a, b) in enumerate(((50, 49), (49, 51), (51, 50))))
        stationary = sum(cfg.gamma ** i * self.components(50, 50, cfg)["progress"]
                         for i in range(3))
        self.assertAlmostEqual(loop, stationary)
        terminal = cfg.progress_weight * (cfg.gamma ** 3 - 1) * distance_potential(50, 5, cfg)
        self.assertAlmostEqual(loop, terminal)

    def test_safe_detour_beats_shortcut_through_a_conflict(self):
        safe = self.components(after=50.5)
        unsafe = self.components(after=49, metrics=metrics(
            flight_time=5, intrusion_events=1, intrusion_time=5,
        ))
        self.assertGreater(sum(safe.values()), sum(unsafe.values()))

    def test_entry_and_exit_cannot_farm_restricted_area_bonuses(self):
        entry = self.components(metrics=metrics(
            flight_time=5, restricted_area_events=1, time_in_restricted_area=5,
        ))
        exit_reward = self.components(
            metrics=metrics(flight_time=10, restricted_area_events=1, time_in_restricted_area=5),
            baseline=metrics(flight_time=5, restricted_area_events=1, time_in_restricted_area=5),
        )
        self.assertEqual(entry["restricted_area"], -5.25)
        self.assertEqual(exit_reward["restricted_area"], 0)
        self.assertLess(entry["restricted_area"] + exit_reward["restricted_area"], 0)

    def test_goal_bonus_is_once_and_timeout_keeps_nonterminal_potential(self):
        arrival = self.components(metrics=metrics(flight_time=1, waypoint_reached=1), after=2)
        self.assertEqual(arrival["goal"], 20)
        again = self.components(metrics=metrics(flight_time=2, waypoint_reached=1),
                                baseline=metrics(flight_time=1, waypoint_reached=1), after=2)
        self.assertEqual(again["goal"], 0)
        timeout = self.components(after=49, metrics=metrics(flight_time=3000),
                                  baseline=metrics(flight_time=2995))
        ordinary = self.components(after=49)
        self.assertEqual(timeout["progress"], ordinary["progress"])
        self.assertEqual(timeout["goal"], 0)

    def test_duration_costs_scale_with_actual_seconds_and_do_not_mutate_metrics(self):
        baseline = metrics()
        current = metrics(flight_time=5, intrusion_time=10,
                          time_in_restricted_area=5, time_outside_sector=5)
        snapshot = deepcopy(current)
        five = self.components(metrics=current, baseline=baseline, risk=0.5)
        one = self.components(metrics=metrics(flight_time=1, intrusion_time=2,
                                             time_in_restricted_area=1, time_outside_sector=1), risk=0.5)
        for key in ("intrusion", "restricted_area", "sector", "flight_time", "predicted_conflict"):
            self.assertAlmostEqual(five[key], 5 * one[key])
        self.assertEqual(current, snapshot)
        self.assertEqual(baseline, metrics())

    def test_prediction_warns_before_intrusion_and_configuration_is_validated(self):
        cfg = NavigationRewardConfig()
        far = predicted_conflict_risk(100, 6, 12, 5, cfg)
        near = predicted_conflict_risk(30, 2, 8, 5, cfg)
        self.assertGreater(near, far)
        self.assertEqual(predicted_conflict_risk(30, 2, 4, 5, cfg), 0)
        self.assertEqual(predicted_conflict_risk(-1, 2, 8, 5, cfg), 0)
        for kwargs in ({"gamma": 0}, {"gamma": 1.1}, {"distance_scale_km": 0},
                       {"goal_bonus": float("nan")}, {"intrusion_second_cost": -1},
                       {"conflict_horizon_seconds": 900}):
            with self.assertRaises(ValueError):
                NavigationRewardConfig(**kwargs)


class RewardEnvironmentTests(unittest.TestCase):
    def test_same_policy_actions_produce_identical_official_safety_metrics(self):
        snapshots = []
        spaces = []
        for env_class in (MultiAgentEnv, MultiAgentRewardEnv):
            env = env_class(n_agents=10)
            try:
                observations, _ = env.reset(seed=901)
                spaces.append((env.observation_space(env.agents[0]), env.action_space(env.agents[0])))
                for _ in range(4):
                    observations, _, _, _, _ = env.step({a: np.zeros(2) for a in env.agents})
                snapshots.append({a: {k: v for k, v in info.items() if k != "total_reward"}
                                  for a, info in env.metrics.items()})
                if env_class is MultiAgentRewardEnv:
                    for agent, info in env.metrics.items():
                        self.assertAlmostEqual(sum(env.get_reward_components(agent).values()), info["total_reward"])
                        self.assertEqual(set(env._get_info(agent)), set(metrics()))
                    env.reset(seed=902)
                    self.assertTrue(all(value == 0 for agent in env.agents
                                        for value in env.get_reward_components(agent).values()))
            finally:
                env.close()
        self.assertEqual(spaces[0], spaces[1])
        self.assertEqual(snapshots[0], snapshots[1])

    def test_timeout_diagnostics_reach_adapter_without_changing_official_info(self):
        env = MultiAgentRewardEnv(n_agents=2, episode_time_limit=5)
        vec = SharedPolicyVecEnv(env)
        try:
            vec.seed(123)
            vec.reset()
            _, _, dones, infos = vec.step(np.zeros((2, 2)))
            self.assertTrue(all(dones))
            for info in infos:
                self.assertTrue(info["TimeLimit.truncated"])
                self.assertEqual(set(info["final_metrics"]), set(metrics()))
                self.assertEqual(set(info["reward_components"]), set(REWARD_COMPONENT_KEYS))
                self.assertAlmostEqual(sum(info["reward_components"].values()),
                                       info["final_metrics"]["total_reward"])
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
