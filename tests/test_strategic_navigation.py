"""Shrinking zones remain aircraft-local observations and training costs."""

from dataclasses import asdict
import unittest

import bluesky as bs
import numpy as np

from bluesky_zoo.competition.competition import CompetitionZooEnv
from bluesky_zoo.passage_navigation import MultiAgentPassageEnv
from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv
from bluesky_zoo.strategic_navigation import (
    MultiAgentStrategicEnv, ZoneConfig, ZoneSchedule, make_zone_schedule,
    resolve_zone_config, zone_penalty,
)
from core.scenario import ScenarioGenerator
from test_multi_agent_env import fixed_scenario


class ZoneScheduleTests(unittest.TestCase):
    def test_monotonic_distance_adaptive_schedule_and_capture_floor(self):
        config = ZoneConfig()
        short = make_zone_schedule(20, 5, 3000, config)
        long = make_zone_schedule(80, 5, 3000, config)
        self.assertAlmostEqual(short.initial_radius_nm, 30)
        self.assertGreater(long.end_seconds, short.end_seconds)
        self.assertEqual(short.radius_nm(0), short.radius_nm(config.grace_seconds))
        samples = [long.radius_nm(t) for t in np.linspace(0, 4000, 101)]
        self.assertTrue(all(after <= before for before, after in zip(samples, samples[1:])))
        self.assertAlmostEqual(long.radius_nm(long.end_seconds), 5 / 1.852)
        self.assertAlmostEqual(long.radius_nm(4000), 5 / 1.852)
        distant = make_zone_schedule(300, 5, 3000, config)
        self.assertEqual(distant.end_seconds, 3000)
        self.assertAlmostEqual(distant.radius_nm(3000), 5 / 1.852)

    def test_penalty_has_no_crossing_bonus_scales_in_seconds_and_is_capped(self):
        config = ZoneConfig(outside_second_cost=0.01, excess_scale_nm=10)
        self.assertEqual(zone_penalty(19, 20, 5, config), 0)
        self.assertEqual(zone_penalty(20, 20, 5, config), 0)
        self.assertAlmostEqual(zone_penalty(25, 20, 5, config), -0.0125)
        self.assertAlmostEqual(zone_penalty(25, 20, 5, config),
                               5 * zone_penalty(25, 20, 1, config))
        self.assertEqual(zone_penalty(30, 20, 5, config), -0.05)
        self.assertEqual(zone_penalty(300, 20, 5, config), -0.05)
        self.assertEqual(zone_penalty(300, 20, 0, config), 0)
        disabled = ZoneConfig(outside_second_cost=0)
        self.assertEqual(zone_penalty(300, 20, 5, disabled), 0)

    def test_configuration_roundtrip_and_invalid_values(self):
        config = ZoneConfig(detour_allowance_nm=15, outside_second_cost=0.005)
        self.assertEqual(resolve_zone_config(asdict(config)), config)
        self.assertEqual(resolve_zone_config(config, grace_seconds=None), config)
        self.assertEqual(resolve_zone_config(config, grace_seconds=60).grace_seconds, 60)
        for kwargs in ({"outside_second_cost": -1}, {"closing_speed_kt": 0},
                       {"arrival_time_factor": 0.5}, {"minimum_shrink_seconds": 0},
                       {"excess_scale_nm": 0}, {"grace_seconds": np.nan},
                       {"detour_allowance_nm": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ZoneConfig(**kwargs)
        for distance, capture, time in ((-1, 5, 3000), (20, 0, 3000), (20, 5, 0)):
            with self.assertRaises(ValueError):
                make_zone_schedule(distance, capture, time, config)
        for values in ((2, 3, 0, 10), (10, 0, 0, 10), (10, 3, 10, 10),
                       (np.inf, 3, 0, 10)):
            with self.assertRaises(ValueError):
                ZoneSchedule(*values)
        # Short test episodes still reach the normal capture radius at their
        # inherited timeout; the schedule never extends that timeout.
        short = make_zone_schedule(80, 5, 5, config)
        self.assertLess(short.grace_seconds, short.end_seconds)
        self.assertEqual(short.end_seconds, 5)
        self.assertAlmostEqual(short.radius_nm(5), 5 / 1.852)


class StrategicEnvironmentTests(unittest.TestCase):
    def test_world_lifecycle_and_scoring_are_inherited_with_original_defaults(self):
        self.assertIs(MultiAgentStrategicEnv.step, CompetitionZooEnv.step)
        self.assertIs(MultiAgentStrategicEnv._update_metrics, CompetitionZooEnv._update_metrics)
        self.assertIs(MultiAgentStrategicEnv._get_info, CompetitionZooEnv._get_info)
        env = MultiAgentStrategicEnv()
        try:
            self.assertEqual(env.n_agents, 10)
            self.assertEqual(env.n_obstacles, 5)
            self.assertEqual(env.intrusion_distance, 5)
            self.assertEqual(env.distance_margin, 5)
            self.assertEqual(env.episode_time_limit, 3000)
            self.assertEqual(env.sim_dt, 1)
            self.assertIsInstance(env.scenario_generator, ScenarioGenerator)
            observations, infos = env.reset(seed=27)
            for agent, observation in observations.items():
                self.assertTrue(env.observation_space(agent).contains(observation))
                self.assertEqual(len(infos[agent]), 9)
                self.assertNotIn("zone", infos[agent])
            self.assertEqual(env.get_zone_config(), asdict(env.zone_config))
        finally:
            env.close()

    def test_zone_never_restarts_from_current_position_and_resets_per_episode(self):
        env = MultiAgentStrategicEnv(n_agents=2)
        try:
            observations, _ = env.reset(seed=31)
            a, b = env.agents
            first = env.navigation_states[a]
            other = env.navigation_states[b]
            schedule = first.zone_schedule
            self.assertIsNot(schedule, other.zone_schedule)
            original_other = other.zone_schedule
            initial_observation = observations[a]["zone"].copy()
            env.sim_time = 500
            idx = bs.traf.id2idx(a)
            bs.traf.lat[idx] += 0.1
            later = env._get_obs(a)["zone"]
            self.assertIs(first.zone_schedule, schedule)
            self.assertIs(other.zone_schedule, original_other)
            self.assertLess(later[0], initial_observation[0])
            self.assertLess(later[2], initial_observation[2])
            self.assertTrue(env.observation_space(a).contains(env._get_obs(a)))
            reset_observations, _ = env.reset(seed=31)
            self.assertIsNot(first.zone_schedule, schedule)
            self.assertEqual(first.zone_schedule, schedule)
            np.testing.assert_allclose(reset_observations[a]["zone"], initial_observation)
            self.assertEqual(env.get_reward_components(a)["zone"], 0)
        finally:
            env.close()

    def test_outside_zone_adds_only_a_reward_cost_and_preserves_objective_metrics(self):
        snapshots = []
        for cls in (MultiAgentPassageEnv, MultiAgentStrategicEnv):
            env = cls(n_agents=4)
            try:
                env.reset(seed=815)
                if cls is MultiAgentStrategicEnv:
                    for state in env.navigation_states.values():
                        # A virtual circle with no remaining slack; no physical
                        # airspace or lifecycle property is changed.
                        capture_nm = env.distance_margin / 1.852
                        state.zone_schedule = ZoneSchedule(capture_nm, capture_nm, 0, 1)
                for _ in range(3):
                    _, rewards, terminations, truncations, infos = env.step(
                        {agent: np.zeros(2) for agent in env.agents})
                    if cls is MultiAgentStrategicEnv:
                        for agent, reward in rewards.items():
                            state = env.navigation_states[agent]
                            self.assertAlmostEqual(sum(state.last_reward_components.values()), reward)
                            self.assertAlmostEqual(sum(env.get_reward_components(agent).values()),
                                                   env.metrics[agent]["total_reward"])
                            self.assertLessEqual(state.last_reward_components["zone"], 0)
                            self.assertNotIn("zone", infos[agent])
                        self.assertTrue(env.agents)
                        self.assertFalse(any(terminations.values()))
                        self.assertFalse(any(truncations.values()))
                snapshots.append({agent: {key: value for key, value in data.items() if key != "total_reward"}
                                  for agent, data in env.metrics.items()})
            finally:
                env.close()
        self.assertEqual(*snapshots)

    def test_early_arrivals_and_timeouts_keep_actual_seconds_and_adapter_diagnostics(self):
        env = MultiAgentStrategicEnv(scenario=fixed_scenario(first_at_goal=True), n_obstacles=0,
                                     episode_time_limit=5)
        vec = SharedPolicyVecEnv(env)
        try:
            vec.seed(4)
            vec.reset()
            _, _, dones, infos = vec.step(np.zeros((3, 2)))
            self.assertTrue(all(dones))
            early = infos[0]
            self.assertEqual(early["final_metrics"]["waypoint_reached"], 1)
            self.assertEqual(early["final_metrics"]["flight_time"], 1)
            self.assertFalse(early["TimeLimit.truncated"])
            for info in infos[1:]:
                self.assertEqual(info["final_metrics"]["flight_time"], 5)
                self.assertTrue(info["TimeLimit.truncated"])
                # The bounded zone cost cannot exceed flown seconds * rate.
                self.assertGreaterEqual(info["reward_components"]["zone"],
                                        -5 * env.zone_config.outside_second_cost)
            for info in infos:
                self.assertEqual(set(info["reward_components"]), set(env.reward_component_keys))
                self.assertAlmostEqual(sum(info["reward_components"].values()),
                                       info["final_metrics"]["total_reward"])
                self.assertEqual(len(info["final_metrics"]), 9)
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
