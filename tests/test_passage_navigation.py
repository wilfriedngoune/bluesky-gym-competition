"""Narrow gaps remain geometrically visible and reachable by the action mapping."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
import csv
import unittest

import bluesky as bs
import numpy as np

from bluesky_gym.envs.common import functions as fn
from bluesky_zoo.multi_agent_traffic_env import MultiAgentTrafficEnv
from bluesky_zoo.passage_navigation import (MultiAgentPassageEnv, PassageNavigationConfig,
                                           RAY_OFFSETS_DEG, ray_distances)
from bluesky_zoo.passage_scenarios import PassageTrainingGenerator, narrow_passage_scenario
from core.scenario import ScenarioGenerator
from scripts.evaluate_competition import make_env, policy_reward_kwargs
from scripts.navigation_trace import NavigationTrace
from test_multi_agent_env import fixed_scenario
from train_zoo import train


def straight_gap():
    scenario = narrow_passage_scenario(gap_nm=4)
    center = np.array(scenario.center)
    ll = lambda point: tuple(fn.nm_to_latlong(center, np.array(point)))
    scenario.agents = [replace(scenario.agents[0], start=ll((-30, 0)), goal=ll((30, 0))),
                       replace(scenario.agents[1], start=ll((-30, -60)), goal=ll((30, -60)))]
    return scenario


class PassageNavigationTests(unittest.TestCase):
    def test_rays_handle_nearest_intersection_parallel_behind_and_collinear(self):
        starts = np.array([[5, -2], [8, -2]])
        ends = np.array([[5, 2], [8, 2]])
        np.testing.assert_allclose(ray_distances(np.zeros(2), [0, 90, 180], starts, ends, 20), [5, 20, 20])
        np.testing.assert_allclose(ray_distances(np.array([5, -5]), [90], starts[:1], ends[:1], 20), [3])
        np.testing.assert_allclose(ray_distances(np.array([5, 0]), [90], starts[:1], ends[:1], 20), [0])

    def test_gap_is_visible_and_zero_offset_can_cross_without_restricted_entries(self):
        env = MultiAgentPassageEnv(scenario=straight_gap())
        try:
            obs, _ = env.reset(seed=1)
            self.assertTrue(env.observation_space(env.agents[0]).contains(obs[env.agents[0]]))
            # Move observation origin to the gap's midpoint; exact straight
            # directions remain open while both lateral rays see the walls.
            idx = bs.traf.id2idx(env.agents[0])
            bs.traf.lat[idx], bs.traf.lon[idx] = env.scenario.center
            middle = env._get_obs(env.agents[0])["clearance_rays"]
            self.assertEqual(middle[list(RAY_OFFSETS_DEG).index(0)], 1)
            self.assertAlmostEqual(middle[list(RAY_OFFSETS_DEG).index(90)], 0.1, places=6)
            self.assertAlmostEqual(middle[list(RAY_OFFSETS_DEG).index(-90)], 0.1, places=6)
            # Reset after the synthetic observation check; now exercise BlueSky.
            env.reset(seed=1)
            for _ in range(600):
                if not env.agents:
                    break
                env.step({agent: np.zeros(2) for agent in env.agents})
            first = env.metrics[env.possible_agents[0]]
            self.assertEqual(first["waypoint_reached"], 1)
            self.assertEqual(first["restricted_area_events"], 0)
            self.assertEqual(first["time_in_restricted_area"], 0)
        finally:
            env.close()

    def test_parallel_heading_is_available_even_with_legacy_buffer(self):
        config = PassageNavigationConfig(obstacle_buffer_nm=3, obstacle_release_nm=4)
        env = MultiAgentPassageEnv(n_agents=2, passage_config=config)
        try:
            env.reset(seed=5)
            state = env.navigation_states[env.agents[0]]
            with patch.object(state, "_get_navigation_reference", return_value=(90, 1)), \
                    patch.object(bs.stack, "stack"):
                env._get_action(env.agents[0], [-0.5, 0])
            self.assertAlmostEqual(state.last_target_heading, 0)
            self.assertEqual(state.max_heading_offset, 180)
        finally:
            env.close()

    def test_spaces_fixed_across_density_and_inside_obstacle_rays_zero(self):
        space = None
        for count in (2, 4, 6, 10):
            env = MultiAgentPassageEnv(n_agents=count)
            try:
                observations, _ = env.reset(seed=7)
                current = env.observation_space(env.agents[0])
                if space is None:
                    space = current
                self.assertEqual(space, current)
                self.assertTrue(all(current.contains(obs) for obs in observations.values()))
                idx = bs.traf.id2idx(env.agents[0])
                vertices = np.array(env.scenario.obstacles[0].vertices)
                bs.traf.lat[idx], bs.traf.lon[idx] = vertices.mean(axis=0)
                # For convex generated polygons the vertex mean is internal.
                np.testing.assert_array_equal(env._get_obs(env.agents[0])["clearance_rays"], np.zeros(24))
                old_edges = env.polygon_edges()[0].copy()
                env.reset(seed=8)
                self.assertFalse(np.array_equal(old_edges, env.polygon_edges()[0]))
            finally:
                env.close()

    def test_identical_issued_commands_preserve_all_official_metrics(self):
        snapshots = []
        for cls in (MultiAgentTrafficEnv, MultiAgentPassageEnv):
            env = cls(n_agents=4)
            try:
                env.reset(seed=815)
                with patch.object(env.navigation_class, "_get_navigation_reference", return_value=(0, 0)):
                    for _ in range(3):
                        env.step({agent: np.zeros(2) for agent in env.agents})
                snapshots.append({agent: {key: value for key, value in data.items() if key != "total_reward"}
                                  for agent, data in env.metrics.items()})
            finally:
                env.close()
        self.assertEqual(*snapshots)

    def test_custom_training_is_seeded_and_evaluation_uses_original_generator(self):
        generator = PassageTrainingGenerator(ScenarioGenerator(n_agents=4), 4, 1)
        self.assertEqual(generator.generate(np.random.default_rng(4)), generator.generate(np.random.default_rng(4)))
        original = ScenarioGenerator(n_agents=4)
        wrapper = PassageTrainingGenerator(original, 4, 0)
        self.assertEqual(wrapper.generate(np.random.default_rng(5)), original.generate(np.random.default_rng(5)))
        env = make_env("ma", mdp="navigation-v4")
        try:
            self.assertIsInstance(env.scenario_generator, ScenarioGenerator)
            self.assertEqual(env.n_agents, 10)
            self.assertEqual(env.n_obstacles, 5)
            self.assertEqual(env.episode_time_limit, 3000)
            self.assertEqual(env.intrusion_distance, 5)
        finally:
            env.close()

    def test_saved_profile_reconstructs_and_v3_inputs_cannot_initialize_v4(self):
        saved = asdict(PassageNavigationConfig(obstacle_buffer_nm=0.25, obstacle_release_nm=0.75, lookahead_nm=12))
        act = lambda obs: np.zeros(2)
        act.model = SimpleNamespace(gamma=0.998, competition_passage_config=saved)
        env = make_env("ma", mdp="navigation-v4", n_agents=2, **policy_reward_kwargs(act, "navigation-v4"))
        try:
            self.assertEqual(asdict(env.passage_config), saved)
        finally:
            env.close()
        old = MultiAgentTrafficEnv(n_agents=2)
        model = SimpleNamespace(observation_space=old.observation_space(old.possible_agents[0]),
                                action_space=old.action_space(old.possible_agents[0]))
        old.close()
        with patch("train_zoo.PPO.load", return_value=model):
            with self.assertRaisesRegex(ValueError, "navigation-v4 adds clearance rays"):
                train(4, mdp="navigation-v4", n_agents=2, n_steps=2, batch_size=2,
                      init_model="v3", checkpoint_freq=0, eval_episodes=0)
        for kwargs in ({"obstacle_buffer_nm": -1}, {"obstacle_release_nm": 0},
                       {"lookahead_nm": 0}, {"max_heading_offset_deg": 181}):
            with self.assertRaises(ValueError):
                PassageNavigationConfig(**kwargs)

    def test_trace_includes_early_arrivals_after_aircraft_deletion(self):
        env = MultiAgentTrafficEnv(scenario=fixed_scenario(first_at_goal=True), n_obstacles=0)
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"trace_{uuid4().hex}.csv"
        self.addCleanup(path.unlink, missing_ok=True)
        trace = NavigationTrace(path)
        try:
            env.reset(seed=4)
            agents = list(env.agents)
            before = trace.before(env, agents)
            actions = {agent: np.zeros(2) for agent in agents}
            _, _, _, _, infos = env.step(actions)
            trace.after(env, before, actions, infos, 0)
        finally:
            trace.close()
            env.close()
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        arrival = next(row for row in rows if row["agent_id"] == "A")
        self.assertEqual(float(arrival["time_after"]), 1)
        self.assertEqual(float(arrival["waypoint_reached"]), 1)
        self.assertEqual(float(arrival["time_before"]), 0)


if __name__ == "__main__":
    unittest.main()
