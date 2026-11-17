"""Aircraft warning bubbles affect learning/search, not official separation."""

import contextlib
from dataclasses import asdict, replace
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import bluesky as bs
import numpy as np

from bluesky_gym.envs.common import functions as fn
from bluesky_zoo.lookahead import LookaheadConfig
from bluesky_zoo.lookahead_policy import LookaheadPolicy
from bluesky_zoo.navigation_reward import (
    NavigationRewardConfig, aircraft_buffer_risk, make_reward_config, reward_components,
)
from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv
from core.scenario import AgentSpec, Scenario
from core.tools import kwikqdrdist
import test_lookahead_pipeline as lookahead_fixture
from test_navigation_reward import metrics
import train_zoo


def parallel_scenario(distance=6.0):
    center = np.array([52.0, 4.0])
    ll = lambda n, e: tuple(fn.nm_to_latlong(center, np.array([n, e])))
    return Scenario(center=tuple(center),
                    sector=[ll(-100, -100), ll(-100, 100), ll(100, 100), ll(100, -100)],
                    agents=[AgentSpec(a, ll(0, east), ll(80, east), 0, 150)
                            for a, east in (("A", 0), ("B", distance), ("C", -distance))])


class AircraftBufferTests(unittest.TestCase):
    def test_risk_is_smooth_outside_actual_intrusion_and_saturates_inside(self):
        cfg = NavigationRewardConfig(aircraft_buffer_second_cost=0.05)
        for distance in (8, 9, 16, 100):
            self.assertEqual(aircraft_buffer_risk(distance, 5, cfg), 0)
        self.assertAlmostEqual(aircraft_buffer_risk(7, 5, cfg), 1 / 9)
        self.assertAlmostEqual(aircraft_buffer_risk(6, 5, cfg), 4 / 9)
        for distance in (5, 4, 0):
            self.assertEqual(aircraft_buffer_risk(distance, 5, cfg), 1)
        for values in ({"aircraft_buffer_nm": 5}, {"aircraft_buffer_nm": 0},
                       {"aircraft_buffer_nm": np.inf}, {"aircraft_buffer_second_cost": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                NavigationRewardConfig(**values)

    def test_cost_scales_with_seconds_and_each_neighbour_without_mutating_metrics(self):
        cfg = NavigationRewardConfig(aircraft_buffer_second_cost=0.05)
        current = metrics(flight_time=5)
        args = dict(previous_distance_nm=50, distance_nm=49, capture_radius_km=5,
                    metrics=current, baseline=metrics(), risk=0, action=np.zeros(2), previous_action=np.zeros(2))
        one = reward_components(cfg, **args, aircraft_risk=1)["aircraft_buffer"]
        two = reward_components(cfg, **args, aircraft_risk=2)["aircraft_buffer"]
        self.assertEqual(one, -0.25)
        self.assertEqual(two, 2 * one)
        self.assertEqual(current, metrics(flight_time=5))
        args["metrics"] = metrics(flight_time=1)
        self.assertEqual(reward_components(cfg, **args, aircraft_risk=1)["aircraft_buffer"], one / 5)

    def test_new_v5_defaults_and_legacy_profiles_are_reproducible(self):
        self.assertEqual(make_reward_config("navigation-v5").aircraft_buffer_second_cost, 0.05)
        self.assertEqual(make_reward_config("navigation-v4").aircraft_buffer_second_cost, 0)
        old = asdict(NavigationRewardConfig(gamma=0.998, goal_bonus=31))
        old.pop("aircraft_buffer_second_cost")
        old.pop("aircraft_buffer_nm")
        restored = make_reward_config("navigation-v5", saved=old)
        self.assertEqual(restored.aircraft_buffer_second_cost, 0)
        self.assertEqual(restored.goal_bonus, 31)
        enabled = make_reward_config("navigation-v5", saved=old,
                                     aircraft_buffer_second_cost=0.03, aircraft_buffer_nm=9)
        self.assertEqual(enabled.aircraft_buffer_second_cost, 0.03)
        self.assertEqual(enabled.aircraft_buffer_nm, 9)
        self.assertEqual(enabled.goal_bonus, 31)

    def test_both_participants_and_all_neighbours_are_penalized_above_five_nm(self):
        env = MultiAgentStrategicEnv(scenario=parallel_scenario(), n_obstacles=0)
        self.addCleanup(env.close)
        env.reset(seed=42)
        _, _, _, _, infos = env.step({a: np.array([0, -0.4]) for a in env.agents})
        for agent in env.agents:
            i = bs.traf.id2idx(agent)
            risk = sum(aircraft_buffer_risk(kwikqdrdist(
                bs.traf.lat[i], bs.traf.lon[i], bs.traf.lat[bs.traf.id2idx(other)],
                bs.traf.lon[bs.traf.id2idx(other)])[1], 5, env.reward_config)
                for other in env.agents if other != agent)
            parts = env.get_reward_components(agent)
            self.assertAlmostEqual(parts["aircraft_buffer"], -0.05 * 5 * risk)
            self.assertLess(parts["aircraft_buffer"], 0)
            self.assertAlmostEqual(sum(parts.values()), infos[agent]["total_reward"])
            self.assertEqual(infos[agent]["intrusion_time"], 0)
            self.assertEqual(infos[agent]["intrusion_events"], 0)
            self.assertEqual(len(infos[agent]), 9)
        self.assertAlmostEqual(env.get_reward_components("A")["aircraft_buffer"],
                               2 * env.get_reward_components("B")["aircraft_buffer"])
        env.reset(seed=42)
        self.assertTrue(all(env.get_reward_components(a)["aircraft_buffer"] == 0 for a in env.agents))

    def test_actual_intrusion_penalty_remains_and_objective_metrics_do_not_change(self):
        snapshots = []
        for cost in (0, 0.05):
            env = MultiAgentStrategicEnv(scenario=parallel_scenario(4), n_obstacles=0,
                                         reward_config=make_reward_config("navigation-v5",
                                                                         aircraft_buffer_second_cost=cost))
            try:
                env.reset(seed=42)
                _, _, _, _, infos = env.step({a: np.array([0, -0.4]) for a in env.agents})
                snapshots.append({a: {k: v for k, v in info.items() if k != "total_reward"}
                                  for a, info in infos.items()})
                self.assertEqual(env.intrusion_distance, 5)
                for a in env.agents:
                    parts = env.get_reward_components(a)
                    self.assertLess(parts["intrusion"], 0)
                    self.assertEqual(parts["aircraft_buffer"] < 0, cost > 0)
            finally:
                env.close()
        self.assertEqual(*snapshots)

    def test_lookahead_accounts_for_same_proximity_cost(self):
        # Reuse the real tiny-PPO fixture, with the official nonempty input shape.
        env, model = lookahead_fixture.LookaheadPipelineTests.make_model(self, n_agents=2)
        env.reset(seed=37)
        a, b = env.agents
        i, j = bs.traf.id2idx(a), bs.traf.id2idx(b)
        bs.traf.lat[j], bs.traf.lon[j] = fn.nm_to_latlong(
            np.array(env.scenario.center), fn.latlong_to_nm(np.array(env.scenario.center),
            np.array([bs.traf.lat[i], bs.traf.lon[i]])) + np.array([0, 6]))
        config = LookaheadConfig(simulations=1, max_depth=1, rollout_seconds=2)
        enabled = LookaheadPolicy(model, env, config)
        scene = enabled.snapshot()
        commands = {row.agent: (row.motion.heading_deg, 150) for row in scene.aircraft}
        action = np.array([0, -0.4])
        with_cost = enabled.transition(scene, 0, action, commands)
        env.reward_config = replace(env.reward_config, aircraft_buffer_second_cost=0)
        disabled = LookaheadPolicy(model, env, config)
        disabled.snapshot()
        without_cost = disabled.transition(scene, 0, action, commands)
        self.assertLess(with_cost.reward, without_cost.reward)
        self.assertEqual(with_cost.state, without_cost.state)
        self.assertEqual(env.sim_time, scene.time)

    def test_cli_forwards_warning_settings_to_every_curriculum_stage(self):
        path = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("bubble_" + uuid4().hex)
        with patch("train_zoo.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            train_zoo.main(["train", "16", "--mdp", "navigation-v5", "--curriculum", "2", "10",
                            "--model", str(path / "model"), "--metrics", str(path / "metrics.csv"),
                            "--checkpoint-dir", str(path / "checkpoints"), "--eval-episodes", "0",
                            "--aircraft-buffer-nm", "9", "--aircraft-buffer-cost", "0.03"])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--aircraft-buffer-nm") + 1], "9.0")
            self.assertEqual(command[command.index("--aircraft-buffer-cost") + 1], "0.03")


if __name__ == "__main__":
    unittest.main()
