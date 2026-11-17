"""Less frequent manoeuvres retain per-second scoring and saved policy timing."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import bluesky as bs
import numpy as np

from bluesky_gym.envs.common.decision_timing import (
    forecast_segment_seconds, rescale_discount, resolve_decision_seconds, validate_decision_seconds,
)
from bluesky_gym.envs.common import functions as fn
from bluesky_zoo.lookahead import LookaheadConfig
from bluesky_zoo.lookahead_policy import LookaheadPolicy
from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv
from core.scenario import AgentSpec, Scenario
from scripts import evaluate_competition as evaluation
import test_lookahead_pipeline as lookahead_fixture
from test_multi_agent_env import fixed_scenario
import train_zoo


class DecisionTimingTests(unittest.TestCase):
    def test_defaults_validation_and_physical_discount(self):
        self.assertEqual(resolve_decision_seconds("navigation-v5"), 5)
        self.assertEqual(resolve_decision_seconds("base"), 10)
        self.assertEqual(resolve_decision_seconds("navigation-v5", saved=15), 15)
        self.assertEqual(resolve_decision_seconds("navigation-v5", saved=15, override=10), 10)
        for seconds in (1, 5, 10, 15, 20, 30, 60, 120):
            self.assertEqual(validate_decision_seconds(seconds), seconds)
            self.assertEqual(3000 % seconds, 0)
        for invalid in (0, -5, 7, 121, 15.0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_decision_seconds(invalid)
        adjusted = rescale_discount(0.998, 5, 15)
        self.assertAlmostEqual(adjusted ** (300 / 15), 0.998 ** (300 / 5))
        self.assertEqual(rescale_discount(adjusted, 15, 15), adjusted)
        self.assertAlmostEqual(rescale_discount(adjusted, 15, 5), 0.998)
        self.assertEqual(forecast_segment_seconds(5, 15), 15)
        self.assertEqual(forecast_segment_seconds(15, 10), 20)
        self.assertEqual(forecast_segment_seconds(120, 100), 100)

    def test_targets_are_held_and_metrics_sample_every_second_until_timeout(self):
        env = MultiAgentStrategicEnv(scenario=fixed_scenario(), n_obstacles=0,
                                     action_frequency=15, episode_time_limit=30)
        self.addCleanup(env.close)
        env.reset(seed=4)
        samples = {a: [] for a in env.agents}
        original_update = env._update_metrics

        def sample(agent, *args, **kwargs):
            result = original_update(agent, *args, **kwargs)
            state = env.navigation_states[agent]
            samples[agent].append((env.sim_time, state.last_target_heading, state.last_target_speed))
            return result

        with patch.object(env, "_get_action", wraps=env._get_action) as commands, \
                patch.object(env, "_update_metrics", side_effect=sample):
            for decision in range(2):
                active = list(env.agents)
                actions = {a: np.array([0.1 * decision, -0.4]) for a in active}
                _, _, terminations, truncations, infos = env.step(actions)
                self.assertEqual(commands.call_count, (decision + 1) * 3)
                self.assertEqual(env.sim_time, 15 * (decision + 1))
                self.assertFalse(any(terminations.values()))
                self.assertEqual(all(truncations.values()), decision == 1)
                for info in infos.values():
                    self.assertEqual(info["flight_time"], env.sim_time)
                    self.assertEqual(len(info), 9)
        self.assertFalse(env.agents)
        for agent, history in samples.items():
            self.assertEqual([entry[0] for entry in history], list(range(1, 31)))
            for start in (0, 15):
                self.assertEqual(len({entry[1:] for entry in history[start:start + 15]}), 1)

    def test_arrival_still_terminates_on_its_first_second_with_long_decisions(self):
        env = MultiAgentStrategicEnv(scenario=fixed_scenario(first_at_goal=True), n_obstacles=0,
                                     action_frequency=15, episode_time_limit=30)
        self.addCleanup(env.close)
        env.reset(seed=4)
        _, _, terminations, _, infos = env.step({a: np.zeros(2) for a in env.agents})
        self.assertTrue(terminations["A"])
        self.assertEqual(infos["A"]["flight_time"], 1)
        self.assertEqual(infos["B"]["flight_time"], 15)
        self.assertEqual(env.sim_time, 15)
        self.assertEqual(env.agents, ["B", "C"])

    def test_decision_frequency_does_not_change_scoring_for_same_physical_commands(self):
        center = np.array([52., 4.])
        ll = lambda n, e: tuple(fn.nm_to_latlong(center, np.array([n, e])))
        scenario = Scenario(
            center=tuple(center),
            sector=[ll(-100, -100), ll(-100, 100), ll(100, 100), ll(100, -100)],
            agents=[AgentSpec("A", ll(0, -2.75), ll(0, 80), 90, 150),
                    AgentSpec("B", ll(0, 2.75), ll(0, -80), 270, 150)],
        )
        snapshots = []
        for seconds in (5, 15):
            env = MultiAgentStrategicEnv(scenario=scenario, n_obstacles=0,
                                         action_frequency=seconds, episode_time_limit=30)
            try:
                env.reset(seed=4)
                with contextlib.ExitStack() as stack:
                    for agent, heading in (("A", 90), ("B", 270)):
                        stack.enter_context(patch.object(env.navigation_states[agent], "_get_navigation_reference",
                                                         return_value=(heading, 0)))
                    for _ in range(30 // seconds):
                        _, _, _, _, infos = env.step({a: np.array([0, -0.4]) for a in env.agents})
                snapshots.append({a: {k: v for k, v in info.items() if k != "total_reward"}
                                  for a, info in infos.items()})
                self.assertTrue(all(info["intrusion_time"] > 0 for info in infos.values()))
            finally:
                env.close()
        self.assertEqual(snapshots[0], snapshots[1])

    def test_evaluation_restores_timing_and_old_checkpoints_keep_original_defaults(self):
        act = lambda observation: np.zeros(2)
        act.model = SimpleNamespace()
        self.assertEqual(evaluation.policy_reward_kwargs(act, "navigation"), {})
        act.model.competition_decision_seconds = 15
        self.assertEqual(evaluation.policy_reward_kwargs(act, "navigation"), {"decision_seconds": 15})
        env = evaluation.make_env("ma", mdp="navigation", n_agents=2,
                                  **evaluation.policy_reward_kwargs(act, "navigation"))
        self.addCleanup(env.close)
        self.assertEqual(env.action_frequency, 15)
        self.assertEqual(env.sim_dt, 1)
        self.assertEqual(env.episode_time_limit, 3000)
        self.assertEqual(3000 // env.action_frequency, 200)
        with self.assertRaisesRegex(ValueError, "integer divisor"):
            train_zoo.make_env(mdp="navigation-v5", decision_seconds=7)

    def test_search_forecasts_obey_the_real_hold_duration(self):
        env, model = lookahead_fixture.LookaheadPipelineTests.make_model(self, n_agents=2,
                                                                        action_frequency=15)
        env.reset(seed=37)
        config = LookaheadConfig(simulations=1, max_depth=1, rollout_seconds=5)
        planner = LookaheadPolicy(model, env, config)
        scene = planner.snapshot()
        # Put goals far away in the copied scene to isolate the timing check.
        scene = replace(scene, aircraft=tuple(replace(a, goal=(1000., 1000.)) for a in scene.aircraft))
        commands = {a.agent: (a.motion.heading_deg, 150) for a in scene.aircraft}
        predicted = planner.transition(scene, 0, np.array([0, -0.4]), commands)
        self.assertEqual(predicted.state.time - scene.time, 15)
        self.assertEqual(env.sim_time, scene.time)
        self.assertAlmostEqual(predicted.discount, model.gamma)

    def test_cli_forwards_interval_to_each_curriculum_stage(self):
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("timing_" + uuid4().hex)
        with patch("train_zoo.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            train_zoo.main(["train", "16", "--mdp", "navigation-v5", "--curriculum", "2", "10",
                            "--decision-seconds", "15", "--eval-episodes", "0", "--model", str(directory / "model"),
                            "--metrics", str(directory / "metrics.csv"),
                            "--checkpoint-dir", str(directory / "checkpoints")])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--decision-seconds") + 1], "15")


if __name__ == "__main__":
    unittest.main()
