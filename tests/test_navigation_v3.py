"""Full-traffic observations, reproducible profiles and curriculum orchestration."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import bluesky as bs
import numpy as np

from bluesky_zoo.multi_agent_reward_env import MultiAgentRewardEnv
from bluesky_zoo.multi_agent_traffic_env import MultiAgentTrafficEnv
from bluesky_zoo.navigation_reward import make_reward_config, predicted_conflict_risk
from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv
from core.tools import kwikqdrdist
from scripts.evaluate_competition import make_env, policy_reward_kwargs
from test_multi_agent_env import fixed_scenario
from train_zoo import train, train_curriculum


class NavigationV3Tests(unittest.TestCase):
    def test_all_curriculum_sizes_have_identical_spaces_and_correct_masks(self):
        shared_space = None
        for count in (2, 4, 6, 10):
            env = MultiAgentTrafficEnv(n_agents=count)
            try:
                observations, _ = env.reset(seed=70)
                for agent, observation in observations.items():
                    space = env.observation_space(agent)
                    if shared_space is None:
                        shared_space = space
                    self.assertEqual(space, shared_space)
                    self.assertTrue(space.contains(observation))
                    self.assertEqual(observation["intruders"].shape, (9, 7))
                    np.testing.assert_array_equal(observation["intruder_mask"],
                                                  [1] * (count - 1) + [0] * (10 - count))
                    np.testing.assert_array_equal(observation["intruders"][count - 1:],
                                                  np.zeros((10 - count, 7)))
                    self.assertEqual(observation["remaining_time"][0], 1)
                    idx = bs.traf.id2idx(agent)
                    actual_distances = [kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                                   bs.traf.lat[bs.traf.id2idx(other)],
                                                   bs.traf.lon[bs.traf.id2idx(other)])[1] / 100
                                        for other in env.agents if other != agent]
                    np.testing.assert_allclose(sorted(observation["intruders"][:count - 1, 4]),
                                               sorted(np.clip(actual_distances, 0, 3)))
            finally:
                env.close()

    def test_removing_an_aircraft_updates_mask_and_clock_without_history_leak(self):
        env = MultiAgentTrafficEnv(scenario=fixed_scenario(first_at_goal=True), n_obstacles=0)
        try:
            env.reset(seed=4)
            observations, _, terminated, _, _ = env.step({a: np.zeros(2) for a in env.agents})
            self.assertTrue(terminated["A"])
            self.assertEqual(env.agents, ["B", "C"])
            for agent in env.agents:
                self.assertEqual(observations[agent]["intruder_mask"].sum(), 1)
                self.assertAlmostEqual(observations[agent]["remaining_time"][0], 1 - 5 / 3000)
                self.assertTrue(env.observation_space(agent).contains(observations[agent]))
            # Returned masks are snapshots, not shared references to mutable history.
            snapshot = observations["B"]["intruder_mask"].copy()
            env.navigation_states["B"].intruder_mask.fill(0)
            np.testing.assert_array_equal(observations["B"]["intruder_mask"], snapshot)
            observations, _ = env.reset(seed=4)
            self.assertEqual(observations["B"]["intruder_mask"].sum(), 2)
            self.assertEqual(observations["B"]["remaining_time"][0], 1)
        finally:
            env.close()

    def test_timeout_final_observation_has_zero_time_and_reset_has_full_time(self):
        vec = SharedPolicyVecEnv(MultiAgentTrafficEnv(n_agents=2, episode_time_limit=5))
        try:
            vec.seed(50)
            vec.reset()
            observations, _, dones, infos = vec.step(np.zeros((2, 2)))
            self.assertTrue(all(dones))
            np.testing.assert_array_equal(observations["remaining_time"], [[1], [1]])
            for info in infos:
                self.assertTrue(info["TimeLimit.truncated"])
                self.assertEqual(info["terminal_observation"]["remaining_time"][0], 0)
                self.assertEqual(info["terminal_observation"]["intruder_mask"].sum(), 1)
        finally:
            vec.close()

    def test_current_intrusion_is_sorted_before_approaching_traffic(self):
        env = MultiAgentTrafficEnv(n_agents=2)
        try:
            state = env.navigation_states[env.possible_agents[0]]
            distant = np.zeros(7)
            distant[4] = 0.2
            close = np.zeros(7)
            close[4] = 0.04
            with patch.object(state, "_get_intruder_candidates", return_value=[(0, distant), (1004, close)]):
                features = state._get_intruder_features(0)
            self.assertEqual(features[0, 4], 0.04)
            np.testing.assert_array_equal(state.intruder_mask, [1, 1, 0, 0, 0, 0, 0, 0, 0])
        finally:
            env.close()

    def test_new_observation_does_not_change_scoring_for_identical_actions(self):
        snapshots = []
        for cls in (MultiAgentRewardEnv, MultiAgentTrafficEnv):
            env = cls(n_agents=4)
            try:
                env.reset(seed=815)
                for _ in range(3):
                    env.step({a: np.zeros(2) for a in env.agents})
                snapshots.append({a: {k: v for k, v in m.items() if k != "total_reward"}
                                  for a, m in env.metrics.items()})
            finally:
                env.close()
        self.assertEqual(snapshots[0], snapshots[1])

    def test_old_checkpoint_rejected_with_actionable_message(self):
        env = MultiAgentRewardEnv(n_agents=2)
        old = SimpleNamespace(observation_space=env.observation_space(env.possible_agents[0]),
                              action_space=env.action_space(env.possible_agents[0]))
        env.close()
        with patch("train_zoo.PPO.load", return_value=old):
            with self.assertRaisesRegex(ValueError, "V2/SA checkpoints cannot load directly into v3"):
                train(4, mdp="navigation-v3", n_agents=2, n_steps=2, batch_size=2,
                      init_model="old_model", checkpoint_freq=0, eval_episodes=0)

    def test_versioned_warning_defaults_overrides_and_saved_parameters(self):
        v2 = make_reward_config("navigation-v2")
        v3 = make_reward_config("navigation-v3")
        self.assertEqual(v2.gamma, 0.99)
        self.assertEqual(v3.gamma, 0.998)
        self.assertEqual(v3.conflict_horizon_seconds, 180)
        self.assertEqual(v3.conflict_risk_second_cost, 0.05)
        self.assertEqual(predicted_conflict_risk(150, 2, 12, 5, v2), 0)
        self.assertGreater(predicted_conflict_risk(150, 2, 12, 5, v3), 0)
        overridden = make_reward_config("navigation-v3", gamma=0.997,
                                        conflict_horizon_seconds=240, conflict_risk_second_cost=0.08)
        saved = overridden.__dict__.copy()
        act = lambda obs: np.zeros(2)
        act.model = SimpleNamespace(gamma=0.997, competition_reward_config=saved)
        kwargs = policy_reward_kwargs(act, "navigation-v3")
        self.assertEqual(kwargs["reward_config"], saved)
        env = make_env("ma", n_agents=2, mdp="navigation-v3", **kwargs)
        try:
            self.assertEqual(env.reward_config, overridden)
        finally:
            env.close()
        with self.assertRaisesRegex(ValueError, "require navigation-v2 or navigation-v3"):
            make_reward_config("base", conflict_risk_second_cost=0.1)
        with self.assertRaises(ValueError):
            MultiAgentTrafficEnv(n_agents=11)


class CurriculumTests(unittest.TestCase):
    def args(self):
        base = Path(__file__).resolve().parents[1] / ".cache" / "tests" / f"curriculum_{uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        self.addCleanup((base / "model_curriculum" / "curriculum.json").unlink, missing_ok=True)
        return SimpleNamespace(
            curriculum=[2, 4, 6, 10], mdp="navigation-v3", n_agents=10,
            timesteps=103, model=str(base / "model"), metrics=str(base / "metrics.csv"),
            checkpoint_dir=str(base / "checkpoints"), init_model=None, seed=10,
            n_steps=2, batch_size=2, learning_rate=0.0003, n_epochs=1,
            device="cpu", torch_threads=1, checkpoint_freq=0, gamma=None,
            conflict_horizon=240, conflict_risk_cost=0.07, conflict_buffer_nm=None,
            sector_buffer_cost=0.01, sector_buffer_nm=5,
            heading_deadband_deg=1, speed_deadband_kt=1,
            eval_episodes=20,
        )

    def test_curriculum_shares_weights_forwards_options_and_evaluates_only_last_stage(self):
        args = self.args()
        with patch("train_zoo.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            train_curriculum(args)
        commands = [call.args[0] for call in run.call_args_list]
        value = lambda command, flag: command[command.index(flag) + 1]
        self.assertEqual([int(value(c, "--n-agents")) for c in commands], [2, 4, 6, 10])
        self.assertEqual([int(value(c, "--eval-episodes")) for c in commands], [0, 0, 0, 20])
        self.assertEqual(sum(int(c[c.index("train") + 1]) for c in commands), 103)
        self.assertNotIn("--init-model", commands[0])
        for i, command in enumerate(commands):
            self.assertEqual(value(command, "--conflict-horizon"), "240")
            self.assertEqual(value(command, "--conflict-risk-cost"), "0.07")
            self.assertEqual(value(command, "--sector-buffer-cost"), "0.01")
            self.assertEqual(value(command, "--sector-buffer-nm"), "5")
            self.assertEqual(value(command, "--heading-deadband-deg"), "1")
            self.assertEqual(value(command, "--speed-deadband-kt"), "1")
            if i:
                self.assertEqual(value(command, "--init-model"), value(commands[i - 1], "--model") + ".zip")
        self.assertEqual(value(commands[-1], "--model"), args.model)

    def test_invalid_curriculum_does_not_launch_and_failures_stop_later_stages(self):
        args = self.args()
        args.curriculum = [4, 2, 10]
        with patch("train_zoo.subprocess.run") as run:
            with self.assertRaises(ValueError):
                train_curriculum(args)
            run.assert_not_called()
        args.curriculum = [2, 4, 10]
        with patch("train_zoo.subprocess.run", side_effect=subprocess.CalledProcessError(1, "stage")) as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                train_curriculum(args)
            self.assertEqual(run.call_count, 1)
        manifest_path = Path(args.model).parent / "model_curriculum" / "curriculum.json"
        self.assertEqual(json.loads(manifest_path.read_text())["stages"][0]["status"], "failed")

    def test_v4_curriculum_forwards_passage_profile_and_training_mixture(self):
        args = self.args()
        args.mdp = "navigation-v4"
        args.obstacle_buffer_nm = 0.25
        args.obstacle_release_nm = 0.75
        args.max_heading_offset_deg = 150
        args.lookahead_nm = 12
        args.passage_training_probability = 0.25
        with patch("train_zoo.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            train_curriculum(args)
        for call in run.call_args_list:
            command = call.args[0]
            for flag, expected in (("--obstacle-buffer-nm", "0.25"), ("--obstacle-release-nm", "0.75"),
                                   ("--max-heading-offset-deg", "150"), ("--lookahead-nm", "12"),
                                   ("--passage-training-probability", "0.25")):
                self.assertEqual(command[command.index(flag) + 1], expected)


if __name__ == "__main__":
    unittest.main()
