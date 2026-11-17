"""Evaluation/reel regressions without starting BlueSky's global simulator."""

import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import evaluate_competition as evaluation
from scripts import record_competition_gifs as recorder


class EndingParallelEnv:
    """One aircraft arrives early; the remaining one times out later."""

    possible_agents = ["first", "second"]

    def __init__(self):
        self.seeds = []
        self.closed = False
        self.actions = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.agents = list(self.possible_agents)
        self.steps = 0
        return {a: {"value": np.zeros(1)} for a in self.agents}, {}

    def step(self, actions):
        self.actions.append(set(actions))
        self.steps += 1
        if self.steps == 1:
            self.agents = ["second"]
            return (
                {a: {"value": np.zeros(1)} for a in self.possible_agents},
                {}, {"first": True, "second": False},
                {"first": False, "second": False},
                {"first": {"waypoint_reached": 1}, "second": {"waypoint_reached": 0}},
            )
        self.agents = []
        return ({"second": {"value": np.zeros(1)}}, {}, {"second": False},
                {"second": True}, {"second": {"waypoint_reached": 0}})

    def close(self):
        self.closed = True


class FullBudgetEnv:
    episode_time_limit = 3000.0
    action_frequency = 5
    sim_dt = 1

    def __init__(self):
        self.closed = False

    def reset(self, seed=None):
        self.steps = 0
        self.agents = ["aircraft"]
        return {"aircraft": {}}, {}

    def step(self, actions):
        self.steps += 1
        if self.steps == 600:
            self.agents = []
        return {"aircraft": {}}, {}, {}, {}, {}

    def render(self):
        return self.steps

    def close(self):
        self.closed = True


class CompetitionPipelineTests(unittest.TestCase):
    def test_v2_uses_saved_reward_config_and_rejects_discount_mismatch(self):
        act = lambda obs: np.zeros(2)
        config = {"gamma": 0.998, "goal_bonus": 31}
        act.model = SimpleNamespace(gamma=0.998, competition_reward_config=config)
        self.assertEqual(evaluation.policy_reward_kwargs(act, "navigation-v2"),
                         {"reward_config": config})
        self.assertEqual(evaluation.policy_reward_kwargs(act, "navigation"), {})
        act.model.gamma = 0.99
        with self.assertRaisesRegex(ValueError, "differs from PPO gamma"):
            evaluation.policy_reward_kwargs(act, "navigation-v2")

    def test_v2_is_ma_only_and_labeled_checkpoints_cannot_silently_use_old_reward(self):
        with self.assertRaisesRegex(ValueError, "multi-agent reward experiment"):
            evaluation.make_env("sa", mdp="navigation-v2")
        act = lambda obs: np.zeros(2)
        act.model = SimpleNamespace(competition_mdp="navigation-v2")
        with self.assertRaisesRegex(ValueError, "trained with --mdp navigation-v2"):
            evaluation.validate_policy_spaces(act, None, "ma", "navigation")

    def test_metrics_include_each_arrival_and_timeout_and_seed_only_once(self):
        env = EndingParallelEnv()
        with patch.object(evaluation, "make_env", return_value=env) as make_env:
            records = evaluation.run_multi_agent(2, lambda obs: np.zeros(2), 2, mdp="navigation")
        self.assertEqual([r["waypoint_reached"] for r in records], [1, 0, 1, 0])
        self.assertEqual(env.seeds, [42, None])
        self.assertEqual(env.actions, [{"first", "second"}, {"second"}] * 2)
        self.assertTrue(env.closed)
        make_env.assert_called_once_with("ma", n_agents=2, mdp="navigation")

    def test_evaluation_closes_environment_when_policy_fails(self):
        env = EndingParallelEnv()
        act = Mock(side_effect=RuntimeError("policy failed"))
        # A plain callable has no attached PPO model.
        act.model = None
        with patch.object(evaluation, "make_env", return_value=env):
            with self.assertRaisesRegex(RuntimeError, "policy failed"):
                evaluation.run_multi_agent(1, act, 2)
        self.assertTrue(env.closed)

    def test_ppo_prediction_is_deterministic_and_uses_dict_directly(self):
        model = Mock()
        model.predict.return_value = (np.array([0.2, -0.1]), None)
        fake_sb3 = SimpleNamespace(PPO=SimpleNamespace(load=Mock(return_value=model)))
        with patch.dict(sys.modules, {"stable_baselines3": fake_sb3}):
            act = evaluation.load_policy("ma", "checkpoint.zip", mdp="navigation")
        observation = {"goal_direction": np.zeros(2)}
        np.testing.assert_array_equal(act(observation), [0.2, -0.1])
        model.predict.assert_called_once_with(observation, deterministic=True)

    def test_checkpoint_space_mismatch_explains_mdp(self):
        act = lambda obs: np.zeros(2)
        act.model = SimpleNamespace(observation_space="wrong", action_space="action")
        env = SimpleNamespace(observation_space="expected", action_space="action")
        with self.assertRaisesRegex(ValueError, "incompatible with --env sa --mdp navigation"):
            evaluation.validate_policy_spaces(act, env, "sa", "navigation")

    def test_shared_policy_batches_only_live_aircraft_and_keeps_agent_order(self):
        model = Mock()
        model.predict.return_value = (np.array([[0.2, 0.3], [0.4, 0.5]]), None)
        act = lambda obs: self.fail("loaded PPO should use batched inference")
        act.model = model
        observations = {
            "departed": {"goal": np.array([99.0, 99.0])},
            "second": {"goal": np.array([2.0, 3.0])},
            "first": {"goal": np.array([0.0, 1.0])},
        }
        actions = evaluation.actions_for_agents(act, observations, ["first", "second"])
        model.predict.assert_called_once()
        batch = model.predict.call_args.args[0]
        np.testing.assert_array_equal(batch["goal"], [[0.0, 1.0], [2.0, 3.0]])
        self.assertTrue(model.predict.call_args.kwargs["deterministic"])
        self.assertEqual(list(actions), ["first", "second"])
        np.testing.assert_array_equal(actions["second"], [0.4, 0.5])

    def test_nonstandard_agent_count_is_explicitly_nonofficial(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(evaluation, "run_multi_agent", return_value=[dict.fromkeys(evaluation.METRIC_KEYS, 0)] * 2000), \
                patch.object(evaluation, "summarize"), patch.object(evaluation, "write_csv"), \
                patch.object(evaluation, "write_report"):
            evaluation.evaluate("ma", 1000, 2, None, None, mdp="navigation")
        self.assertIn("NONOFFICIAL CHECK", output.getvalue())
        self.assertIn("10 agents", output.getvalue())

    def test_navigation_reel_runs_past_old_400_step_cap_to_full_budget(self):
        env = FullBudgetEnv()
        with patch.object(recorder, "make_env", return_value=env):
            frames = recorder.rollout_multi_agent(7, mdp="navigation")
        self.assertEqual(len(frames), 601)
        self.assertEqual(frames[-1], 600)
        self.assertTrue(env.closed)

    def test_reel_respects_explicit_step_cap(self):
        env = FullBudgetEnv()
        with patch.object(recorder, "make_env", return_value=env):
            frames = recorder.rollout_multi_agent(7, max_steps=3, mdp="navigation")
        self.assertEqual(frames, [0, 1, 2, 3])
        self.assertTrue(env.closed)

    def test_reel_closes_environment_when_reset_fails(self):
        env = FullBudgetEnv()
        env.reset = Mock(side_effect=RuntimeError("reset failed"))
        with patch.object(recorder, "make_env", return_value=env):
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                recorder.rollout_multi_agent(7, mdp="navigation")
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
