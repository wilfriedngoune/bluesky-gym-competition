"""Learned v5 inference preserves the live simulator and evaluation protocol."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import bluesky as bs
import numpy as np
import torch
from stable_baselines3 import PPO

from bluesky_gym.envs.common.command_filter import CommandDeadbandConfig
from bluesky_zoo.lookahead import LookaheadConfig
from bluesky_zoo.lookahead_policy import LookaheadPolicy
from bluesky_zoo.passage_navigation import PassageNavigationConfig
from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv
from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv, ZoneConfig
from core.scenario import ScenarioGenerator
from scripts import evaluate_competition as evaluation
from test_competition_pipeline import EndingParallelEnv
from test_multi_agent_env import fixed_scenario


class LookaheadPipelineTests(unittest.TestCase):
    def make_model(self, **env_kwargs):
        """Use real PPO inference after one small update, with a fixed seed."""
        torch.set_num_threads(1)
        env = MultiAgentStrategicEnv(**env_kwargs)
        vec = SharedPolicyVecEnv(env)
        self.addCleanup(vec.close)
        model = PPO(
            "MultiInputPolicy", vec, n_steps=2, batch_size=2, n_epochs=1,
            gamma=0.998, seed=17, device="cpu", verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])),
        )
        model.learn(total_timesteps=2 * vec.num_envs)
        model.competition_mdp = "navigation-v5"
        model.competition_reward_config = asdict(env.reward_config)
        model.competition_action_config = asdict(env.action_config)
        model.competition_passage_config = asdict(env.passage_config)
        model.competition_zone_config = asdict(env.zone_config)
        model.competition_lookahead_config = asdict(LookaheadConfig(
            simulations=4, max_depth=2, candidate_count=3, rollout_seconds=2,
            discount=model.gamma,
        ))
        return env, model

    def live_state(self, env):
        history_fields = (
            "last_action", "previous_action", "previous_goal_distance",
            "previous_escape_clearance", "was_inside_restricted",
            "active_buffer_obstacle", "_has_target_command", "last_reference_heading",
            "last_target_heading", "last_target_speed", "last_navigation_mode",
            "last_command_obstacle", "intruder_mask", "zone_schedule",
            "reward_component_totals", "last_reward_components",
        )
        return dict(
            traffic={name: np.array(getattr(bs.traf, name), copy=True)
                     for name in ("lat", "lon", "hdg", "cas", "tas")},
            agents=list(env.agents), time=env.sim_time,
            metrics=deepcopy(env.metrics), baseline=deepcopy(env._reward_baseline),
            random=deepcopy(env._np_random.bit_generator.state),
            histories={agent: {name: deepcopy(getattr(state, name)) for name in history_fields}
                       for agent, state in env.navigation_states.items()},
        )

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        else:
            self.assertEqual(actual, expected)

    def test_safety_filter_preserves_live_state_and_uses_fixed_horizon(self):
        from dataclasses import replace
        env, model = self.make_model(n_agents=2)
        observations, _ = env.reset(seed=37)
        cfg = LookaheadConfig(**model.competition_lookahead_config)
        planner = LookaheadPolicy(model, env, replace(cfg, safety_filter=True, safety_horizon_seconds=20))
        before = self.live_state(env)
        with patch.object(bs.sim, "step", side_effect=AssertionError("live simulation")), \
                patch.object(bs.stack, "stack", side_effect=AssertionError("live command")), \
                patch.object(env, "_get_obs", side_effect=AssertionError("live observation")):
            actions = planner.actions(observations, env.agents)
        self.assert_nested_equal(before, self.live_state(env))
        for agent in env.agents:
            diagnostics = planner.last_safety_diagnostics[agent]
            self.assertEqual(diagnostics["threshold_nm"], 8.)
            for forecast in diagnostics["candidates"]:
                self.assertEqual(forecast.duration_seconds, 20.)
            self.assertEqual(planner.last_diagnostics[agent]["simulations"], cfg.simulations)
        reordered = planner.actions(observations, reversed(env.agents))
        for agent in env.agents:
            np.testing.assert_array_equal(actions[agent], reordered[agent])

    def test_two_pass_uses_complete_proposals_and_preserves_live_state(self):
        env, model = self.make_model(n_agents=3)
        observations, _ = env.reset(seed=37)
        config = LookaheadConfig(simulations=5, max_depth=2, candidate_count=3,
                                 rollout_seconds=2, coordination_passes=2)
        planner = LookaheadPolicy(model, env, config)
        before = self.live_state(env)
        observer = Mock()
        observer.wants.return_value = True
        observed_passes = []
        observer.record_search.side_effect = lambda current, *args: observed_passes.append(current.active_pass)
        planner.observer = observer
        actions = planner.actions(observations, env.agents)
        self.assertEqual(observed_passes, [1, 1, 1])
        records = deepcopy(planner.last_pass_diagnostics)
        self.assert_nested_equal(before, self.live_state(env))
        self.assertEqual(len(records), 2)
        scene = planner.snapshot()
        expected = {a.agent: planner._command(a, records[0][a.agent]["action"])[:2]
                    for a in scene.aircraft}
        for agent in env.agents:
            self.assertEqual(records[0][agent]["simulations"], 3)
            self.assertEqual(records[1][agent]["simulations"], 2)
            self.assertEqual(records[1][agent]["assumed_commands"], expected)
            self.assertEqual(planner.last_diagnostics[agent]["simulations"], 5)
            np.testing.assert_array_equal(actions[agent], records[1][agent]["action"])
        reordered = planner.actions(observations, reversed(env.agents))
        for agent in env.agents:
            np.testing.assert_array_equal(actions[agent], reordered[agent])

    def test_coordination_configuration_defaults_and_budget_validation(self):
        self.assertEqual(LookaheadConfig().coordination_passes, 1)
        for passes in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                LookaheadConfig(coordination_passes=passes)
        with self.assertRaises(ValueError):
            LookaheadConfig(simulations=1, coordination_passes=2)

    def test_real_policy_search_returns_bounded_actions_without_mutating_live_state(self):
        env, model = self.make_model(n_agents=2)
        observations, _ = env.reset(seed=37)
        planner = LookaheadPolicy(model, env, LookaheadConfig(**model.competition_lookahead_config))
        before = self.live_state(env)
        # An accidental observation call can change live obstacle hysteresis;
        # forbidding all live hooks catches that even when the resulting state
        # happens to equal its original value on this scenario.
        with patch.object(env, "_get_obs", side_effect=AssertionError("live observation hook")), \
                patch.object(bs.sim, "step", side_effect=AssertionError("live simulation step")), \
                patch.object(bs.stack, "stack", side_effect=AssertionError("live command")):
            actions = planner.actions(observations, env.agents)
            reordered = planner.actions(observations, reversed(env.agents))
        self.assertEqual(set(actions), set(env.agents))
        self.assert_nested_equal(before, self.live_state(env))
        for agent, action in actions.items():
            self.assertEqual(action.shape, (2,))
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(np.all(np.abs(action) <= 1))
            np.testing.assert_array_equal(action, reordered[agent])
            self.assertEqual(planner.last_diagnostics[agent]["simulations"], 4)
            self.assertLessEqual(planner.last_diagnostics[agent]["expanded_nodes"], 5)

    def test_forecast_inputs_have_matching_features_and_search_uses_exact_root_observation(self):
        env, model = self.make_model(n_agents=2)
        observations, _ = env.reset(seed=37)
        planner = LookaheadPolicy(model, env, LookaheadConfig(**model.competition_lookahead_config))
        scene = planner.snapshot()
        for index, aircraft in enumerate(scene.aircraft):
            forecast = planner.observation(scene, index)
            actual = observations[aircraft.agent]
            self.assertEqual(set(forecast), set(actual))
            self.assertTrue(env.observation_space(aircraft.agent).contains(forecast))
            for key in actual:
                # Forecasts use a local planar metric. Live features use the
                # existing geodesic helper, so normalized geometry is close
                # rather than numerically identical.
                np.testing.assert_allclose(forecast[key], actual[key], atol=0.01, rtol=0.01,
                                           err_msg=f"forecast input: {aircraft.agent}/{key}")
        with patch.object(planner, "_expansion", wraps=planner._expansion) as expansion:
            planner.actions(observations, env.agents)
        root_calls = [call for call in expansion.call_args_list if call.kwargs.get("root_observation") is not None]
        self.assertEqual(len(root_calls), len(env.agents))
        for call in root_calls:
            scene, index = call.args
            self.assertIs(call.kwargs["root_observation"], observations[scene.aircraft[index].agent])

    def test_planning_remains_correct_after_early_aircraft_deletion_changes_traffic_indices(self):
        scenario = fixed_scenario(first_at_goal=True)
        # Keep the official five-obstacle input shape: SB3's policy.predict
        # cannot infer a batch dimension from an empty (N, 0, 7) observation.
        scenario.obstacles = ScenarioGenerator(n_agents=3).generate(np.random.default_rng(51)).obstacles
        env, model = self.make_model(scenario=scenario)
        observations, _ = env.reset(seed=4)
        observations, _, terminations, _, _ = env.step({a: np.zeros(2) for a in env.agents})
        self.assertTrue(terminations["A"])
        self.assertEqual(env.agents, ["B", "C"])
        self.assertEqual(bs.traf.id2idx("B"), 0)
        planner = LookaheadPolicy(model, env, LookaheadConfig(simulations=1, max_depth=1,
                                                            candidate_count=3, rollout_seconds=1))
        before = self.live_state(env)
        actions = planner.actions(observations, env.agents)
        self.assertEqual(set(actions), {"B", "C"})
        self.assert_nested_equal(before, self.live_state(env))
        for action in actions.values():
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(np.all(np.abs(action) <= 1))

    def test_checkpoint_restores_zone_passage_reward_and_search_profiles(self):
        zone = ZoneConfig(grace_seconds=45, outside_second_cost=0.005)
        passage = PassageNavigationConfig(obstacle_buffer_nm=0.25, obstacle_release_nm=0.75,
                                          lookahead_nm=12)
        command = CommandDeadbandConfig(heading_deadband_deg=1, speed_deadband_kt=1)
        env, model = self.make_model(n_agents=2, zone_config=zone, passage_config=passage,
                                     action_config=command)
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / f"lookahead_{uuid4().hex}.zip"
        self.addCleanup(checkpoint.unlink, missing_ok=True)
        model.save(checkpoint)
        act = evaluation.load_policy("ma", checkpoint, mdp="navigation-v5")
        self.assertEqual(asdict(act.lookahead_config), model.competition_lookahead_config)
        kwargs = evaluation.policy_reward_kwargs(act, "navigation-v5")
        self.assertEqual(kwargs["zone_config"], asdict(zone))
        self.assertEqual(kwargs["passage_config"], asdict(passage))
        self.assertEqual(kwargs["action_config"], asdict(command))
        self.assertEqual(kwargs["reward_config"], asdict(env.reward_config))
        observations, _ = env.reset(seed=37)
        actions = evaluation.actions_for_agents(act, observations, env.agents, env=env)
        self.assertEqual(set(actions), set(env.agents))
        self.assertIs(act.lookahead_planner.env, env)
        self.assertEqual(act.lookahead_planner.config, act.lookahead_config)
        with self.assertRaisesRegex(ValueError, "RL checkpoint"):
            evaluation.load_policy("ma", mdp="navigation-v5")
        with self.assertRaisesRegex(ValueError, "discount must match"):
            LookaheadPolicy(model, env, LookaheadConfig(discount=0.99))
        model.competition_mdp = "navigation-v4"
        with self.assertRaisesRegex(ValueError, "learned navigation-v5 checkpoint"):
            LookaheadPolicy(model, env, act.lookahead_config)

    def test_disabling_search_uses_one_deterministic_batched_actor_prediction(self):
        model = Mock()
        model.predict.return_value = (np.array([[0.2, -0.3], [-0.4, 0.5]]), None)
        act = lambda obs: self.fail("shared policy should batch predictions")
        act.model = model
        act.lookahead_config = LookaheadConfig(simulations=1)
        act.disable_lookahead = True
        observations = {"first": {"goal": np.array([1.0])}, "second": {"goal": np.array([2.0])}}
        with patch("bluesky_zoo.lookahead_policy.LookaheadPolicy") as planner:
            actions = evaluation.actions_for_agents(act, observations, ["second", "first"])
        planner.assert_not_called()
        model.predict.assert_called_once()
        np.testing.assert_array_equal(model.predict.call_args.args[0]["goal"], [[2], [1]])
        self.assertTrue(model.predict.call_args.kwargs["deterministic"])
        np.testing.assert_array_equal(actions["first"], [-0.4, 0.5])

    def test_v5_evaluation_keeps_seed_stream_and_records_each_early_arrival_once(self):
        env = EndingParallelEnv()
        act = lambda observation: np.zeros(2)
        act.model = SimpleNamespace(observation_space=SimpleNamespace(spaces={}), gamma=0.998)
        act.lookahead_config = LookaheadConfig(simulations=1, max_depth=1, candidate_count=3,
                                              rollout_seconds=1)
        planned_agents = []

        class RecordingPlanner:
            def __init__(self, model, target, config):
                self.env, self.config = target, config

            def actions(self, observations, agents):
                planned_agents.append(list(agents))
                return {agent: np.zeros(2) for agent in agents}

        with patch.object(evaluation, "make_env", return_value=env), \
                patch.object(evaluation, "validate_policy_spaces"), \
                patch("bluesky_zoo.lookahead_policy.LookaheadPolicy", RecordingPlanner):
            records = evaluation.run_multi_agent(2, act, 2, mdp="navigation-v5")
        self.assertEqual(env.seeds, [42, None])
        self.assertEqual(planned_agents, [["first", "second"], ["second"]] * 2)
        self.assertEqual([row["waypoint_reached"] for row in records], [1, 0, 1, 0])
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
