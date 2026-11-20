"""SA learned search forecasts visible traffic without touching the scored world."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import bluesky as bs
import numpy as np
import torch
from stable_baselines3 import PPO

from bluesky_gym.envs.common.command_filter import CommandDeadbandConfig
from bluesky_zoo.lookahead import AircraftState, LookaheadConfig
from bluesky_zoo.single_agent_lookahead import SingleAgentLookaheadPolicy
from bluesky_zoo.transfer_navigation import MDP, SingleAgentTransferEnv, TransferConfig


class SingleAgentLookaheadTests(unittest.TestCase):
    def make_planner(self, config=None, **env_kwargs):
        torch.set_num_threads(1)
        env = SingleAgentTransferEnv(**env_kwargs)
        self.addCleanup(env.close)
        model = PPO("MultiInputPolicy", env, n_steps=2, batch_size=2, n_epochs=1,
                    gamma=.998, seed=17, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])))
        model.competition_mdp = MDP
        obs, _ = env.reset(seed=37)
        config = config or LookaheadConfig(simulations=4, max_depth=2, candidate_count=3)
        return env, model, SingleAgentLookaheadPolicy(model, env, config), obs

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def live_state(self, env):
        state = env.navigation_state
        fields = ("last_action", "previous_action", "previous_goal_distance",
                  "previous_escape_clearance", "was_inside_restricted",
                  "active_buffer_obstacle", "_has_target_command", "last_reference_heading",
                  "last_target_heading", "last_target_speed", "last_navigation_mode",
                  "last_command_obstacle", "intruder_mask", "zone_schedule",
                  "reward_component_totals", "last_reward_components",
                  "progress_anchor_time", "progress_anchor_distance", "recent_progress_rate",
                  "stalled_seconds", "avoidance_obstacle", "avoidance_side",
                  "avoidance_started", "avoidance_switched")
        return dict(traffic={name: np.array(getattr(bs.traf, name), copy=True)
                             for name in ("lat", "lon", "hdg", "cas", "tas")},
                    ids=list(bs.traf.id), metrics=deepcopy(env.metrics), time=env.sim_time,
                    baseline=deepcopy(getattr(env, "_reward_baseline", None)),
                    random=deepcopy(env.np_random.bit_generator.state),
                    numpy_random=deepcopy(np.random.get_state()), torch_random=torch.get_rng_state().clone(),
                    history={name: deepcopy(getattr(state, name)) for name in fields})

    def test_reconstructed_features_match_live_contract_and_root_is_exact(self):
        env, model, planner, observation = self.make_planner()
        scene = planner.snapshot()
        forecast = planner.observation(scene, 0)
        self.assertEqual(set(observation), set(forecast))
        self.assertTrue(env.observation_space.contains(forecast))
        self.assertEqual(forecast["intruders"].shape, (10, 7))
        self.assertEqual(forecast["intruder_mask"].sum(), 10)
        for key in observation:
            np.testing.assert_allclose(forecast[key], observation[key], atol=.01, rtol=.01,
                                       err_msg=f"reconstructed feature {key}")
        with patch.object(planner, "_expansion", wraps=planner._expansion) as expansion:
            planner.action(observation)
        roots = [call for call in expansion.call_args_list
                 if call.kwargs.get("root_observation") is not None]
        self.assertEqual(len(roots), 1)
        self.assertIs(roots[0].kwargs["root_observation"], observation)

    def test_planning_is_deterministic_and_leaves_world_rng_and_memory_untouched(self):
        env, _, planner, observation = self.make_planner()
        before = self.live_state(env)
        with ExitStack() as stack:
            for target, method in ((env, "_get_obs"), (env, "_get_action"), (env, "_get_reward"),
                                   (env.navigation_state, "_get_obs"),
                                   (env.navigation_state, "_get_action"),
                                   (env.navigation_state, "_get_reward"),
                                   (bs.sim, "step"), (bs.stack, "stack")):
                stack.enter_context(patch.object(target, method,
                    side_effect=AssertionError("Search called a live simulator/MDP hook")))
            first = planner.action(observation)
            second = planner.action(observation)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2,))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.all(np.abs(first) <= 1))
        self.assert_nested_equal(before, self.live_state(env))

    def test_scripted_routes_are_unread_and_forecasts_keep_all_scripted_aircraft(self):
        env, _, planner, _ = self.make_planner()

        class HiddenRoutes:
            def __iter__(self):
                raise AssertionError("Search read hidden future intruder routes")

        with patch.object(env.scenario, "intruder_routes", HiddenRoutes()):
            scene = planner.snapshot()
            own = replace(scene.aircraft[0], goal=(1000., 1000.))
            scene = replace(scene, aircraft=(own, *scene.aircraft[1:]))
            result = planner.transition(scene, 0, np.zeros(2))
        self.assertEqual(len(result.state.aircraft), 11)
        duration = result.state.time - scene.time
        self.assertEqual(duration, 15.)
        for previous, following in zip(scene.aircraft[1:], result.state.aircraft[1:]):
            self.assertEqual(previous.goal, ())
            self.assertFalse(following.arrived)
            motion = previous.motion
            angle = math.radians(motion.heading_deg)
            expected = (motion.north_nm + duration * motion.speed_kt / 3600. * math.cos(angle),
                        motion.east_nm + duration * motion.speed_kt / 3600. * math.sin(angle))
            np.testing.assert_allclose((following.motion.north_nm, following.motion.east_nm), expected,
                                       atol=1e-10)
            self.assertEqual(following.motion.heading_deg, motion.heading_deg)
            self.assertEqual(following.motion.speed_kt, motion.speed_kt)

    def test_progress_memory_is_branch_local_and_same_clock_is_idempotent(self):
        _, _, planner, _ = self.make_planner()
        own = planner.snapshot().aircraft[0]
        parent = replace(own, goal=(own.motion.north_nm + 100., own.motion.east_nm),
                         memory=replace(own.memory, progress_anchor_time=0.,
                                        progress_anchor_distance=100., stalled_seconds=0.))
        child = planner._refresh_memory(parent, 10.)
        self.assertEqual(child.memory.stalled_seconds, 10.)
        self.assertEqual(parent.memory.stalled_seconds, 0.)
        self.assertEqual(planner._refresh_memory(child, 10.), child)
        # A sibling making progress clears its own stall state only.
        sibling = replace(parent, motion=replace(parent.motion, north_nm=parent.motion.north_nm + 1.))
        advanced = planner._refresh_memory(sibling, 10.)
        self.assertEqual(advanced.memory.stalled_seconds, 0.)
        self.assertAlmostEqual(advanced.memory.recent_progress_rate, .1)
        self.assertEqual(child.memory.stalled_seconds, 10.)

    def test_snapshot_translates_geodesic_anchor_before_planar_progress_forecasts(self):
        env, model, planner, _ = self.make_planner()
        offsets = []
        for index in range(20):
            observation, _ = env.reset(seed=42 if index == 0 else None)
            before = self.live_state(env)
            live = env.navigation_state
            scene = planner.snapshot()
            own = scene.aircraft[0]
            planar_distance = float(np.linalg.norm(
                np.asarray(own.goal) - np.array([own.motion.north_nm, own.motion.east_nm])))
            offsets.append(abs(live.progress_anchor_distance - planar_distance))
            self.assertEqual(own.memory.progress_anchor_distance, planar_distance)
            for field in own.memory.__dataclass_fields__:
                if field != "progress_anchor_distance":
                    self.assertEqual(getattr(own.memory, field), getattr(live, field))
            # A stationary copied aircraft must not acquire fictitious progress
            # from subtracting a geodesic anchor and a planar endpoint distance.
            refreshed = planner._refresh_memory(own, scene.time + 5.)
            self.assertEqual(refreshed.memory.recent_progress_rate, 0.)
            self.assertEqual(refreshed.memory.stalled_seconds, live.stalled_seconds + 5.)
            # Actor/critic still receive the exact live root observation.
            with patch.object(model.policy, "obs_to_tensor", wraps=model.policy.obs_to_tensor) as tensor:
                planner._expansion(scene, 0, root_observation=observation)
            self.assertIs(tensor.call_args.args[0], observation)
            self.assert_nested_equal(before, self.live_state(env))
        self.assertGreater(max(offsets), .01, "Regression cases must expose the distance-metric mismatch")

    def test_obstacle_side_switch_is_retained_once_per_branch_and_inside_uses_escape(self):
        _, _, planner, _ = self.make_planner()
        own = planner.snapshot().aircraft[0]
        own = replace(own, motion=AircraftState(.2, 0., 0., 150.), goal=(10., 0.), active_obstacle=0,
                      memory=replace(own.memory, avoidance_obstacle=0, avoidance_side=-1,
                                     avoidance_started=0., avoidance_switched=False, stalled_seconds=150.))

        def areas(position, index):
            return (.2, False, np.array([0., 0.])) if index == 1 else (100., index == 0, np.array([0., 0.]))

        with patch.object(planner, "_area_state", side_effect=areas):
            switched, _, mode = planner._navigation_reference(own, 150.)
            again, _, _ = planner._navigation_reference(switched, 150.)
        self.assertEqual(mode, 1)
        self.assertEqual(switched.memory.avoidance_side, 1)
        self.assertTrue(switched.memory.avoidance_switched)
        self.assertEqual(again, switched)
        self.assertEqual(own.memory.avoidance_side, -1)

        def occupied(position, index):
            return (.2, True, np.array([0., 0.])) if index == 1 else (100., index == 0, np.array([0., 0.]))

        with patch.object(planner, "_area_state", side_effect=occupied):
            inside, heading, mode = planner._navigation_reference(own, 150.)
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(abs(heading), 180.)
        self.assertEqual(inside.memory.avoidance_side, -1)

    def test_saved_speed_range_and_command_deadbands_are_used(self):
        cfg = TransferConfig(min_target_speed_kt=135., max_target_speed_kt=205.)
        env, _, planner, observation = self.make_planner(transfer_config=cfg,
            action_config=CommandDeadbandConfig(heading_deadband_deg=2., speed_deadband_kt=2.))
        own = planner.snapshot().aircraft[0]
        heading, speed, _ = planner._command(own, np.array([0., 0.]))
        self.assertEqual(speed, 170.)
        held = replace(own, has_command=True, target_heading=heading + 1., target_speed=171.)
        kept_heading, kept_speed, _ = planner._command(held, np.zeros(2))
        self.assertAlmostEqual(kept_heading, held.target_heading)
        self.assertEqual(kept_speed, 171.)
        self.assertEqual(planner._command(own, np.array([0., -1.]))[1], 135.)
        self.assertEqual(planner._command(own, np.array([0., 1.]))[1], 205.)
        with patch.object(bs.stack, "stack"):
            env._get_action(env.agent, np.array([.2, -.5]))
        state = env.navigation_state
        forecast_heading, forecast_speed, _ = planner._command(own, np.array([.2, -.5]))
        self.assertAlmostEqual((forecast_heading - state.last_target_heading + 180.) % 360. - 180.,
                               0., delta=1.)
        self.assertEqual(forecast_speed, state.last_target_speed)

    def test_time_limit_clips_forecast_and_keeps_bootstrap_but_goal_terminates(self):
        _, _, planner, _ = self.make_planner()
        scene = planner.snapshot()
        own = replace(scene.aircraft[0], goal=(1000., 1000.))
        near_limit = replace(scene, aircraft=(own, *scene.aircraft[1:]), time=2998.)
        frames = []
        timeout = planner.transition(near_limit, 0, np.zeros(2), sample_observer=frames.append)
        self.assertEqual(timeout.state.time, 3000.)
        self.assertFalse(timeout.terminal)
        self.assertEqual(len(frames), 2)
        self.assertGreater(timeout.discount, 0.)
        self.assertLessEqual(timeout.discount, 1.)
        at_goal = replace(own, goal=(own.motion.north_nm, own.motion.east_nm))
        goal_scene = replace(scene, aircraft=(at_goal, *scene.aircraft[1:]))
        reached = planner.transition(goal_scene, 0, np.zeros(2))
        self.assertTrue(reached.terminal)
        self.assertTrue(reached.state.aircraft[0].arrived)
        self.assertLessEqual(reached.state.time - scene.time, 5.)
        self.assertTrue(all(not aircraft.arrived for aircraft in reached.state.aircraft[1:]))

    def test_macro_reward_matches_three_discounted_live_decision_intervals(self):
        env, model, planner, _ = self.make_planner()
        short = SingleAgentLookaheadPolicy(model, env, replace(planner.config, rollout_seconds=5.))
        short._geometry()
        scene = planner.snapshot()
        own = replace(scene.aircraft[0], motion=AircraftState(0., 0., 0., 170.),
                      goal=(100., 0.), tas_cas_ratio=1., active_obstacle=None,
                      memory=replace(scene.aircraft[0].memory,
                                     progress_anchor_time=0., progress_anchor_distance=100.))
        # Closing traffic crosses the warning buffer across the three decisions;
        # endpoint shaping therefore differs at 5, 10 and 15 seconds.
        intruder = replace(scene.aircraft[1], motion=AircraftState(8.05, 0., 180., 170.),
                           tas_cas_ratio=1.)
        scene = replace(scene, aircraft=(own, intruder), time=0.)
        with ExitStack() as stack:
            for target in (planner, short):
                stack.enter_context(patch.object(target, "_inside_areas",
                    return_value=(True, *(False for _ in env.scenario.obstacles))))
                stack.enter_context(patch.object(target, "_navigation_reference",
                    side_effect=lambda aircraft, time: (aircraft, 0., 0)))
            macro = planner.transition(scene, 0, np.zeros(2))
            sequential = scene
            expected_reward = 0.
            discount = 1.
            for _ in range(3):
                step = short.transition(sequential, 0, np.zeros(2))
                expected_reward += discount * step.reward
                discount *= step.discount
                sequential = step.state
        self.assertAlmostEqual(macro.reward, expected_reward, places=11)
        self.assertAlmostEqual(macro.discount, model.gamma ** 3, places=14)
        self.assertEqual(macro.state.time, sequential.time)
        self.assertEqual(macro.state.aircraft, sequential.aircraft)

    def test_rejects_wrong_contract_discount_and_unsupported_filter(self):
        env, model, _, _ = self.make_planner()
        for config in (LookaheadConfig(discount=.99), LookaheadConfig(safety_filter=True),
                       LookaheadConfig(coordination_passes=2)):
            with self.assertRaises(ValueError):
                SingleAgentLookaheadPolicy(model, env, config)
        model.competition_mdp = "navigation-v5"
        with self.assertRaises(ValueError):
            SingleAgentLookaheadPolicy(model, env, LookaheadConfig())


if __name__ == "__main__":
    unittest.main()
