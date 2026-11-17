"""Arrived forecast aircraft require neither new paths nor arrival retesting."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from bluesky_zoo.lookahead import LookaheadConfig
from bluesky_zoo.lookahead_policy import LookaheadPolicy
from bluesky_zoo import lookahead_policy as planning
import test_lookahead_pipeline as fixtures


class ArrivedForecastOptimizationTests(unittest.TestCase):
    def make_scene(self):
        env, model = fixtures.LookaheadPipelineTests.make_model(self, n_agents=3)
        env.reset(seed=37)
        planner = LookaheadPolicy(model, env, LookaheadConfig(simulations=1, max_depth=1, rollout_seconds=5))
        scene = planner.snapshot()
        scene = replace(scene, aircraft=tuple(replace(a, goal=(1000., 1000.)) for a in scene.aircraft))
        commands = {a.agent: (a.motion.heading_deg, 150.) for a in scene.aircraft}
        return planner, scene, commands

    def test_arrived_neighbour_is_frozen_without_path_or_position_work(self):
        planner, scene, commands = self.make_scene()
        arrived = replace(scene.aircraft[2], arrived=True)
        scene = replace(scene, aircraft=(*scene.aircraft[:2], arrived))
        frames = []
        position = planning._position
        def active_position(motion):
            self.assertIsNot(motion, arrived.motion, "Already-arrived traffic was retested")
            return position(motion)
        with patch.object(planner, "_forecast_path", wraps=planner._forecast_path) as paths, \
                patch.object(planning, "_position", side_effect=active_position):
            result = planner.transition(scene, 0, np.array([0., -.4]), commands, frames.append)
        self.assertEqual([call.args[0].agent for call in paths.call_args_list],
                         [a.agent for a in scene.aircraft[:2]])
        self.assertIs(result.state.aircraft[2], arrived)
        self.assertEqual(len(frames), 5)
        for frame in frames:
            self.assertIs(frame.aircraft[2], arrived)

        # Removing traffic which has already arrived must give exactly the
        # same reward and dynamics for every remaining forecast aircraft.
        reference_scene = replace(scene, aircraft=scene.aircraft[:2])
        reference_frames = []
        reference = planner.transition(reference_scene, 0, np.array([0., -.4]), commands,
                                       reference_frames.append)
        self.assertEqual((result.reward, result.terminal, result.discount),
                         (reference.reward, reference.terminal, reference.discount))
        self.assertEqual(result.state.aircraft[:2], reference.state.aircraft)
        self.assertEqual(result.state.time, reference.state.time)
        for actual, expected in zip(frames, reference_frames):
            self.assertEqual(replace(actual, aircraft=actual.aircraft[:2]), expected)

    def test_new_arrival_is_sampled_before_removal_and_stays_frozen(self):
        planner, scene, commands = self.make_scene()
        neighbour = scene.aircraft[1]
        neighbour = replace(neighbour, goal=(neighbour.motion.north_nm, neighbour.motion.east_nm))
        scene = replace(scene, aircraft=(scene.aircraft[0], neighbour, scene.aircraft[2]))
        frames = []
        result = planner.transition(scene, 0, np.array([0., -.4]), commands, frames.append)
        self.assertFalse(frames[0].aircraft[1].arrived)
        self.assertTrue(all(frame.aircraft[1].arrived for frame in frames[1:]))
        arrival_motion = frames[0].aircraft[1].motion
        self.assertTrue(all(frame.aircraft[1].motion == arrival_motion for frame in frames))
        self.assertTrue(result.state.aircraft[1].arrived)
        self.assertEqual(result.state.aircraft[1].motion, arrival_motion)
        self.assertEqual(planner.env.sim_time, scene.time)
