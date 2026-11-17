"""Command deadbands retain accepted targets, not the aircraft's current heading."""

from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bluesky as bs
import numpy as np

from bluesky_gym.envs.common.command_filter import CommandDeadbandConfig, resolve_command_config
from bluesky_zoo.multi_agent_env import MultiAgentEnv
from scripts.evaluate_competition import policy_reward_kwargs
from test_multi_agent_env import fixed_scenario


class CommandDeadbandTests(unittest.TestCase):
    def make_env(self, heading=1, speed=1):
        env = MultiAgentEnv(scenario=fixed_scenario(), n_obstacles=0,
                           action_config=CommandDeadbandConfig(heading, speed))
        self.addCleanup(env.close)
        env.reset(seed=4)
        return env

    def command(self, env, heading, speed=170, agent="A"):
        state = env.navigation_states[agent]
        action = np.array([0, 2 * (speed - 120) / 100 - 1])
        with patch.object(state, "_get_navigation_reference", return_value=(heading, 0)), \
                patch.object(bs.stack, "stack") as stack:
            env._get_action(agent, action)
        return [call.args[0] for call in stack.call_args_list]

    def test_small_requests_accumulate_against_last_accepted_command(self):
        env = self.make_env()
        self.assertEqual(len(self.command(env, 30, 170)), 2)
        for heading, speed in ((30.3, 170.3), (30.6, 170.6), (30.9, 170.9)):
            self.assertEqual(self.command(env, heading, speed), [])
        state = env.navigation_states["A"]
        self.assertEqual(state.last_target_heading, 30)
        self.assertEqual(state.last_target_speed, 170)
        self.assertEqual(len(self.command(env, 31.2, 171.2)), 2)
        self.assertAlmostEqual(state.last_target_heading, 31.2)
        self.assertAlmostEqual(state.last_target_speed, 171.2)

    def test_heading_wrap_and_channels_are_independent(self):
        env = self.make_env()
        self.command(env, 179.8)
        self.assertEqual(self.command(env, -179.8), [])
        commands = self.command(env, -179.8, 172)
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].startswith("SPD A "))
        commands = self.command(env, -178, 172.2)
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].startswith("HDG A "))
        self.assertEqual(env.navigation_states["A"].last_target_speed, 172)

    def test_first_commands_are_sent_and_reset_is_aircraft_local(self):
        env = self.make_env()
        self.assertEqual(len(self.command(env, 0)), 2)
        self.assertEqual(self.command(env, 0.5), [])
        self.assertEqual(len(self.command(env, 0.5, agent="B")), 2)
        env.reset(seed=4)
        self.assertEqual(len(self.command(env, 0.5)), 2)

    def test_threshold_is_inclusive_and_zero_restores_original_commands(self):
        env = self.make_env()
        self.command(env, 30, 170)
        self.assertEqual(len(self.command(env, 31, 171)), 2)
        env.close()
        env = self.make_env(0, 0)
        self.command(env, 30)
        self.assertEqual(len(self.command(env, 30)), 2)

    def test_configuration_inheritance_and_evaluation_restore(self):
        saved = asdict(CommandDeadbandConfig(1, 2))
        self.assertEqual(asdict(resolve_command_config(saved)), saved)
        self.assertEqual(resolve_command_config(saved, heading_deadband_deg=0), CommandDeadbandConfig(0, 2))
        act = lambda observation: np.zeros(2)
        act.model = SimpleNamespace(competition_action_config=saved)
        self.assertEqual(policy_reward_kwargs(act, "navigation"), {"action_config": saved})
        act.model = SimpleNamespace()
        self.assertEqual(policy_reward_kwargs(act, "navigation"), {})
        for values in ((-1, 0), (181, 0), (0, -1), (float("nan"), 0), (0, float("inf"))):
            with self.assertRaises(ValueError):
                CommandDeadbandConfig(*values)


if __name__ == "__main__":
    unittest.main()
