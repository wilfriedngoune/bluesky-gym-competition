"""Targeted diagnostic replay keeps the reset stream and caps work explicitly."""

import contextlib
import csv
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from scripts import evaluate_competition as evaluation
from scripts.navigation_trace import NavigationTrace
import train_zoo


class ReplayEnv:
    possible_agents = ["A"]

    def __init__(self, finish=True):
        self.finish = finish
        self.seeds = []
        self.steps = 0
        self.closed = False

    def observation_space(self, agent):
        return "observations"

    def action_space(self, agent):
        return "actions"

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.agents = ["A"]
        return {"A": {}}, {}

    def step(self, actions):
        self.steps += 1
        if self.finish:
            self.agents = []
        return {"A": {}}, {}, {"A": self.finish}, {"A": False}, {"A": {"waypoint_reached": 0}}

    def close(self):
        self.closed = True


class TargetedReplayTests(unittest.TestCase):
    def act(self):
        act = lambda observation: np.zeros(2)
        act.model = SimpleNamespace(observation_space="observations", action_space="actions",
                                    competition_mdp="navigation")
        return act

    def test_skips_reset_stream_without_policy_actions_and_seeds_once(self):
        env = ReplayEnv()
        trace = Mock()
        trace.before.return_value = {}
        with patch.object(evaluation, "load_policy", return_value=self.act()), \
                patch.object(evaluation, "actions_for_agents", return_value={"A": np.zeros(2)}) as actions, \
                patch.object(train_zoo, "make_env", return_value=env), \
                patch("scripts.navigation_trace.NavigationTrace", return_value=trace), \
                contextlib.redirect_stdout(io.StringIO()):
            train_zoo.replay("checkpoint", n_episodes=2, seed=42, scenario_start=3)
        self.assertEqual(env.seeds, [42, None, None, None])
        self.assertEqual(actions.call_count, 2)
        self.assertEqual(env.steps, 2)
        self.assertEqual([call.args[-1] for call in trace.after.call_args_list], [2, 3])
        self.assertTrue(env.closed)
        trace.close.assert_called_once()

    def test_step_cap_stops_unfinished_scenario_without_fabricating_final_info(self):
        env = ReplayEnv(finish=False)
        output = io.StringIO()
        with patch.object(evaluation, "load_policy", return_value=self.act()), \
                patch.object(evaluation, "actions_for_agents", return_value={"A": np.zeros(2)}), \
                patch.object(train_zoo, "make_env", return_value=env), \
                contextlib.redirect_stdout(output):
            train_zoo.replay("checkpoint", n_episodes=1, seed=42, max_steps=2)
        self.assertEqual(env.steps, 2)
        self.assertEqual(env.seeds, [42])
        self.assertTrue(env.closed)
        self.assertIn("unfinished flights have no final evaluation record", output.getvalue())
        self.assertNotIn("metrics=", output.getvalue())

    def test_invalid_replay_bounds_fail_before_loading_checkpoint(self):
        with patch.object(evaluation, "load_policy") as loader:
            for args in (dict(scenario_start=0), dict(max_steps=0), dict(n_episodes=0)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    train_zoo.replay(**args)
        loader.assert_not_called()

    def test_trace_records_wrapped_target_changes_and_safety_durations(self):
        path = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("trace_" + uuid4().hex + ".csv")
        self.addCleanup(path.unlink, missing_ok=True)
        state = SimpleNamespace(last_reference_heading=0, last_target_heading=-179.8, last_target_speed=150.4,
                                last_navigation_mode=2, last_command_obstacle=None)
        env = SimpleNamespace(navigation_states={"A": state})
        trace = NavigationTrace(path)
        try:
            trace.after(env, {"A": dict(agent_id="A", time_before=0,
                                        previous_target_heading_deg=179.8, previous_target_speed_kt=150)},
                        {"A": np.zeros(2)}, {"A": dict(flight_time=15, waypoint_reached=0,
                            intrusion_events=0, restricted_area_events=0, sector_exit_events=1,
                            intrusion_time=0, time_in_restricted_area=0, time_outside_sector=12)}, 9)
        finally:
            trace.close()
        with path.open(newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        self.assertAlmostEqual(float(row["target_heading_change_deg"]), 0.4)
        self.assertAlmostEqual(float(row["target_speed_change_kt"]), 0.4)
        self.assertEqual(float(row["time_outside_sector"]), 12)
        self.assertEqual(row["episode_index"], "9")


if __name__ == "__main__":
    unittest.main()
