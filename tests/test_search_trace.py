"""Search diagnostics observe the existing tree and leave flight control intact."""

from copy import deepcopy
import csv
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import json

import bluesky as bs
import numpy as np

from bluesky_zoo.lookahead import LookaheadConfig, PolicyExpansion, Transition, policy_value_search
from bluesky_zoo.lookahead_policy import LookaheadPolicy
from scripts.search_trace import SearchTrace
import test_lookahead_pipeline as fixtures


class SearchObserverTests(unittest.TestCase):
    def run_search(self, observer=None, simulations=12):
        calls = []
        actions = np.array([[0., 0.], [1., 0.], [-1., 0.]])
        def evaluate(state):
            return PolicyExpansion(actions, np.array([.8, .1, .1]), float(state[1]) * .2)
        def transition(state, action):
            calls.append((state, tuple(action)))
            after = (state[0] + 1, state[1] + float(action[0]))
            return Transition(after, float(action[0]), terminal=after[0] >= 3)
        return policy_value_search((0, 0.), evaluate=evaluate, transition=transition,
                                   config=LookaheadConfig(simulations=simulations, max_depth=3),
                                   observer=observer), calls

    def test_observer_preserves_actions_values_visits_and_rollout_count(self):
        records = []
        normal, calls = self.run_search()
        traced, traced_calls = self.run_search(records.append)
        for field in ("action", "visit_counts", "priors", "action_values"):
            np.testing.assert_array_equal(getattr(normal, field), getattr(traced, field))
        self.assertEqual((normal.value, normal.expanded_nodes, normal.deepest_depth),
                         (traced.value, traced.expanded_nodes, traced.deepest_depth))
        self.assertEqual(calls, traced_calls)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(sum(c.visits for c in record.candidates), 12)
        self.assertEqual(record.candidates[record.selected_index].action, tuple(normal.action))
        for candidate in record.candidates:
            self.assertLessEqual(candidate.explored_depth, 3)
            for step in candidate.principal_path:
                self.assertIsNotNone(step.state_after)
                self.assertEqual(step.state_after[0], step.state_before[0] + 1)

    def test_unvisited_candidates_are_explicitly_unscored(self):
        records = []
        self.run_search(records.append, simulations=1)
        unvisited = [c for c in records[0].candidates if not c.visits]
        self.assertEqual(len(unvisited), 2)
        for candidate in unvisited:
            self.assertIsNone(candidate.mean_return)
            self.assertIsNone(candidate.first_reward)
            self.assertIsNone(candidate.first_leaf_value)
            self.assertEqual(candidate.principal_path, ())


class SearchTraceTests(unittest.TestCase):
    assert_nested_equal = fixtures.LookaheadPipelineTests.assert_nested_equal

    def test_real_logger_is_read_only_and_records_matching_candidate_forecasts(self):
        self._exercise_logger(safety_filter=False)

    def test_filtered_logger_records_executed_selection_and_separate_safety_forecast(self):
        self._exercise_logger(safety_filter=True)

    def _exercise_logger(self, safety_filter):
        from dataclasses import replace
        env, model = fixtures.LookaheadPipelineTests.make_model(self, n_agents=3)
        observations, _ = env.reset(seed=37)
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("searchtrace_" + uuid4().hex)
        directory.mkdir(parents=True)
        model_path = directory / "model.zip"
        model.save(model_path)
        self.addCleanup(model_path.unlink, missing_ok=True)
        config = replace(LookaheadConfig(**model.competition_lookahead_config),
                         safety_filter=safety_filter, safety_horizon_seconds=20.)
        act = SimpleNamespace(lookahead_config=config, disable_lookahead=False)
        planner = LookaheadPolicy(model, env, config)
        expected = planner.actions(observations, env.agents)
        expected_search = deepcopy(planner.last_diagnostics)
        before = fixtures.LookaheadPipelineTests.live_state(self, env)
        own, *others = env.agents
        trace = SearchTrace(directory / "trace", env, act, model_path=model_path, seed=37, scenario_start=1,
                            start=0, end=5, agents=[own], neighbours=others)
        for path in [*trace.paths.values(), trace.settings_path]:
            self.addCleanup(path.unlink, missing_ok=True)
        success = False
        try:
            planner.observer = trace
            trace.begin_episode(0)
            with patch.object(env, "_get_obs", side_effect=AssertionError("live observation hook")), \
                    patch.object(bs.sim, "step", side_effect=AssertionError("live simulator step")), \
                    patch.object(bs.stack, "stack", side_effect=AssertionError("live command")):
                trace.capture_actual()
                actual = planner.actions(observations, env.agents)
            for agent in env.agents:
                np.testing.assert_array_equal(actual[agent], expected[agent])
            self.assertEqual(planner.last_diagnostics, expected_search)
            self.assert_nested_equal(before, fixtures.LookaheadPipelineTests.live_state(self, env))
            env.step(actual)
            trace.record_commands(0., actual)
            trace.capture_actual()
            success = True
        finally:
            trace.close(completed=success)
        def read(name):
            with trace.paths[name].open(newline="", encoding="utf-8") as file:
                return list(csv.DictReader(file))
        candidates, forecasts, encounters, commands = [read(name) for name in ("candidates", "forecasts", "actual", "commands")]
        self.assertEqual(len(candidates), config.candidate_count)
        self.assertEqual(sum(int(row["visits"]) for row in candidates), config.simulations)
        self.assertEqual(sum(row["selected"] == "True" for row in candidates), 1)
        self.assertTrue(all(row["safety_filter_enabled"] == str(safety_filter) for row in candidates))
        if safety_filter:
            self.assertTrue(all(float(row["safety_forecast_seconds"]) == 20. for row in candidates))
            chosen = next(row for row in candidates if row["selected"] == "True")
            np.testing.assert_array_equal([float(chosen["action_heading"]), float(chosen["action_speed"])], actual[own])
        self.assertTrue(forecasts)
        self.assertTrue(all(row["root_agent"] == own and row["other_agent"] in others for row in forecasts))
        self.assertTrue(all(float(row["time"]) > float(row["decision_time"]) for row in forecasts))
        self.assertEqual({float(row["time"]) for row in encounters}, {0., 5.})
        self.assertEqual({row["agent"] for row in commands}, set(env.possible_agents))
        settings = json.loads(trace.settings_path.read_text())
        self.assertTrue(settings["completed"])
        self.assertEqual(settings["decision_seconds"], 5)
        self.assertEqual(len(settings["model_sha256"]), 64)
        existing = trace.settings_path.read_bytes()
        with self.assertRaises(FileExistsError):
            SearchTrace(directory / "trace", env, act, model_path=model_path, seed=37, scenario_start=1)
        self.assertEqual(trace.settings_path.read_bytes(), existing)

    def test_invalid_window_and_aircraft_fail_before_creating_files(self):
        env = SimpleNamespace(possible_agents=["A", "B"])
        act = SimpleNamespace()
        for args in (dict(start=-1), dict(start=5, end=4), dict(end=float("nan")), dict(agents=["unknown"])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                SearchTrace("unused", env, act, model_path="unused.zip", seed=42, scenario_start=1, **args)


if __name__ == "__main__":
    unittest.main()
