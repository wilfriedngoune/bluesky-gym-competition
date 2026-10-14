"""Regression checks for evaluation protocol, metrics and immutable world settings."""

import csv
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from scripts import evaluate_competition as official
from scripts.evaluate_single_agent import evaluate_models, run_development
from scripts.experiment_support import FIXED_CONFIG, metrics_summary, validate_environment


@contextmanager
def temporary_directory(root):
    # Use normal inherited Windows permissions instead of tempfile's mode 0700.
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if root.resolve() not in resolved.parents:
            raise ValueError("Test cleanup must stay inside the test artifact directory")
        shutil.rmtree(resolved)


def record(**changes):
    result = {key: 0.0 for key in official.METRIC_KEYS}
    result.update(waypoint_reached=1, flight_time=10.0, total_reward=100.0)
    result.update(changes)
    return result


@dataclass
class FakeScenario:
    identifier: int


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scenario = FakeScenario(int(self.rng.integers(0, 1000000)))
        self.steps = 0
        return {}, {}

    def step(self, action):
        self.steps += 1
        # The first step contains an intermediate metric; final info is authoritative.
        info = record(flight_time=self.steps, intrusion_events=1 if self.steps == 1 else 0)
        return {}, 99.0, self.steps == 2, False, info

    def close(self):
        self.closed = True


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_safe_completion_requires_goal_and_zero_events_and_time(self):
        summary = metrics_summary([
            record(), record(waypoint_reached=0, flight_time=3000),
            record(intrusion_time=1), record(restricted_area_events=1),
            record(time_outside_sector=1),
        ])
        self.assertEqual(summary["safe_goal_completion_rate"], 0.2)
        self.assertEqual(summary["goal_completion_rate"], 0.8)
        self.assertEqual(summary["flight_time_success_mean_s"], 10.0)
        self.assertEqual(summary["failure_counts"], {
            "goal_not_reached": 1, "loss_of_separation": 1,
            "restricted_area": 1, "sector_exit": 1,
        })

    def test_no_success_has_no_successful_flight_time(self):
        self.assertIsNone(metrics_summary([record(waypoint_reached=0)])["flight_time_success_mean_s"])

    def test_development_seeds_once_and_pairs_scenarios(self):
        with temporary_directory(self.temp_root) as temp:
            sequences = []
            for seed in (1234, 1234, 5678):
                folder = Path(temp) / str(len(sequences))
                folder.mkdir()
                env = FakeEnv()
                records = run_development(env, lambda obs: [0, 0], 3, seed, folder)
                self.assertEqual(env.seeds, [seed, None, None])
                self.assertEqual([item["flight_time"] for item in records], [2, 2, 2])
                self.assertEqual((folder / "failures.jsonl").read_text(), "")
                with (folder / "episodes.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row["safe_goal_completion"] for row in rows], ["True"] * 3)
                sequences.append([json.loads(line)["scenario_sha256"] for line in (folder / "scenarios.jsonl").read_text().splitlines()])
            self.assertEqual(sequences[0], sequences[1])
            self.assertNotEqual(sequences[0], sequences[2])

    def test_official_loop_seeds_first_reset_only_and_closes(self):
        env = FakeEnv()
        with patch.object(official, "make_env", return_value=env):
            rows = official.run_single_agent(3, lambda obs: [0, 0])
        self.assertEqual(env.seeds, [42, None, None])
        self.assertEqual(len(rows), 3)
        self.assertTrue(env.closed)

    def test_invalid_protocol_settings_rejected_before_run_creation(self):
        for protocol, seed, episodes in (
            ("official", 1234, None), ("official", None, 20),
            ("development", 42, 20), ("development", 1234, 0),
        ):
            with self.subTest(protocol=protocol, seed=seed, episodes=episodes), \
                 patch("scripts.evaluate_single_agent.create_run") as create:
                with self.assertRaises(ValueError):
                    evaluate_models(["neutral"], protocol, seed, episodes)
                create.assert_not_called()

    def test_single_agent_model_loader_rejects_multi_agent_use(self):
        with self.assertRaisesRegex(ValueError, "single-agent PPO only"):
            official.load_policy("ma", "single_agent_model")

    def test_tracked_official_run_delegates_to_canonical_harness(self):
        with temporary_directory(self.temp_root) as temp, \
             patch("scripts.evaluate_single_agent.provenance", return_value={}), \
             patch.object(official, "run_single_agent", return_value=[record()] * 1000) as run, \
             patch.object(official, "summarize"):
            folder = evaluate_models(["neutral"], "official", out_dir=Path(temp) / "run")
            self.assertEqual(run.call_args.args[0], 1000)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual((manifest["seed"], manifest["episodes"], manifest["status"]), (42, 1000, "completed"))

    def test_development_failure_closes_environment_and_marks_manifest(self):
        env = FakeEnv()
        with temporary_directory(self.temp_root) as temp, \
             patch("scripts.evaluate_single_agent.provenance", return_value={}), \
             patch("scripts.evaluate_single_agent.validate_environment"), \
             patch("scripts.evaluate_single_agent.environment_config", return_value={}), \
             patch.object(official, "make_env", return_value=env), \
             patch.object(env, "step", side_effect=RuntimeError("simulation failed")):
            folder = Path(temp) / "failed_run"
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                evaluate_models(["neutral"], episodes=1, out_dir=folder)
            self.assertTrue(env.closed)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("simulation failed", manifest["error"])

    def test_guard_rejects_world_changes_and_scoring_overrides(self):
        from bluesky_gym.envs.competition_env import CompetitionEnv
        from core.scenario import ScenarioGenerator

        env = CompetitionEnv.__new__(CompetitionEnv)
        for key, value in FIXED_CONFIG.items():
            setattr(env, key, value)
        env._fixed_scenario = None
        env.scenario_generator = ScenarioGenerator.__new__(ScenarioGenerator)
        validate_environment(env)
        for key, value in FIXED_CONFIG.items():
            with self.subTest(key=key):
                setattr(env, key, None)
                with self.assertRaises(ValueError):
                    validate_environment(env)
                setattr(env, key, value)

        class ChangedScoring(CompetitionEnv):
            def _update_metrics(self, ac_id, dt=None):
                pass

        changed = ChangedScoring.__new__(ChangedScoring)
        changed.__dict__.update(env.__dict__)
        with self.assertRaises(ValueError):
            validate_environment(changed)


if __name__ == "__main__":
    unittest.main()
