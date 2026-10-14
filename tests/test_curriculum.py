"""Curriculum scenario integrity, stage gates, and trainer progression."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.scenario import ScenarioGenerator
from scripts.curriculum import (
    STAGES, CurriculumStage, ScenarioSubsetGenerator, gate_passed,
    make_training_env, set_training_stage,
)
from scripts.experiment_support import metrics_summary, validate_environment
from scripts.train_curriculum import train_curriculum
from train_single_agent import EpisodeMetricsCallback
from test_experiments import record, temporary_directory


class FakePPO:
    instances = []

    def __init__(self, policy, env, **kwargs):
        self.env = env
        self.num_timesteps = 0
        self.n_steps = kwargs["n_steps"]
        self._custom_logger = False
        self.set_env_calls = 0
        self.instances.append(self)

    def set_logger(self, logger):
        self.logger = logger
        self._custom_logger = True

    def learn(self, total_timesteps, reset_num_timesteps, callback):
        if reset_num_timesteps:
            raise AssertionError("Curriculum must preserve the cumulative training counter")
        self.num_timesteps += int(np.ceil(total_timesteps / self.n_steps)) * self.n_steps

    def save(self, path):
        Path(path).write_text(f"test model at {self.num_timesteps}")

    def set_env(self, env, force_reset):
        if not force_reset:
            raise AssertionError("A stage transition must reset the episode")
        self.set_env_calls += 1

    def get_env(self):
        return self.env


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_subsets_preserve_sector_agents_hazard_geometry_and_rng_stream(self):
        for stage in STAGES:
            with self.subTest(stage=stage.name):
                reduced_rng = np.random.default_rng(1234)
                original_rng = np.random.default_rng(1234)
                generator = ScenarioSubsetGenerator(ScenarioGenerator(n_intruders=10), stage)
                original = ScenarioGenerator(n_intruders=10)
                for _ in range(2):
                    subset = generator.generate(reduced_rng)
                    full = original.generate(original_rng)
                    expected = asdict(full)
                    expected["obstacles"] = expected["obstacles"][:stage.obstacles]
                    expected["intruder_routes"] = expected["intruder_routes"][:stage.intruders]
                    self.assertEqual(asdict(subset), expected)
                self.assertEqual(reduced_rng.bit_generator.state, original_rng.bit_generator.state)

    def test_actual_stage_resets_keep_spaces_and_restore_original_generator(self):
        from bluesky_gym.envs.competition_env import CompetitionEnv

        env = make_training_env(STAGES[0])
        observation_space = env.observation_space
        action_space = env.action_space
        try:
            for index, stage in enumerate(STAGES):
                set_training_stage(env, stage)
                obs, _ = env.reset(seed=1000 + index)
                self.assertIs(env.observation_space, observation_space)
                self.assertIs(env.action_space, action_space)
                self.assertTrue(observation_space.contains(obs))
                self.assertEqual(obs["intruder_distance"].shape, (10,))
                self.assertEqual(obs["obstacle_distance"].shape, (5,))
                self.assertEqual(len(env.intruder_ids), stage.intruders)
                self.assertLessEqual(len(env.scenario.obstacles), stage.obstacles)
                self.assertIs(type(env)._get_obs, CompetitionEnv._get_obs)
                if stage.name == "full":
                    self.assertIs(type(env.scenario_generator), ScenarioGenerator)
                    validate_environment(env)
                else:
                    with self.assertRaisesRegex(ValueError, "original random scenario generator"):
                        validate_environment(env)
        finally:
            env.close()

    def test_gate_requires_completion_and_safety_and_checks_slot_limits(self):
        summary = metrics_summary([record(), record(intrusion_events=1)])
        self.assertTrue(gate_passed(summary, 0.95, 0.50))
        self.assertFalse(gate_passed(summary, 0.95, 0.80))
        with self.assertRaises(ValueError):
            CurriculumStage("invalid", 6, 11)

    def test_episode_log_appends_across_learning_chunks(self):
        with temporary_directory(self.temp_root) as folder:
            callback = EpisodeMetricsCallback(folder / "episodes.csv", append=True)
            for steps in (64, 128):
                callback._on_training_start()
                callback.num_timesteps = steps
                callback.locals = {"dones": [True], "infos": [record()]}
                callback._on_step()
            callback.close()
            with (folder / "episodes.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["timesteps"] for row in rows], ["64", "128"])

    def run_fake_trainer(self, folder, summary, **overrides):
        FakePPO.instances = []
        env = MagicMock()
        with patch("scripts.train_curriculum.provenance", return_value={}), \
             patch("scripts.train_curriculum.make_training_env", return_value=env), \
             patch("scripts.train_curriculum.set_training_stage"), \
             patch("scripts.train_curriculum.environment_config", return_value={}), \
             patch("scripts.train_curriculum.Monitor", side_effect=lambda value, **kwargs: value), \
             patch("scripts.train_curriculum.PPO", FakePPO), \
             patch("scripts.train_curriculum.configure", return_value=MagicMock()), \
             patch("scripts.train_curriculum.checkpoint_evaluation", return_value=summary,
                   side_effect=summary if callable(summary) else None) as evaluation:
            options = dict(out_dir=folder, stage_budget=128, chunk_steps=64,
                           min_stage_steps=128, n_steps=64, batch_size=32,
                           eval_episodes=2, passes_required=2)
            options.update(overrides)
            run = train_curriculum(**options)
        env.close.assert_called_once()
        return run, evaluation

    def test_progression_reuses_one_model_and_evaluates_full_distribution(self):
        with temporary_directory(self.temp_root) as folder:
            run, evaluation = self.run_fake_trainer(folder / "run", metrics_summary([record(), record()]))
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["curriculum_complete"])
            self.assertEqual(len(FakePPO.instances), 1)
            self.assertEqual(FakePPO.instances[0].set_env_calls, 5)
            self.assertEqual(manifest["actual_timesteps"], 768)
            self.assertTrue((run / "model.zip").exists())
            self.assertEqual(len(manifest["stages"]), 6)
            self.assertEqual(sum(len(call.args) == 4 for call in evaluation.call_args_list), 6)

    def test_unmet_gate_stops_and_preserves_checkpoint_without_final_model(self):
        with temporary_directory(self.temp_root) as folder:
            run, _ = self.run_fake_trainer(folder / "run", metrics_summary([record(waypoint_reached=0)]))
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "stage_not_mastered")
            self.assertFalse(manifest["curriculum_complete"])
            self.assertEqual(len(manifest["stages"]), 1)
            self.assertTrue((run / "latest_model.zip").exists())
            self.assertFalse((run / "model.zip").exists())

    def test_failed_check_resets_consecutive_passes(self):
        passed = metrics_summary([record()])
        failed = metrics_summary([record(waypoint_reached=0)])
        results = iter([passed, failed, passed, passed])
        with temporary_directory(self.temp_root) as folder:
            run, _ = self.run_fake_trainer(folder / "run", lambda *args: next(results),
                                          stage_budget=192, min_stage_steps=0)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "stage_not_mastered")
            checks = manifest["stages"][0]["checks"]
            self.assertEqual([check["consecutive_passes"] for check in checks], [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
