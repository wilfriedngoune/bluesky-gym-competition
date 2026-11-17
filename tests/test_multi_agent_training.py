"""Training CSV records must describe real flights and unmodified metrics."""

import csv
from pathlib import Path
import unittest
from uuid import uuid4

from train_zoo import CompetitionMetricsCallback, METRIC_KEYS


class TrainingMetricsTests(unittest.TestCase):
    def test_reward_columns_use_episode_component_totals(self):
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"components_{uuid4().hex}.csv"
        self.addCleanup(path.unlink, missing_ok=True)
        callback = CompetitionMetricsCallback(path, reward_component_keys=("goal", "progress"))
        callback._on_training_start()
        callback.locals = {"infos": [{
            "final_metrics": dict.fromkeys(METRIC_KEYS, 0),
            "agent_id": "A", "scenario_index": 0,
            "reward_components": {"goal": 20.0, "progress": -0.3},
        }, {}]}
        callback._on_step()
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["reward_goal"]), 20)
        self.assertEqual(float(rows[0]["reward_progress"]), -0.3)

    def test_arrivals_and_timeouts_keep_raw_metrics_and_ignore_absorbing_slots(self):
        # Keep test artifacts in the workspace, including under restricted Windows runners.
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"training_{uuid4().hex}.csv"
        self.addCleanup(path.unlink, missing_ok=True)
        callback = CompetitionMetricsCallback(path)
        callback._on_training_start()
        metrics = {key: 0 for key in METRIC_KEYS}
        metrics.update(waypoint_reached=1, flight_time=812, total_reward=25.5)
        arrival = {
            "final_metrics": metrics, "agent_id": "A", "scenario_index": 0,
            "episode": {"r": 999, "l": 163},
        }
        callback.locals = {"infos": [arrival, {}, {"flight_time": 5}]}
        callback.num_timesteps = 30
        self.assertTrue(callback._on_step())
        timeout = {
            "final_metrics": {**metrics, "waypoint_reached": 0, "flight_time": 3000},
            "agent_id": "B", "scenario_index": 0,
        }
        callback.locals = {"infos": [{}, timeout, {}]}
        callback.num_timesteps = 60
        self.assertTrue(callback._on_step())
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["agent_id"] for row in rows], ["A", "B"])
        self.assertEqual([row["episode"] for row in rows], ["0", "1"])
        self.assertEqual([row["scenario_index"] for row in rows], ["0", "0"])
        self.assertEqual([row["waypoint_reached"] for row in rows], ["1", "0"])
        self.assertEqual([row["total_timesteps"] for row in rows], ["30", "60"])
        self.assertEqual(float(rows[0]["total_reward"]), 25.5)
        self.assertEqual(float(rows[1]["flight_time"]), 3000)


if __name__ == "__main__":
    unittest.main()
