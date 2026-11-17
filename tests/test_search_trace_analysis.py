"""Prediction analysis joins the same episode, directed pair and actual time."""

import csv
import json
from pathlib import Path
import unittest
from uuid import uuid4

from scripts.analyze_search_trace import analyze, write_analysis


class SearchTraceAnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("searchanalysis_" + uuid4().hex)
        directory.mkdir(parents=True)
        self.prefix = directory / "trace"
        self.settings = Path(str(self.prefix) + ".settings.json")
        self.settings.write_text(json.dumps({"completed": True, "decision_seconds": 5}), encoding="utf-8")
        self.addCleanup(self.settings.unlink, missing_ok=True)

    def csv(self, name, rows):
        path = Path(str(self.prefix) + f".{name}.csv")
        self.addCleanup(path.unlink, missing_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_prediction_miss_and_wrong_neighbour_command_are_reported(self):
        self.csv("candidates", [dict(episode_index=8, time=0, root_agent="A", candidate_index=0,
                    selected=True, visits=3, mean_return=10, explored_depth=1, principal_path_depth=1,
                    forecast_duration_seconds=5)])
        self.csv("forecasts", [dict(episode_index=8, decision_time=0, root_agent="A", candidate_index=0,
                    other_agent="B", time=5, predicted_separation_nm=10, own_north_nm=5, own_east_nm=0,
                    other_north_nm=15, other_east_nm=0, assumed_other_heading_deg=5, assumed_other_speed_kt=150)])
        self.csv("actual", [dict(episode_index=8, time=t, own_agent="A", other_agent="B",
                    own_north_nm=5, own_east_nm=0, other_north_nm=9, other_east_nm=0,
                    separation_nm=4, tcpa_seconds=50, dcpa_nm=1) for t in (0, 5)])
        self.csv("commands", [dict(episode_index=8, time=0, agent="B", target_heading_deg=95, target_speed_kt=120)])
        summary, comparisons = analyze(self.prefix)
        self.assertEqual(summary["root_decisions"], 1)
        self.assertEqual(summary["predicted_safe_actual_intrusion_windows"], 1)
        self.assertEqual(summary["predicted_and_actual_intrusion_windows"], 0)
        self.assertEqual(summary["mean_other_command_heading_difference_deg"], 90)
        row = comparisons[0]
        self.assertEqual(row["mean_abs_separation_error_nm"], 6)
        self.assertEqual(row["first_decision_own_position_error_nm"], 0)
        self.assertEqual(row["first_decision_other_position_error_nm"], 6)
        self.assertEqual(row["other_command_speed_difference_kt"], 30)
        write_analysis(self.prefix)
        for suffix in ("json", "csv", "md"):
            path = Path(str(self.prefix) + ".analysis." + suffix)
            self.addCleanup(path.unlink, missing_ok=True)
            self.assertTrue(path.is_file())

    def test_incomplete_replay_is_not_treated_as_complete_evidence(self):
        self.settings.write_text('{"completed": false}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze(self.prefix)


if __name__ == "__main__":
    unittest.main()
