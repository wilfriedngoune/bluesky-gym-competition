"""The sharing template must aggregate real final flights with correct denominators."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from uuid import uuid4

from scripts.competition_report import TOTAL_KEYS, build_report, render_report
from train_zoo import finish_run_reports


def flight(**updates):
    values = dict.fromkeys(TOTAL_KEYS, 0)
    values.update(flight_time=100, **updates)
    return values


class CompetitionReportTests(unittest.TestCase):
    def test_totals_and_clean_flights_include_clean_timeouts_but_not_occupancy_violations(self):
        records = [flight(waypoint_reached=1), flight(),
                   flight(waypoint_reached=1, intrusion_events=2, intrusion_time=7),
                   flight(time_in_restricted_area=3)]
        report = build_report(records, kind="ma", phase="evaluation", n_scenarios=2,
                              n_agents=2, seed=42)
        self.assertEqual(report["totals"]["waypoint_reached"], 2)
        self.assertEqual(report["totals"]["flight_time"], 400)
        self.assertEqual(report["totals"]["intrusion_events"], 2)
        self.assertEqual(report["totals"]["intrusion_time"], 7)
        self.assertEqual(report["totals"]["violation-free"], 2)
        self.assertEqual(report["totals"]["violation-free_success"], 1)
        self.assertEqual(report["rates"]["goal_completion_pct"], 50)
        self.assertEqual(report["rates"]["violation_free_success_pct"], 25)
        self.assertFalse(report["official_protocol"])

    def test_ma_1000_scenarios_means_10000_aircraft_records(self):
        records = [flight(waypoint_reached=1)] * 10000
        report = build_report(records, kind="ma", phase="evaluation", n_scenarios=1000, seed=42)
        self.assertTrue(report["official_protocol"])
        self.assertEqual(report["rates"]["goal_completion_pct"], 100)
        text = render_report([report])
        self.assertIn("| scenarios | NA | 1,000 episodes |", text)
        self.assertIn("| waypoint_reached | NA | 10,000 | goals reached |", text)
        with self.assertRaisesRegex(ValueError, "record count"):
            build_report(records[:1000], kind="ma", phase="evaluation", n_scenarios=1000, seed=42)

    def test_two_tracks_have_separate_denominators_and_no_fabricated_values(self):
        sa = build_report([flight(waypoint_reached=1)], kind="sa", phase="evaluation", n_scenarios=1)
        ma = build_report([flight(waypoint_reached=1), flight()], kind="ma", phase="evaluation",
                          n_scenarios=1, n_agents=2)
        text = render_report([sa, ma])
        self.assertIn("| goal completion | 100 | 50 | % |", text)
        self.assertIn("| flight_time | 100 | 200 | seconds |", text)

    def test_training_partial_scenario_is_labeled_and_empty_sample_is_not_zero_percent(self):
        records = [dict(flight(), scenario_index=0), dict(flight(), scenario_index=0),
                   dict(flight(waypoint_reached=1), scenario_index=1)]
        report = build_report(records, kind="ma", phase="training", n_agents=2)
        self.assertEqual(report["completed_training_scenarios"], 1)
        self.assertEqual(report["partial_training_scenarios"], 1)
        self.assertIn("TRAINING STATISTICS - NOT EVALUATION", render_report([report]))
        self.assertFalse(report["official_protocol"])
        empty = build_report([], kind="ma", phase="training")
        self.assertIsNone(empty["rates"]["goal_completion_pct"])
        self.assertIn("| goal completion | NA | NA |", render_report([empty]))

    def test_missing_or_invalid_metrics_cannot_silently_become_clean_flights(self):
        with self.assertRaises(KeyError):
            build_report([{"waypoint_reached": 1}], kind="sa", phase="training")
        for record in (flight(intrusion_events=-1), flight(intrusion_time=float("nan")),
                       flight(waypoint_reached=2), flight(sector_exit_events=0.5)):
            with self.assertRaises(ValueError):
                build_report([record], kind="sa", phase="training")


class EndOfRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base = self.directory / f"run_{uuid4().hex}"
        self.csv_path = Path(str(self.base) + ".csv")
        self.metadata_path = Path(str(self.base) + ".metadata.json")
        self.metadata_path.write_text("{}", encoding="utf-8")
        self.addCleanup(self.metadata_path.unlink, missing_ok=True)
        self.addCleanup(self.csv_path.unlink, missing_ok=True)
        for suffix in (".training_report.md", ".training_report.json"):
            self.addCleanup(Path(str(self.base) + suffix).unlink, missing_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[*TOTAL_KEYS, "scenario_index"])
            writer.writeheader()
            writer.writerow(dict(flight(waypoint_reached=1), scenario_index=0))

    def finish(self, episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            finish_run_reports(self.base, self.csv_path, mdp="navigation-v2", n_agents=10,
                               eval_episodes=episodes, metadata_path=self.metadata_path)

    def test_end_of_run_saves_table_then_starts_separate_standard_evaluation(self):
        with patch("train_zoo.subprocess.run") as run:
            self.finish(1000)
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("--episodes") + 1], "1000")
        self.assertEqual(args[args.index("--n-agents") + 1], "10")
        self.assertEqual(args[args.index("--mdp") + 1], "navigation-v2")
        self.assertTrue(Path(args[args.index("--model") + 1]).is_absolute())
        self.assertTrue(run.call_args.kwargs["check"])
        report = json.loads(Path(str(self.base) + ".training_report.json").read_text())["reports"][0]
        self.assertEqual(report["totals"]["waypoint_reached"], 1)
        status = json.loads(self.metadata_path.read_text())["reports"]
        self.assertEqual(status["evaluation_status"], "complete")

    def test_skipping_evaluation_keeps_training_report_and_failure_is_not_success(self):
        with patch("train_zoo.subprocess.run") as run:
            self.finish(0)
            run.assert_not_called()
        self.assertEqual(json.loads(self.metadata_path.read_text())["reports"]["evaluation_status"], "skipped")
        with patch("train_zoo.subprocess.run", side_effect=subprocess.CalledProcessError(1, "evaluation")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.finish(1)
        self.assertTrue(Path(str(self.base) + ".training_report.md").exists())
        self.assertEqual(json.loads(self.metadata_path.read_text())["reports"]["evaluation_status"], "failed")


if __name__ == "__main__":
    unittest.main()
