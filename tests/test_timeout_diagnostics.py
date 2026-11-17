"""Timeout reports preserve provenance and never invent terminal positions."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from bluesky_zoo.lookahead import LookaheadConfig
from scripts import diagnose_navigation_timeouts as diagnostics


class TimeoutDiagnosticsTests(unittest.TestCase):
    def directory(self):
        root = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        directory = root / ("timeouts_" + uuid4().hex)
        directory.mkdir()
        def cleanup():
            resolved = directory.resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise RuntimeError("Refusing to clean outside test workspace")
            shutil.rmtree(resolved)
        self.addCleanup(cleanup)
        return directory

    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def evaluation(self, directory):
        records = []
        for index in range(4):
            records.append(dict(episode_index=index, waypoint_reached=int(index in (0, 2)),
                                flight_time=1000. if index in (0, 2) else 3000.,
                                **{name: 0. for name in diagnostics.SAFETY_FIELDS}))
        records[-1]["time_in_restricted_area"] = 1.
        path = directory / "evaluation.csv"
        self.write_csv(path, records)
        return path

    def trace_rows(self, agent="A", *, reached=False, end=3000.):
        rows = []
        for index, (lat, lon) in enumerate(((52., 4.), (52.1, 4.), (52.1, 4.1), (52., 4.1), (52., 4.))):
            rows.append(dict(episode_index=0, agent_id=agent, time_before=index * 75.,
                             time_after=end if index == 4 else index * 75. + 5.,
                             lat_before=lat, lon_before=lon, heading_before_deg=(index * 90.) % 360,
                             speed_before_kt=150., goal_distance_before_nm=20., inside_obstacle=False,
                             inside_sector=True, waypoint_reached=int(reached and index == 4),
                             **{name: 0. for name in diagnostics.SAFETY_FIELDS}))
        return rows

    def test_inventory_uses_ordered_blocks_and_has_no_aircraft_identity(self):
        directory = self.directory()
        source = self.evaluation(directory)
        inventory = diagnostics.inventory_failures(source, n_agents=2)
        self.assertEqual(inventory["missed_arrivals"], 2)
        self.assertEqual(inventory["clean_missed_arrivals"], 1)
        self.assertEqual([row["scenario"] for row in inventory["scenarios"]], [1, 2])
        self.assertEqual([row["records"][0]["evaluation_row_index"] for row in inventory["scenarios"]], [1, 3])
        self.assertTrue(all("agent_id" not in record for row in inventory["scenarios"] for record in row["records"]))

    def test_inventory_rejects_incomplete_blocks_and_mismatched_seed_metadata(self):
        directory = self.directory()
        source = self.evaluation(directory)
        with self.assertRaisesRegex(ValueError, "complete ordered"):
            diagnostics.inventory_failures(source, n_agents=3)
        source.with_suffix(".settings.json").write_text(json.dumps(dict(seed=43, n_agents=2, n_scenarios=2)))
        with self.assertRaisesRegex(ValueError, "seed"):
            diagnostics.inventory_failures(source, n_agents=2)

    def test_trace_distinguishes_arrival_timeout_and_capped_with_heuristic_windows(self):
        directory = self.directory()
        source = directory / "trace.csv"
        rows = self.trace_rows("arrival", reached=True, end=305.)
        rows += self.trace_rows("timeout")
        rows += self.trace_rows("capped", end=305.)
        self.write_csv(source, rows)
        report = diagnostics.analyze_trace(source)
        self.assertEqual(report["summary"], dict(arrivals=1, timeouts=1, capped_or_incomplete=1))
        for aircraft in report["aircraft"]:
            self.assertEqual(aircraft["last_observed_goal_distance_nm"], 20.)
            self.assertEqual(aircraft["last_distance_observation_time_seconds"], 300.)
            self.assertFalse(aircraft["distance_is_terminal_position"])
            self.assertEqual(len(aircraft["low_progress_windows"]), 1)
            self.assertEqual(len(aircraft["possible_circling_windows"]), 1)
            self.assertGreater(aircraft["observed_path_length_nm"], 2.)
            self.assertEqual(aircraft["absolute_observed_heading_change_deg"], 360.)

    def test_list_only_does_not_load_policy_and_refuses_output_overwrite(self):
        directory = self.directory()
        source = self.evaluation(directory)
        model = directory / "model.zip"
        model.write_bytes(b"checkpoint-provenance-test")
        output = directory / "diagnostics"
        with patch("scripts.evaluate_competition.load_policy", side_effect=AssertionError("Policy loaded")), redirect_stdout(io.StringIO()):
            report = diagnostics.run_diagnostics(model=model, evaluation=source, out_dir=output,
                                                  list_only=True, n_agents=2)
        self.assertTrue(report["completed"])
        self.assertEqual(report["phase"], "inventory_only")
        self.assertFalse(report["replay_coverage_complete"])
        self.assertEqual(report["replays"], [])
        self.assertEqual(len(report["model_sha256"]), 64)
        existing = (output / "diagnostics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            diagnostics.run_diagnostics(model=model, evaluation=source, out_dir=output, list_only=True, n_agents=2)
        self.assertEqual((output / "diagnostics.json").read_bytes(), existing)

    def test_replay_receives_saved_settings_and_caps_remain_incomplete(self):
        directory = self.directory()
        source = self.evaluation(directory)
        model = directory / "model.zip"
        model.write_bytes(b"checkpoint-provenance-test")
        def replay(*args, **kwargs):
            self.assertEqual(kwargs["lookahead_preset"], "saved")
            self.assertIsNone(kwargs["lookahead_simulations"])
            self.assertEqual(kwargs["scenario_start"], 2)
            self.assertEqual(kwargs["max_steps"], 4)
            self.write_csv(Path(kwargs["trajectory_csv"]), self.trace_rows(end=305.))
        policy = SimpleNamespace(lookahead_config=LookaheadConfig())
        with patch("scripts.evaluate_competition.load_policy", return_value=policy), \
                patch("scripts.evaluate_competition.policy_reward_kwargs", return_value={}), \
                patch("train_zoo.replay", side_effect=replay) as replay_call, redirect_stdout(io.StringIO()):
            report = diagnostics.run_diagnostics(model=model, evaluation=source, out_dir=directory / "diagnostics",
                scenarios=[2], max_steps=4, n_agents=2)
        replay_call.assert_called_once()
        self.assertTrue(report["completed"])
        self.assertFalse(report["replay_coverage_complete"])
        self.assertEqual(report["replays"][0]["summary"]["capped_or_incomplete"], 1)
        self.assertFalse(report["inference_overridden"])

    def test_rejects_success_only_scenario_selection_before_creating_outputs(self):
        directory = self.directory()
        source = self.evaluation(directory)
        model = directory / "model.zip"
        model.write_bytes(b"checkpoint-provenance-test")
        output = directory / "diagnostics"
        with self.assertRaisesRegex(ValueError, "select scenarios"):
            diagnostics.run_diagnostics(model=model, evaluation=source, out_dir=output, list_only=True,
                                        scenarios=[3], n_agents=2)
        self.assertFalse(output.exists())

    def test_saved_actor_only_mode_is_preserved_unless_budget_explicitly_opts_in(self):
        directory = self.directory()
        source = self.evaluation(directory)
        model = directory / "model.zip"
        model.write_bytes(b"checkpoint-provenance-test")
        policy = SimpleNamespace(lookahead_config=LookaheadConfig(), disable_lookahead=True)
        def replay(*args, **kwargs):
            self.assertIsNone(kwargs["disable_lookahead"])
            self.write_csv(Path(kwargs["trajectory_csv"]), self.trace_rows(end=305.))
        for preset, enabled in (("saved", False), ("fast", True)):
            with self.subTest(preset=preset), \
                    patch("scripts.evaluate_competition.load_policy", return_value=policy), \
                    patch("scripts.evaluate_competition.policy_reward_kwargs", return_value={}), \
                    patch("train_zoo.replay", side_effect=replay), redirect_stdout(io.StringIO()):
                report = diagnostics.run_diagnostics(model=model, evaluation=source,
                    out_dir=directory / preset, scenarios=[1], n_agents=2, lookahead_preset=preset)
            self.assertEqual(report["lookahead_enabled"], enabled)
            self.assertEqual(report["inference_overridden"], enabled)


if __name__ == "__main__":
    unittest.main()
