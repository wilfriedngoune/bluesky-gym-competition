"""Held-out comparisons keep matched seeds, saved inference modes and raw metrics."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from bluesky_zoo.lookahead import LookaheadConfig
from scripts import validate_navigation_checkpoints as validation


def record(*, reached=1, occupancy=0.):
    return dict(waypoint_reached=reached, flight_time=3000., intrusion_events=0,
                intrusion_time=occupancy, restricted_area_events=0, time_in_restricted_area=0.,
                sector_exit_events=0, time_outside_sector=0., total_reward=-2.)


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        self.directory = self.base / ("checkpoint_validation_" + uuid4().hex)
        self.directory.mkdir(parents=True)
        def cleanup():
            resolved = self.directory.resolve()
            if not resolved.is_relative_to(self.base.resolve()):
                raise RuntimeError("Refusing to clean outside the test directory")
            shutil.rmtree(resolved)
        self.addCleanup(cleanup)
        self.models = [self.directory / "one.zip", self.directory / "two.zip"]
        for index, model in enumerate(self.models):
            model.write_bytes(f"model{index}".encode())

    def test_matched_seeds_saved_modes_raw_metrics_and_nonofficial_summary(self):
        actors = []
        calls = []
        def load(kind, model, mdp):
            act = SimpleNamespace(disable_lookahead=model.endswith("one.zip"),
                                  lookahead_config=LookaheadConfig())
            actors.append(act)
            return act
        def run(episodes, act, n_agents, **kwargs):
            calls.append((episodes, n_agents, kwargs["seed"], act.disable_lookahead))
            return [record(), record(reached=0)]
        output = self.directory / "validation"
        with patch.object(validation.evaluation, "load_policy", side_effect=load), \
                patch.object(validation.evaluation, "run_multi_agent", side_effect=run), \
                patch.object(validation.evaluation, "policy_reward_kwargs", return_value={"decision_seconds":5}), \
                redirect_stdout(io.StringIO()):
            summary = validation.validate_checkpoints(self.models, out_dir=output,
                seeds=[101,102], episodes=1, n_agents=2, progress=False)
        self.assertEqual(calls, [(1,2,101,True),(1,2,102,True),(1,2,101,False),(1,2,102,False)])
        self.assertTrue(summary["completed"])
        self.assertEqual(summary["phase"], "validation")
        self.assertFalse(summary["official_protocol"])
        self.assertEqual(summary["inference_override"], "saved")
        self.assertEqual(len(list(output.glob("*.csv"))),4)
        self.assertTrue((output / "summary.md").is_file())
        for item in summary["models"]:
            self.assertEqual(len(item["model_sha256"]),64)
            self.assertEqual(item["aggregate"]["aircraft_records"],4)
            self.assertEqual(item["aggregate"]["clean_arrivals"],2)
            self.assertEqual(item["aggregate"]["clean_arrival_pct"],50.)
            self.assertEqual(item["aggregate"]["totals"]["total_reward"],-8.)
        persisted = json.loads((output / "summary.json").read_text())
        self.assertEqual(persisted["models"][0]["lookahead_enabled"],False)
        self.assertEqual(persisted["models"][1]["lookahead_enabled"],True)
        self.assertEqual([model.read_bytes() for model in self.models],[b"model0",b"model1"])

    def test_occupancy_without_event_is_not_clean_and_reward_is_not_a_score(self):
        metrics = validation.aggregate_metrics([record(),record(occupancy=.5),record(reached=0)])
        self.assertEqual(metrics["clean_flights"],2)
        self.assertEqual(metrics["clean_arrivals"],1)
        self.assertAlmostEqual(metrics["clean_arrival_pct"],100/3)
        self.assertEqual(metrics["totals"]["intrusion_events"],0.)
        self.assertEqual(metrics["totals"]["intrusion_time"],.5)
        self.assertEqual(metrics["totals"]["total_reward"],-6.)

    def test_explicit_actor_only_override_is_used(self):
        act = SimpleNamespace(disable_lookahead=False,lookahead_config=LookaheadConfig())
        def run(*args,**kwargs):
            self.assertTrue(args[1].disable_lookahead)
            return [record(),record()]
        with patch.object(validation.evaluation,"load_policy",return_value=act), \
                patch.object(validation.evaluation,"run_multi_agent",side_effect=run), \
                patch.object(validation.evaluation,"policy_reward_kwargs",return_value={}), \
                redirect_stdout(io.StringIO()):
            result = validation.validate_checkpoints(self.models[:1],out_dir=self.directory/"actor",
                seeds=[42],episodes=1,n_agents=2,disable_lookahead=True,progress=False)
        self.assertFalse(result["official_protocol"])
        self.assertEqual(result["inference_override"],"actor_only")

    def test_existing_output_refused_before_loading_models(self):
        output=self.directory/"existing"
        output.mkdir()
        sentinel=output/"keep.txt"
        sentinel.write_text("preserve")
        with patch.object(validation.evaluation,"load_policy") as load:
            with self.assertRaises(FileExistsError):
                validation.validate_checkpoints(self.models,out_dir=output)
        load.assert_not_called()
        self.assertEqual(sentinel.read_text(),"preserve")

    def test_incomplete_rollout_keeps_incomplete_manifest(self):
        act=SimpleNamespace(disable_lookahead=False,lookahead_config=LookaheadConfig())
        output=self.directory/"incomplete"
        with patch.object(validation.evaluation,"load_policy",return_value=act), \
                patch.object(validation.evaluation,"run_multi_agent",return_value=[record()]), \
                patch.object(validation.evaluation,"policy_reward_kwargs",return_value={}):
            with self.assertRaisesRegex(RuntimeError,"one final record"):
                validation.validate_checkpoints(self.models[:1],out_dir=output,
                    episodes=1,n_agents=2,progress=False)
        self.assertFalse(json.loads((output/"summary.json").read_text())["completed"])
        self.assertFalse((output/"summary.md").exists())

    def test_cli_default_and_explicit_search_modes(self):
        for flags, expected in (([],None),(["--no-lookahead"],True),(["--with-lookahead"],False)):
            with patch.object(validation,"validate_checkpoints",return_value={}) as run:
                validation.main(["--models",str(self.models[0]),"--out-dir","unused",*flags])
            self.assertEqual(run.call_args.kwargs["disable_lookahead"],expected)
            self.assertEqual(run.call_args.kwargs["seeds"],[101,102])


if __name__ == "__main__":
    unittest.main()
