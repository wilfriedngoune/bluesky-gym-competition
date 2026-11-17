"""Inference budget overrides are explicit, recorded, and leave saved profiles intact."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from bluesky_zoo.lookahead import LookaheadConfig
from scripts import evaluate_competition as evaluation
import train_zoo


class LookaheadBudgetCliTests(unittest.TestCase):
    def test_evaluation_cli_forwards_preset_and_explicit_overrides(self):
        arguments = ["evaluate_competition", "--env", "ma", "--mdp", "navigation-v5",
                     "--model", "unused.zip", "--episodes", "20", "--lookahead-preset", "fast",
                     "--lookahead-simulations", "18", "--lookahead-depth", "4"]
        with patch("sys.argv", arguments), patch.object(evaluation, "evaluate", return_value={}) as run, redirect_stdout(io.StringIO()):
            evaluation.main()
        self.assertEqual(run.call_args.kwargs["lookahead_preset"], "fast")
        self.assertEqual(run.call_args.kwargs["lookahead_simulations"], 18)
        self.assertEqual(run.call_args.kwargs["lookahead_depth"], 4)

    def test_replay_cli_forwards_preset_and_explicit_overrides(self):
        with patch.object(train_zoo, "replay") as run:
            train_zoo.main(["replay", "unused.zip", "--mdp", "navigation-v5", "--lookahead-preset", "fast",
                            "--lookahead-simulations", "18", "--lookahead-depth", "4"])
        self.assertEqual(run.call_args.kwargs["lookahead_preset"], "fast")
        self.assertEqual(run.call_args.kwargs["lookahead_simulations"], 18)
        self.assertEqual(run.call_args.kwargs["lookahead_depth"], 4)

    def test_resolved_evaluation_settings_record_budget_and_preserve_checkpoint_config(self):
        saved = LookaheadConfig(simulations=48, max_depth=8, coordination_passes=2, safety_filter=True)
        act = SimpleNamespace(lookahead_config=saved)
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("budget_" + uuid4().hex)
        directory.mkdir(parents=True)
        settings_path = directory / "evaluation.settings.json"
        self.addCleanup(directory.rmdir)
        self.addCleanup(settings_path.unlink, missing_ok=True)
        with ExitStack() as stack:
            for name, value in (("load_policy", act), ("run_multi_agent", []), ("policy_reward_kwargs", {}),
                                ("summarize", None), ("write_csv", None), ("build_report", {}), ("write_report", None)):
                stack.enter_context(patch.object(evaluation, name, return_value=value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            evaluation.evaluate("ma", 20, 10, "unused.zip", str(directory / "evaluation.csv"),
                                mdp="navigation-v5", lookahead_preset="fast",
                                lookahead_simulations=18, lookahead_depth=4)
        settings = json.loads(settings_path.read_text())
        self.assertEqual(settings["lookahead_preset"], "fast")
        self.assertEqual(settings["lookahead_config"]["simulations"], 18)
        self.assertEqual(settings["lookahead_config"]["max_depth"], 4)
        self.assertEqual(settings["lookahead_config"]["coordination_passes"], 1)
        self.assertTrue(settings["lookahead_config"]["safety_filter"])
        self.assertEqual(asdict(saved), asdict(LookaheadConfig(simulations=48,max_depth=8,coordination_passes=2,safety_filter=True)))

    def test_disabled_or_non_v5_evaluation_rejects_budget_overrides(self):
        for options in (dict(mdp="base", lookahead_preset="fast"),
                        dict(mdp="navigation-v5", no_lookahead=True, lookahead_preset="fast"),
                        dict(mdp="navigation-v5", no_lookahead=True, lookahead_simulations=12)):
            with self.subTest(options=options), patch.object(evaluation,"load_policy",return_value=SimpleNamespace()), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"Lookahead settings require"):
                    evaluation.evaluate("ma",20,10,"unused.zip","unused.csv",**options)

    def test_actor_only_replay_rejects_budget_override(self):
        act = SimpleNamespace(model=SimpleNamespace())
        with patch.object(evaluation,"load_policy",return_value=act), patch.object(train_zoo.torch,"set_num_threads"):
            with self.assertRaisesRegex(ValueError,"Lookahead settings require lookahead enabled"):
                train_zoo.replay("unused.zip",mdp="navigation-v5",disable_lookahead=True,lookahead_preset="fast")
