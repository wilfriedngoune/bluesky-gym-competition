"""The responsive training recipe preserves reference models and saved options."""

from pathlib import Path
import unittest
from uuid import uuid4

from scripts.train_responsive_v5 import training_command


class ResponsiveTrainingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("responsive_" + uuid4().hex)
        baseline = self.root / "models/ma_navigation_v5_separation/model.zip"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(b"baseline fixture")
        self.baseline = baseline
        self.addCleanup(baseline.unlink, missing_ok=True)

    def value(self, command, flag):
        return command[command.index(flag) + 1]

    def test_continuation_uses_responsive_controls_and_separate_output(self):
        command = training_command(root=self.root)
        for flag, value in (("--decision-seconds", "5"), ("--heading-deadband-deg", "1"),
                            ("--speed-deadband-kt", "1"), ("--aircraft-buffer-nm", "8"),
                            ("--aircraft-buffer-cost", "0.05"), ("--n-agents", "10")):
            self.assertEqual(self.value(command, flag), value)
        self.assertEqual(Path(self.value(command, "--init-model")), self.baseline)
        self.assertNotEqual(Path(self.value(command, "--model")), self.baseline.with_suffix(""))
        self.assertNotIn("--curriculum", command)
        self.assertEqual(self.baseline.read_bytes(), b"baseline fixture")

    def test_fresh_training_has_curriculum_and_no_checkpoint_dependency(self):
        self.baseline.unlink()
        command = training_command(fresh=True, root=self.root)
        self.assertNotIn("--init-model", command)
        self.assertEqual(command[command.index("--curriculum") + 1:], ["2", "4", "6", "10"])

    def test_existing_metrics_or_model_are_preserved(self):
        model = self.root / "new/model"
        model.parent.mkdir(parents=True)
        for path in (model.with_suffix(".zip"), model.with_suffix(".json"), model.parent / "training_metrics.csv"):
            path.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                training_command(model=model, root=self.root)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep")
            path.unlink()

    def test_invalid_source_and_budget_fail_before_training(self):
        for options in (dict(timesteps=0), dict(fresh=True, timesteps=3), dict(eval_episodes=-1),
                        dict(fresh=True, init_model="old.zip"), dict(seed=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                training_command(root=self.root, **options)
        with self.assertRaises(FileNotFoundError):
            training_command(root=self.root, init_model="missing.zip")


if __name__ == "__main__":
    unittest.main()
