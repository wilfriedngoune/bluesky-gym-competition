from pathlib import Path
import tempfile
import unittest
from scripts.train_navigation_v6 import training_command

class NavigationV6LauncherTests(unittest.TestCase):
    def test_training_is_only_ten_aircraft_and_keeps_saved_mdp_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"source.zip";source.write_bytes(b"fixture")
            command=training_command(root=root,init_model="source.zip",model="new/model",targeted_probability=.25)
            self.assertNotIn("--curriculum",command)
            self.assertEqual(command[command.index("--n-agents")+1],"10")
            self.assertEqual(command[command.index("--passage-training-profile")+1],"mixed")
            self.assertEqual(command[command.index("--eval-episodes")+1],"0")
            self.assertNotIn("--decision-seconds",command)
            self.assertNotIn("--aircraft-buffer-cost",command)
            self.assertEqual(source.read_bytes(),b"fixture")
            self.assertFalse((root/"new").exists())

    def test_refuses_existing_artifacts_and_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"source.zip";source.write_bytes(b"fixture")
            with self.assertRaises(ValueError):training_command(root=root,init_model="source.zip",model="source")
            (root/"new").mkdir();(root/"new/model.evaluation.csv").write_text("preserve")
            with self.assertRaises(FileExistsError):training_command(root=root,init_model="source.zip",model="new/model")

    def test_invalid_training_options(self):
        for options in ({"timesteps":0},{"targeted_probability":float("nan")},{"targeted_probability":1.1},
                        {"targeted_profile":"bad"},{"learning_rate":0},{"n_steps":0},{"batch_size":1}):
            with self.subTest(options=options),self.assertRaises(ValueError):training_command(**options)
