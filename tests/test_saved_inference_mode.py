from contextlib import ExitStack,redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from bluesky_zoo.lookahead import LookaheadConfig
from scripts import evaluate_competition as evaluation
from test_competition_pipeline import EndingParallelEnv
import train_zoo

class SavedInferenceModeTests(unittest.TestCase):
    def test_loaded_checkpoint_sets_actor_only_default(self):
        model=SimpleNamespace(competition_disable_lookahead=True,gamma=.998)
        with patch("stable_baselines3.PPO.load",return_value=model):
            act=evaluation.load_policy("ma","unused.zip",mdp="navigation-v5")
        self.assertTrue(act.disable_lookahead)
        self.assertIs(act.model,model)

    def test_evaluation_preserves_default_but_explicit_search_can_enable(self):
        with tempfile.TemporaryDirectory() as directory:
            for index,(override,preset,expected) in enumerate(((None,"saved",True),(False,"saved",False),
                                                            (True,"saved",True),(None,"fast",False))):
                act=SimpleNamespace(disable_lookahead=True,lookahead_config=LookaheadConfig())
                with self.subTest(override=override,preset=preset),ExitStack() as stack:
                    for name,value in (("load_policy",act),("run_multi_agent",[]),("policy_reward_kwargs",{}),
                                       ("summarize",None),("write_csv",None),("build_report",{}),("write_report",None)):
                        stack.enter_context(patch.object(evaluation,name,return_value=value))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    evaluation.evaluate("ma",1,10,"unused.zip",str(Path(directory)/f"result{index}.csv"),
                                        mdp="navigation-v5",no_lookahead=override,lookahead_preset=preset)
                self.assertEqual(act.disable_lookahead,expected)

    def test_replay_cli_distinguishes_default_enable_and_disable(self):
        for flags,expected in (([],None),(["--with-lookahead"],False),(["--no-lookahead"],True)):
            with self.subTest(flags=flags),patch.object(train_zoo,"replay") as replay:
                train_zoo.main(["replay","unused.zip","--mdp","navigation-v5",*flags])
            self.assertIs(replay.call_args.kwargs["disable_lookahead"],expected)

    def test_validation_seed_is_set_on_first_reset_only(self):
        env=EndingParallelEnv()
        with patch.object(evaluation,"make_env",return_value=env),patch.object(evaluation,"validate_policy_spaces"):
            records=evaluation.run_multi_agent(2,lambda observation:np.zeros(2),2,seed=101)
        self.assertEqual(env.seeds,[101,None])
        self.assertEqual(len(records),4)

    def test_gif_recording_preserves_saved_mode(self):
        from scripts import record_competition_gifs as recorder
        for override,expected in ((None,True),(False,False),(True,True)):
            act=SimpleNamespace(disable_lookahead=True)
            with self.subTest(override=override),patch.object(recorder,"policy_from_model",return_value=act), \
                    patch.object(recorder,"rollout_multi_agent",return_value=[]),patch.object(recorder,"save_gif"):
                recorder.record("ma",[101],"unused.gif",mdp="navigation-v5",no_lookahead=override)
            self.assertEqual(act.disable_lookahead,expected)
