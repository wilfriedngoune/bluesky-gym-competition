"""SA search is opt-in, reproducible, and routed through its copied-state adapter."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from bluesky_zoo.lookahead import LookaheadConfig
from scripts import evaluate_competition as evaluation
from scripts import replay_transfer_navigation as replay


class SingleAgentLookaheadCliTests(unittest.TestCase):
    def actor(self, **config):
        return SimpleNamespace(disable_lookahead=True, model=SimpleNamespace(gamma=.998),
                               lookahead_config=LookaheadConfig(**config))

    def test_transfer_checkpoint_retains_actor_default_and_gets_search_budget(self):
        model=SimpleNamespace(gamma=.998,competition_disable_lookahead=True,
                              competition_lookahead_config=None)
        with patch("stable_baselines3.PPO.load",return_value=model):
            act=evaluation.load_policy("sa","unused.zip",mdp="navigation-transfer")
        self.assertTrue(act.disable_lookahead)
        self.assertEqual(act.lookahead_config,LookaheadConfig())
        self.assertIs(act.model,model)

    def test_explicit_search_or_budget_enables_sa_but_actor_override_wins(self):
        for options,enabled in (({},False),({"no_lookahead":False},True),
                                ({"lookahead_preset":"fast"},True),({"no_lookahead":True},False)):
            act=self.actor()
            with self.subTest(options=options),redirect_stdout(io.StringIO()):
                evaluation.configure_lookahead(act,"sa","navigation-transfer",**options)
            self.assertEqual(not act.disable_lookahead,enabled)
        act=self.actor()
        with self.assertRaisesRegex(ValueError,"Lookahead settings require"):
            evaluation.configure_lookahead(act,"sa","navigation-transfer",no_lookahead=True,
                                           lookahead_preset="fast")

    def test_unsupported_transfer_modes_fail_before_rollout(self):
        with self.assertRaisesRegex(ValueError,"actor-only"):
            evaluation.configure_lookahead(self.actor(),"ma","navigation-transfer",no_lookahead=False)
        for options,message in (({"coordination_passes":2},"one search pass"),
                                ({"safety_filter":True},"separation filter")):
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,message):
                evaluation.configure_lookahead(self.actor(**options),"sa","navigation-transfer",no_lookahead=False)
        with self.assertRaisesRegex(ValueError,"learned RL checkpoint"):
            evaluation.configure_lookahead(lambda obs:np.zeros(2),"sa","navigation-transfer",no_lookahead=False)

    def test_single_agent_dispatch_reuses_planner_and_actor_ablation_bypasses_it(self):
        config=LookaheadConfig()
        env=SimpleNamespace()
        observation={"x":np.zeros(1)}
        act=Mock(return_value=np.array([.1,.2]))
        act.model=SimpleNamespace()
        act.lookahead_config=config
        act.disable_lookahead=False
        act.lookahead_planner=None
        act.search_observer=None
        planner=SimpleNamespace(env=env,config=config,action=Mock(return_value=np.array([.3,.4])))
        constructor=Mock(return_value=planner)
        fake_module=SimpleNamespace(SingleAgentLookaheadPolicy=constructor)
        with patch.dict("sys.modules",{"bluesky_zoo.single_agent_lookahead":fake_module}):
            for _ in range(2):
                np.testing.assert_array_equal(evaluation.action_for_single_agent(act,observation,env),[.3,.4])
        constructor.assert_called_once_with(act.model,env,config)
        self.assertEqual(planner.action.call_count,2)
        act.assert_not_called()
        act.disable_lookahead=True
        np.testing.assert_array_equal(evaluation.action_for_single_agent(act,observation,env),[.1,.2])
        act.assert_called_once_with(observation)

    def test_evaluation_records_resolved_search_and_keeps_actor_reports_separate(self):
        directory=Path(__file__).resolve().parents[1]/".cache"/"tests"/("sa_search_cli_"+uuid4().hex)
        directory.mkdir(parents=True)
        self.addCleanup(directory.rmdir)
        for disabled in (True,False):
            output=directory/("actor.csv" if disabled else "search.csv")
            settings=output.with_suffix(".settings.json")
            self.addCleanup(settings.unlink,missing_ok=True)
            act=self.actor()
            with ExitStack() as stack:
                for name,value in (("load_policy",act),("run_single_agent",[]),("policy_reward_kwargs",{}),
                                   ("summarize",None),("write_csv",None),("build_report",{}),("write_report",None)):
                    stack.enter_context(patch.object(evaluation,name,return_value=value))
                stack.enter_context(redirect_stdout(io.StringIO()))
                evaluation.evaluate("sa",20,10,"unchanged.zip",str(output),mdp="navigation-transfer",
                                    no_lookahead=disabled)
            data=json.loads(settings.read_text())
            self.assertEqual(data["lookahead_enabled"],not disabled)
            self.assertEqual(data["n_agents"],1)
            self.assertEqual(data["seed"],42)
            self.assertEqual(data["environment_kind"],"sa")
            self.assertEqual(data["lookahead_config"],None if disabled else act.lookahead_config.__dict__)

    def test_transfer_replay_cli_forwards_explicit_search_budget(self):
        with patch.object(replay,"replay") as run:
            replay.main(["--env","sa","--model","unused.zip","--with-lookahead",
                         "--lookahead-preset","fast","--lookahead-simulations","18","--lookahead-depth","4"])
        self.assertFalse(run.call_args.kwargs["no_lookahead"])
        self.assertEqual(run.call_args.kwargs["lookahead_preset"],"fast")
        self.assertEqual(run.call_args.kwargs["lookahead_simulations"],18)
        self.assertEqual(run.call_args.kwargs["lookahead_depth"],4)
