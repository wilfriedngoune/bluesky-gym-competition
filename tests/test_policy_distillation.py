"""Actor imitation must preserve critics, teacher weights and live-row semantics."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts import distill_navigation_policy as distillation
import test_lookahead_pipeline as fixtures


class PolicyDistillationTests(unittest.TestCase):
    def make_directory(self):
        directory = Path(__file__).resolve().parents[1] / ".cache" / "tests" / ("distill_"+uuid4().hex)
        directory.mkdir(parents=True)
        self.addCleanup(directory.rmdir)
        return directory

    def test_live_examples_exclude_inactive_slots_and_copy_inputs(self):
        examples = distillation.LiveExamples(capacity=2, seed=2)
        observations = {"live": {"value": np.array([1.,2.])}, "padding": {"value": np.zeros(2)}}
        target = np.array([.2,.3], dtype=np.float32)
        examples.add(observations, ["live"], {"live":target})
        observations["live"]["value"][0] = 99
        target[0] = -1
        x,y = examples.arrays()
        np.testing.assert_array_equal(x["value"], [[1.,2.]])
        np.testing.assert_allclose(y, [[.2,.3]])
        for index in range(20):
            examples.add({"live":{"value":np.array([index,1.])}}, ["live"], {"live":[0.,0.]})
        self.assertEqual(examples.seen,21)
        self.assertEqual(len(examples.rows),2)

    def test_actor_loss_decreases_without_changing_critic_variance_or_other_model(self):
        env,model = fixtures.LookaheadPipelineTests.make_model(self,n_agents=2)
        observations,_ = env.reset(seed=30)
        untouched = deepcopy(model.policy.state_dict())
        examples = distillation.LiveExamples(8)
        for _ in range(4):
            examples.add(observations,env.agents,{agent:np.array([.55,-.6]) for agent in env.agents})
        result=distillation.fit_actor(model,examples,epochs=30,batch_size=4,learning_rate=0.01,seed=30)
        self.assertLess(result["final_loss"],result["initial_loss"]*.2)
        after=model.policy.state_dict()
        actor_names=("mlp_extractor.policy_net.","action_net.")
        self.assertTrue(any(not torch.equal(untouched[key],after[key]) for key in untouched
                            if key.startswith(actor_names)))
        for key,value in untouched.items():
            if not key.startswith(actor_names):
                self.assertTrue(torch.equal(value,after[key]),key)

    def test_heading_loss_is_periodic_but_discourages_out_of_bounds_means(self):
        target=torch.tensor([[-.99,0.]])
        close=distillation.imitation_loss(torch.tensor([[.99,0.]]),target)
        self.assertLess(float(close),.001)
        outside=distillation.imitation_loss(torch.tensor([[1.01,0.]]),target)
        self.assertGreater(float(outside),0.)
        self.assertGreater(float(distillation.imitation_loss(torch.tensor([[3.01,0.]]),target)),1.)

    def test_collection_queries_live_agents_and_uses_student_teacher_mixture(self):
        class Env:
            agents=["a","b"]
            def step(self,actions):
                self.last_actions=actions
                departed=list(self.agents)
                self.agents=["b"]
                return {"b":{"value":np.array([2.])}}, {}, {a:a=="a" for a in departed}, \
                    {a:False for a in departed}, {a:{"waypoint_reached":int(a=="a")} for a in departed}
        env=Env()
        examples=distillation.LiveExamples(8)
        obs={"a":{"value":np.array([1.])},"b":{"value":np.array([2.])},"padding":{"value":np.array([0.])}}
        teacher=object()
        queried=[]
        def actions(act,observations,agents,env=None):
            queried.append(tuple(agents))
            return {a:np.array([.5,0.]) if act is teacher else np.array([-.5,0.]) for a in agents}
        with patch("scripts.evaluate_competition.actions_for_agents",side_effect=actions):
            _,counts=distillation.collect_examples(env,obs,teacher,object(),examples,decisions=2,
                                                  teacher_mix=0.,rng=np.random.default_rng(20))
        self.assertEqual(queried,[("a","b"),("a","b"),("b",),("b",)])
        self.assertEqual(examples.seen,3)
        self.assertEqual(counts["teacher_actions"],0)
        self.assertEqual(counts["student_actions"],3)
        np.testing.assert_array_equal(examples.arrays()[1],[[.5,0.],[.5,0.],[.5,0.]])

    def test_output_and_dry_run_never_overwrite_teacher_or_existing_artifacts(self):
        directory=self.make_directory()
        source=directory/"teacher.zip";source.write_bytes(b"teacher")
        self.addCleanup(source.unlink)
        with self.assertRaises(ValueError):distillation.output_paths(source,source)
        result=distillation.distill(source,directory/"student",dry_run=True)
        self.assertFalse((directory/"student.zip").exists())
        self.assertTrue(result["actor_only_student"])
        existing=directory/"student.distillation.npz";existing.write_bytes(b"data")
        self.addCleanup(existing.unlink)
        with self.assertRaises(FileExistsError):distillation.output_paths(source,directory/"student")
        self.assertEqual(source.read_bytes(),b"teacher")

    def test_small_real_distillation_saves_loadable_actor_only_student_and_fixed_teacher(self):
        env,model=fixtures.LookaheadPipelineTests.make_model(self,n_agents=2)
        # Close this live simulator before the distillation factory opens another.
        env.close()
        directory=self.make_directory()
        source=directory/"teacher.zip";model.save(source)
        self.addCleanup(source.unlink)
        before=source.read_bytes()
        output=directory/"student"
        for suffix in (".zip",".distillation.json",".distillation.npz"):
            self.addCleanup(Path(str(output)+suffix).unlink,missing_ok=True)
        config=distillation.DistillationConfig(n_agents=2,iterations=1,decisions_per_iteration=1,
                                               epochs=2,batch_size=2,max_examples=4)
        from scripts import evaluate_competition as evaluation
        loader=evaluation.load_policy
        teacher_snapshot={}
        def capture_teacher(*args,**kwargs):
            teacher=loader(*args,**kwargs)
            teacher_snapshot["model"]=teacher.model
            teacher_snapshot["weights"]=deepcopy(teacher.model.policy.state_dict())
            return teacher
        with patch.object(evaluation,"load_policy",side_effect=capture_teacher):
            metadata=distillation.distill(source,output,config=config,lookahead_simulations=1,lookahead_depth=1)
        self.assertEqual(source.read_bytes(),before)
        for name,value in teacher_snapshot["weights"].items():
            self.assertTrue(torch.equal(value,teacher_snapshot["model"].policy.state_dict()[name]),name)
        self.assertEqual(metadata["retained_examples"],2)
        student=PPO.load(str(output)+".zip",device="cpu")
        self.assertTrue(student.competition_disable_lookahead)
        for name,value in model.policy.state_dict().items():
            if not name.startswith(("mlp_extractor.policy_net.","action_net.")):
                self.assertTrue(torch.equal(value,student.policy.state_dict()[name]),name)
        with np.load(str(output)+".distillation.npz",allow_pickle=False) as dataset:
            self.assertEqual(dataset["actions"].shape,(2,2))
            self.assertTrue(any(key.startswith("obs__") for key in dataset.files))

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (dict(iterations=0),dict(max_examples=0),dict(n_agents=1),
                       dict(seed=-1),dict(teacher_mix=1.1),dict(learning_rate=float("nan"))):
            with self.assertRaises(ValueError):distillation.DistillationConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
