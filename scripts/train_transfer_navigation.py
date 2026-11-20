"""Train a new shared navigation-transfer policy in the SA or MA world.

Older single-agent/v5 weights have different contracts and cannot initialize
this MDP. Once trained here, weights can transfer between hosts with matching
saved configs. No online search, deadline override or scored-world changes.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback
from stable_baselines3.common.utils import check_for_correct_spaces
from bluesky_zoo.transfer_navigation import MDP,resolve_transfer_config
from bluesky_zoo.navigation_reward import make_reward_config
from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv
from bluesky_zoo.active_rollout_buffer import ActiveAircraftCallback,ActiveAircraftRolloutBuffer
from scripts.evaluate_competition import make_env,load_policy,policy_reward_kwargs,METRIC_KEYS
from train_zoo import CompetitionMetricsCallback


class ComponentInfo(gym.Wrapper):
    """Training-only diagnostic info captured before SB3 automatically resets."""
    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        return obs,reward,terminated,truncated,dict(info,
            navigation_reward_components=self.env.unwrapped.get_reward_components())


class SingleMetrics(BaseCallback):
    def __init__(self,path,keys):
        super().__init__();self.path=Path(path);self.keys=keys;self.scenario=0;self.file=None
    def _on_training_start(self):
        self.file=self.path.open("x",newline="",encoding="utf-8")
        self.writer=csv.DictWriter(self.file,fieldnames=["scenario_index",*METRIC_KEYS,*["reward_"+k for k in self.keys]])
        self.writer.writeheader();self.file.flush()
    def _on_step(self):
        for done,info in zip(self.locals["dones"],self.locals["infos"]):
            if done:
                self.writer.writerow(dict(scenario_index=self.scenario,
                    **{k:info[k] for k in METRIC_KEYS},
                    **{"reward_"+k:info["navigation_reward_components"][k] for k in self.keys}))
                self.scenario+=1;self.file.flush()
        return True
    def _on_training_end(self):
        if self.file:self.file.close()


def train(*,kind,output,timesteps=1_000_000,seed=20,init_model=None,
          transfer_config=None,reward_config=None,n_agents=10,n_steps=1024,batch_size=256,
          n_epochs=10,checkpoint_freq=10_000,learning_rate=3e-4,dry_run=False):
    if kind not in ("sa","ma") or timesteps<1 or seed<0 or not 2<=n_agents<=10:
        raise ValueError("Use sa/ma, positive timesteps, nonnegative seed and 2..10 MA aircraft")
    if n_steps<1 or batch_size<2 or n_epochs<1 or checkpoint_freq<0 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("Invalid PPO settings")
    if n_steps*(n_agents if kind=="ma" else 1)<2:raise ValueError("PPO needs at least two rollout samples")
    base=Path(output).resolve();base=base.with_suffix("") if base.suffix==".zip" else base
    metrics=base.parent/"training_metrics.csv";checkpoints=base.parent/"checkpoints"
    if list(base.parent.glob(base.name+".*")) or metrics.exists() or checkpoints.exists() and any(checkpoints.iterdir()):
        raise FileExistsError("Choose a new experiment output directory")
    source=None
    if init_model is not None:
        source=Path(init_model).resolve()
        if not source.is_file():source=Path(str(source)+".zip")
        if not source.is_file():raise FileNotFoundError(source)
        if source==Path(str(base)+".zip"):raise ValueError("Output must differ from initialization checkpoint")
    cfg=resolve_transfer_config(transfer_config)
    plan=dict(mdp=MDP,kind=kind,output=str(base),init_model=str(source) if source else None,
              timesteps=timesteps,seed=seed,transfer_config=asdict(cfg),inference="actor-only")
    torch.set_num_threads(1)
    kwargs={}
    pretrained=None
    if source is not None:
        act=load_policy(kind,str(source),mdp=MDP);pretrained=act.model
        if getattr(pretrained,"competition_mdp",None)!=MDP:
            raise ValueError("Older observation/action contracts cannot initialize navigation-transfer; start fresh")
        kwargs=policy_reward_kwargs(act,MDP)
        saved=resolve_transfer_config(kwargs.get("transfer_config"))
        if transfer_config is not None and cfg!=saved:
            raise ValueError("Changing the shared action/memory contract requires a fresh model")
        cfg=saved
    kwargs["transfer_config"]=cfg
    if reward_config is not None:
        settings=asdict(make_reward_config("navigation-v5",saved=kwargs.get("reward_config")))
        if not isinstance(reward_config,dict) or set(reward_config)-set(settings):
            raise ValueError("Reward overrides must be an object of known reward coefficient names")
        settings.update(reward_config);kwargs["reward_config"]=settings
        make_reward_config("navigation-v5",saved=settings)
    plan["transfer_config"]=asdict(cfg)
    plan["reward_overrides"]=reward_config
    if dry_run:print(json.dumps(plan,indent=2));return plan
    raw=make_env(kind,n_agents=n_agents,mdp=MDP,**kwargs)
    vec=None;single_callback=None
    try:
        if kind=="ma":vec=SharedPolicyVecEnv(raw)
        else:
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec=DummyVecEnv([lambda:ComponentInfo(raw)])
        if pretrained is not None:check_for_correct_spaces(vec,pretrained.observation_space,pretrained.action_space)
        base.parent.mkdir(parents=True,exist_ok=True)
        callbacks=[]
        if kind=="ma":
            callbacks=[ActiveAircraftCallback(),CompetitionMetricsCallback(str(metrics),reward_component_keys=raw.reward_component_keys)]
        else:
            single_callback=SingleMetrics(metrics,raw.reward_component_keys);callbacks=[single_callback]
        model=PPO("MultiInputPolicy",vec,n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs,
                  gamma=raw.reward_config.gamma,learning_rate=learning_rate,seed=seed,device="cpu",verbose=1,
                  policy_kwargs=pretrained.policy_kwargs if pretrained is not None else dict(net_arch=dict(pi=[256,256],vf=[256,256])),
                  rollout_buffer_class=ActiveAircraftRolloutBuffer if kind=="ma" else None)
        if pretrained is not None:model.policy.load_state_dict(pretrained.policy.state_dict())
        model.competition_mdp=MDP;model.competition_environment_kind=kind
        model.competition_transfer_contract_version=1
        model.competition_disable_lookahead=True;model.competition_lookahead_config=None
        for name in ("transfer_config","reward_config","action_config","passage_config","zone_config"):
            setattr(model,"competition_"+name,asdict(getattr(raw,name)))
        model.competition_decision_seconds=raw.action_frequency
        if checkpoint_freq:
            callbacks.append(CheckpointCallback(save_freq=max(1,checkpoint_freq//vec.num_envs),
                              save_path=str(checkpoints),name_prefix=base.name))
        model.learn(total_timesteps=timesteps,callback=callbacks)
        model.save(base)
        metadata=dict(plan,actual_timesteps=model.num_timesteps,
                      active_timesteps=getattr(model,"competition_active_timesteps",model.num_timesteps),
                      timestep_unit="agent slots, including absorbing slots" if kind=="ma" else "single-aircraft decisions",
                      absorbing_timesteps=model.num_timesteps-getattr(model,"competition_active_timesteps",model.num_timesteps),
                      contract_version=1,
                      environment_n_agents=1 if kind=="sa" else n_agents,optimizer_resumed=False,
                      initialization_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
                      observation_shapes={k:list(v.shape) for k,v in model.observation_space.spaces.items()},
                      configs={name:asdict(getattr(raw,name)) for name in
                               ("transfer_config","reward_config","action_config","passage_config","zone_config")},
                      decision_seconds=raw.action_frequency,evaluated=False,metrics=str(metrics))
        Path(str(base)+".json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        print(f"Saved actor-only {MDP} model: {base}.zip; separate evaluation is required.")
        return metadata
    finally:
        if single_callback and single_callback.file and not single_callback.file.closed:single_callback.file.close()
        if vec is not None:vec.close()
        else:raw.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env",choices=["sa","ma"],required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--init-model")
    parser.add_argument("--timesteps",type=int,default=1_000_000)
    parser.add_argument("--seed",type=int,default=20)
    parser.add_argument("--n-agents",type=int,default=10)
    parser.add_argument("--n-steps",type=int,default=1024)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--n-epochs",type=int,default=10)
    parser.add_argument("--checkpoint-freq",type=int,default=10_000)
    parser.add_argument("--learning-rate",type=float,default=3e-4)
    parser.add_argument("--transfer-config",help="JSON TransferConfig; fresh models only when changed")
    parser.add_argument("--reward-config",help="JSON reward coefficient overrides, logged separately")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    try:
        train(kind=args.env,output=args.model,timesteps=args.timesteps,seed=args.seed,init_model=args.init_model,
              transfer_config=json.loads(Path(args.transfer_config).read_text()) if args.transfer_config else None,
              reward_config=json.loads(Path(args.reward_config).read_text()) if args.reward_config else None,
              n_agents=args.n_agents,n_steps=args.n_steps,batch_size=args.batch_size,n_epochs=args.n_epochs,
              checkpoint_freq=args.checkpoint_freq,learning_rate=args.learning_rate,dry_run=args.dry_run)
    except (ValueError,FileNotFoundError,FileExistsError) as error:parser.error(str(error))

if __name__=="__main__":main()
