"""Replay navigation-transfer checkpoints, optionally using single-agent search."""
import argparse
import numpy as np
from scripts.evaluate_competition import (make_env,load_policy,policy_reward_kwargs,validate_policy_spaces,
                                         actions_for_agents,action_for_single_agent,configure_lookahead)
from bluesky_zoo.transfer_navigation import MDP


def replay(*,model,kind,episodes=1,seed=0,n_agents=10,headless=False,max_steps=None,
           no_lookahead=None,lookahead_preset="saved",lookahead_simulations=None,lookahead_depth=None):
    if kind not in ("sa","ma") or episodes<1 or seed<0 or max_steps is not None and max_steps<1:
        raise ValueError("Invalid replay options")
    act=load_policy(kind,model,mdp=MDP)
    configure_lookahead(act,kind,MDP,no_lookahead=no_lookahead,lookahead_preset=lookahead_preset,
                        lookahead_simulations=lookahead_simulations,lookahead_depth=lookahead_depth)
    env=make_env(kind,n_agents=n_agents,mdp=MDP,render_mode=None if headless else "human",
                 **policy_reward_kwargs(act,MDP))
    try:
        validate_policy_spaces(act,env,kind,MDP)
        for episode in range(episodes):
            obs,_=env.reset(seed=seed if episode==0 else None)
            steps=0;finished=False
            while not finished and (max_steps is None or steps<max_steps):
                if kind=="sa":
                    obs,_,done,truncated,info=env.step(action_for_single_agent(act,obs,env));finished=done or truncated
                    if finished:print(f"episode={episode} metrics={info}")
                else:
                    obs,_,done,truncated,infos=env.step(actions_for_agents(act,obs,env.agents,env=env))
                    for agent,info in infos.items():
                        if done[agent] or truncated[agent]:print(f"episode={episode} agent={agent} metrics={info}")
                    finished=not env.agents
                steps+=1
            if not finished:print(f"Diagnostic cap: episode {episode} unfinished after {steps} decisions; no final score.")
    finally:env.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env",choices=["sa","ma"],required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--episodes",type=int,default=1)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--n-agents",type=int,default=10)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--max-steps",type=int,default=None)
    inference=parser.add_mutually_exclusive_group()
    inference.add_argument("--no-lookahead",dest="no_lookahead",action="store_true",default=None)
    inference.add_argument("--with-lookahead",dest="no_lookahead",action="store_false",
                           help="enable learned-policy/value search for --env sa")
    parser.add_argument("--lookahead-preset",choices=["saved","fast"],default="saved")
    parser.add_argument("--lookahead-simulations",type=int,default=None)
    parser.add_argument("--lookahead-depth",type=int,default=None)
    args=parser.parse_args(argv)
    replay(model=args.model,kind=args.env,episodes=args.episodes,seed=args.seed,n_agents=args.n_agents,
           headless=args.headless,max_steps=args.max_steps,no_lookahead=args.no_lookahead,
           lookahead_preset=args.lookahead_preset,lookahead_simulations=args.lookahead_simulations,
           lookahead_depth=args.lookahead_depth)

if __name__=="__main__":main()
