"""Opt-in shared SA/MA navigation contract; old MDPs/checkpoints stay unchanged.

Both hosts inherit their original scenario generation, physics, step, metrics
and termination. Training and default inference use the learned actor alone.
Single-agent inference can explicitly opt into ``single_agent_lookahead``, which
forecasts this contract's progress/avoidance memory. The v5 MA tree predictor
does not model that memory. No hard-deadline override is installed.
"""
from dataclasses import asdict, dataclass
import math
import bluesky as bs
import numpy as np
from gymnasium import spaces

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.navigation_reward import make_reward_config
from bluesky_zoo.passage_navigation import MultiAgentPassageEnv, resolve_passage_config
from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv, _StrategicNavigation, resolve_zone_config
from bluesky_gym.envs.common.command_filter import command_config
from bluesky_gym.envs.common.decision_timing import validate_decision_seconds
from core.tools import kwikqdrdist

MDP="navigation-transfer"

@dataclass(frozen=True)
class TransferConfig:
    min_target_speed_kt: float = 120.
    max_target_speed_kt: float = 220.
    side_commitment: bool = True
    stagnant_rate_nm_s: float = .002
    side_switch_after_s: float = 150.

    def __post_init__(self):
        for name in ("min_target_speed_kt","max_target_speed_kt","stagnant_rate_nm_s","side_switch_after_s"):
            value=getattr(self,name)
            if not math.isfinite(value) or value < 0:raise ValueError(f"Invalid {name}")
        if not 0 < self.min_target_speed_kt < self.max_target_speed_kt:
            raise ValueError("Target speeds must be positive and increasing")
        if self.side_switch_after_s <= 0 or not isinstance(self.side_commitment,bool):
            raise ValueError("Use a positive switch interval and boolean side_commitment")


def resolve_transfer_config(saved=None):
    if saved is not None and not isinstance(saved,(dict,TransferConfig)):
        raise ValueError("Transfer configuration must be a JSON object")
    if isinstance(saved,dict) and set(saved)-set(TransferConfig.__dataclass_fields__):
        raise ValueError("Unknown transfer configuration fields")
    return saved if isinstance(saved,TransferConfig) else TransferConfig(**(saved or {}))


def deadline_features(distance_nm, capture_radius_km, max_speed_kt, elapsed_s, limit_s):
    """Optimistic straight-line capture time, not a route/turn-aware arrival guarantee."""
    if not all(math.isfinite(x) for x in (distance_nm,capture_radius_km,max_speed_kt,elapsed_s,limit_s)):
        raise ValueError("Deadline inputs must be finite")
    if distance_nm<0 or capture_radius_km<0 or max_speed_kt<=0 or elapsed_s<0 or limit_s<=0:
        raise ValueError("Invalid deadline inputs")
    estimate=max(0.,distance_nm-capture_radius_km/1.852)*3600./max_speed_kt
    remaining=max(0.,limit_s-elapsed_s)
    return (np.array([np.clip(estimate/limit_s,0.,2.)]),
            np.array([np.clip((remaining-estimate)/limit_s,-1.,1.)]))


class _TransferNavigation(_StrategicNavigation):
    def __init__(self,env,ac_id):
        super().__init__(env,ac_id)
        self.min_target_speed=env.transfer_config.min_target_speed_kt
        self.max_target_speed=env.transfer_config.max_target_speed_kt
        self.n_tracked_intruders=10
        self._clear_transfer_memory()
        self.observation_space=spaces.Dict({**self.observation_space.spaces,
            "intruders":spaces.Box(-3.,3.,shape=(10,7),dtype=np.float64),
            "intruder_mask":spaces.Box(0.,1.,shape=(10,),dtype=np.float64),
            "time_to_goal":spaces.Box(0.,2.,shape=(1,),dtype=np.float64),
            "time_slack":spaces.Box(-1.,1.,shape=(1,),dtype=np.float64),
            "navigation_memory":spaces.Box(np.array([-1.,0.,-3.,0.]),np.array([1.,1.,3.,1.]),dtype=np.float64)})

    def _clear_transfer_memory(self):
        self.intruder_mask=np.zeros(10)
        self.progress_anchor_time=None
        self.progress_anchor_distance=None
        self.recent_progress_rate=0.
        self.stalled_seconds=0.
        self.avoidance_obstacle=None
        self.avoidance_side=0
        self.avoidance_started=0.
        self.avoidance_switched=False

    def _reset_navigation(self):
        super()._reset_navigation()
        self._clear_transfer_memory()

    def _clock(self):
        return float(self.metrics[self.ac_id]["flight_time"])

    def _refresh_progress(self):
        now=self._clock();distance=self._goal_distance_nm()
        if self.progress_anchor_time is not None and now>self.progress_anchor_time:
            elapsed=now-self.progress_anchor_time
            self.recent_progress_rate=(self.progress_anchor_distance-distance)/elapsed
            if self.recent_progress_rate < self.env.transfer_config.stagnant_rate_nm_s:
                self.stalled_seconds+=elapsed
            else:
                self.stalled_seconds=0.
                self.avoidance_switched=False
        if self.progress_anchor_time is None or now>self.progress_anchor_time:
            self.progress_anchor_time,self.progress_anchor_distance=now,distance

    def _get_intruder_features(self,own_idx):
        candidates=self._get_intruder_candidates(own_idx)
        candidates.sort(key=lambda item:(item[1][4]*100.>=self.env.intrusion_distance,item[0]))
        features=np.zeros((10,7));self.intruder_mask.fill(0.)
        for slot,(_,feature) in enumerate(candidates[:10]):
            features[slot]=feature;self.intruder_mask[slot]=1.
        return features

    def _get_buffer_avoidance_heading(self,own_idx,obstacle_state):
        escape=super()._get_buffer_avoidance_heading(own_idx,obstacle_state)
        cfg=self.env.transfer_config
        if not cfg.side_commitment or obstacle_state["inside"]:return escape
        goal,_=kwikqdrdist(bs.traf.lat[own_idx],bs.traf.lon[own_idx],self.goal_lat,self.goal_lon)
        left,right=escape-90.,escape+90.
        angle_error=lambda a:abs((a-goal+180.)%360.-180.)
        if self.avoidance_obstacle!=obstacle_state["index"]:
            self.avoidance_obstacle=obstacle_state["index"]
            self.avoidance_side=-1 if angle_error(left)<=angle_error(right) else 1
            self.avoidance_started=self._clock();self.avoidance_switched=False
        if (min(self.stalled_seconds,self._clock()-self.avoidance_started)>=cfg.side_switch_after_s
                and not self.avoidance_switched):
            self.avoidance_side*=-1;self.avoidance_switched=True
        tangent=left if self.avoidance_side<0 else right
        # A reference direction only: the RL heading offset retains full freedom.
        directions=np.radians([goal,tangent,escape]);weights=np.array([.15,.6,.25])
        north=float(np.dot(weights,np.cos(directions)));east=float(np.dot(weights,np.sin(directions)))
        return math.degrees(math.atan2(east,north)) if math.hypot(north,east)>1e-12 else escape

    def _get_obs(self,ac_id):
        self._refresh_progress()
        obs=super()._get_obs(ac_id)
        if self.active_buffer_obstacle is None:
            self.avoidance_obstacle=None;self.avoidance_side=0;self.avoidance_switched=False
        estimate,slack=deadline_features(self._goal_distance_nm(),self.env.distance_margin,
            self.max_target_speed,self._clock(),self.env.episode_time_limit)
        obs.update(time_to_goal=estimate,time_slack=slack,navigation_memory=np.array([
            float(self.avoidance_side),np.clip(self.stalled_seconds/self.env.episode_time_limit,0.,1.),
            np.clip(self.recent_progress_rate*3600./self.max_target_speed,-3.,3.),float(self.avoidance_switched)]))
        return obs


class MultiAgentTransferEnv(MultiAgentStrategicEnv):
    navigation_class=_TransferNavigation
    metadata={**MultiAgentStrategicEnv.metadata,"name":"navigation_transfer"}

    def __init__(self,transfer_config=None,**kwargs):
        self.transfer_config=resolve_transfer_config(transfer_config)
        kwargs["action_frequency"]=validate_decision_seconds(kwargs.get("action_frequency",5))
        super().__init__(**kwargs)


class _SingleTransferNavigation(_TransferNavigation):
    @property
    def goal_lat(self):return self.env.goal_lat
    @property
    def goal_lon(self):return self.env.goal_lon
    @property
    def intruder_ids(self):return list(self.env.intruder_ids)
    @property
    def _reward_baseline(self):return self.env._reward_baseline


class SingleAgentTransferEnv(CompetitionEnv):
    """Same shared contract as MA, with the original ten scripted intruders."""
    reward_component_keys=MultiAgentStrategicEnv.reward_component_keys
    polygon_edges=MultiAgentPassageEnv.polygon_edges

    def __init__(self,transfer_config=None,reward_config=None,action_config=None,
                 passage_config=None,zone_config=None,**kwargs):
        count=kwargs.get("n_intruders",10)
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=10:
            raise ValueError("The shared contract supports up to ten scripted intruders")
        self.transfer_config=resolve_transfer_config(transfer_config)
        self.action_config=command_config(action_config)
        self.passage_config=resolve_passage_config(passage_config)
        self.zone_config=resolve_zone_config(zone_config)
        self.reward_config=make_reward_config("navigation-v5",saved=reward_config)
        self._edge_scenario=None;self._polygon_edges=None
        kwargs["action_frequency"]=validate_decision_seconds(kwargs.get("action_frequency",5))
        super().__init__(**kwargs)
        self.navigation_state=_SingleTransferNavigation(self,self.agent)
        self.observation_space=self.navigation_state.observation_space
        self.action_space=self.navigation_state.action_space

    @property
    def sim_time(self):return float(self.metrics.get(self.agent,{}).get("flight_time",0.))

    def reset(self,seed=None,options=None):
        self.navigation_state._reset_navigation()
        self._edge_scenario=None;self._polygon_edges=None
        return super().reset(seed=seed,options=options)

    def _get_obs(self,ac_id):
        state=self.navigation_state;state.ac_id=ac_id
        if state.previous_goal_distance is None:state._initialize_goal_distance(ac_id)
        return state._get_obs(ac_id)

    def _get_action(self,ac_id,action):self.navigation_state._get_action(ac_id,action)
    def _get_reward(self,ac_id):return self.navigation_state._get_reward(ac_id)
    def get_reward_components(self,ac_id=None):return dict(self.navigation_state.reward_component_totals)
    def get_reward_config(self):return asdict(self.reward_config)
