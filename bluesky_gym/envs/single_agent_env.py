"""Continuous navigation policy MDP for the single-agent competition."""

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_gym.envs.common.navigation import NavigationMDP, M_PER_NM, MPS_TO_KT
from bluesky_gym.envs.common.command_filter import command_config


class SingleAgentEnv(NavigationMDP, CompetitionEnv):
    def __init__(self, action_config=None, **kwargs):
        kwargs.setdefault("action_frequency", 5)
        super().__init__(**kwargs)
        self._init_navigation(self.n_obstacles)
        self.action_config = command_config(action_config)

    def reset(self, seed=None, options=None):
        self._reset_navigation()
        observations, info = super().reset(seed=seed, options=options)
        self._initialize_goal_distance(self.agent)
        return observations, info
