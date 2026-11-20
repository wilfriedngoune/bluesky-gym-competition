"""The existing navigation MDP applied independently to each controlled aircraft.

``MultiAgentEnv`` retains CompetitionZooEnv's scenarios, simulation, termination
and objective scoring. All policy observations, actions and shaped rewards come
from the legacy NavigationMDP. Current SingleAgentEnv has evolved separately;
new cross-host training uses the explicit navigation-transfer contract.
"""

from bluesky_gym.envs.common.navigation import NavigationMDP
from bluesky_zoo.competition.competition import CompetitionZooEnv
from bluesky_gym.envs.common.command_filter import command_config


class _AircraftNavigation(NavigationMDP):
    """Aircraft-local MDP state with a live view of its host's world state."""

    def __init__(self, env, ac_id):
        self.env = env
        self.ac_id = ac_id
        self._init_navigation(env.n_obstacles)
        self.action_config = env.action_config

    @property
    def goal_lat(self):
        return self.env._goal[self.ac_id][0]

    @property
    def goal_lon(self):
        return self.env._goal[self.ac_id][1]

    @property
    def scenario(self):
        return self.env.scenario

    @property
    def _obstacle_area_names(self):
        return self.env._obstacle_area_names

    @property
    def intruder_ids(self):
        # Never retain traffic indices: deleting a finished aircraft shifts them.
        return [agent for agent in self.env.agents if agent != self.ac_id]

    @property
    def metrics(self):
        return self.env.metrics

    @property
    def _reward_baseline(self):
        # The single-agent reward consumes one metric snapshot, whereas the
        # multi-agent host stores one snapshot per aircraft.
        return self.env._reward_baseline[self.ac_id]

    def _check_inside(self, name, lat, lon, alt):
        return self.env._check_inside(name, lat, lon, alt)


class MultiAgentEnv(CompetitionZooEnv):
    """PettingZoo ParallelEnv with the single-agent navigation policy contract.

    Observation keys, normalization, five intruder slots, absolute heading and
    speed actions and reward shaping retain the legacy contract. Action history,
    progress and restricted-area hysteresis belong to individual aircraft in
    ``navigation_states``. Use the same ``n_obstacles`` as a transferred model.
    """

    metadata = {**CompetitionZooEnv.metadata, "name": "multi_agent_v0"}
    navigation_class = _AircraftNavigation

    def __init__(self, action_config=None, **kwargs):
        self.action_config = command_config(action_config)
        kwargs.setdefault("action_frequency", 5)
        super().__init__(**kwargs)
        self.navigation_states = {
            agent: self.navigation_class(self, agent)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: state.observation_space
            for agent, state in self.navigation_states.items()
        }
        self.action_spaces = {
            agent: state.action_space
            for agent, state in self.navigation_states.items()
        }

    def reset(self, seed=None, options=None):
        for state in self.navigation_states.values():
            state._reset_navigation()
        # The base reset constructs goals and metrics, then invokes _get_obs.
        return super().reset(seed=seed, options=options)

    def _get_obs(self, ac_id):
        state = self.navigation_states[ac_id]
        if state.previous_goal_distance is None:
            state._initialize_goal_distance(ac_id)
        return state._get_obs(ac_id)

    def _get_action(self, ac_id, action):
        return self.navigation_states[ac_id]._get_action(ac_id, action)

    def _get_reward(self, ac_id):
        return self.navigation_states[ac_id]._get_reward(ac_id)


def parallel_env(**kwargs):
    """Construct the custom navigation environment using PettingZoo convention."""
    return MultiAgentEnv(**kwargs)
