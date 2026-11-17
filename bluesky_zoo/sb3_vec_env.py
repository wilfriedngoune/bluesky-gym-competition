"""Expose one homogeneous ParallelEnv as fixed slots for a shared SB3 policy.

The slots share a scenario and cannot reset independently. Once an aircraft
leaves, its slot stays at zero observation/reward with ``done=True`` until the
last aircraft leaves and the whole scenario resets. Only a real departure
produces ``episode`` and ``final_metrics`` info. Do not apply ``VecMonitor``:
it would count these absorbing steps as additional aircraft episodes.
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info

from bluesky_zoo.active_rollout_buffer import ACTIVE_TRANSITION_KEY


class SharedPolicyVecEnv(VecEnv):
    """Use ``possible_agents`` order as policy slots for one ParallelEnv.

    All agents must have the same spaces and can only disappear during a
    scenario. Observations on a departure step must include the final state,
    particularly for time-limit truncation, so SB3 can bootstrap correctly.
    A scenario is reset only after ``env.agents`` becomes empty.

    ``seed`` and ``set_options`` follow the SB3 deferred-reset API. Because
    there is one actual environment, only slot zero's seed/options are used
    on explicit ``reset``. Automatic scenario resets are unseeded. Likewise,
    ``set_attr`` and ``env_method`` operate once on the shared environment;
    method results are repeated for the selected slots.
    """

    def __init__(self, env):
        self.env = env
        self.possible_agents = tuple(env.possible_agents)
        if not self.possible_agents or len(set(self.possible_agents)) != len(self.possible_agents):
            raise ValueError("possible_agents must be nonempty and contain unique agent IDs")
        self._agent_indices = {agent: index for index, agent in enumerate(self.possible_agents)}
        first_agent = self.possible_agents[0]
        observation_space = env.observation_space(first_agent)
        action_space = env.action_space(first_agent)
        for agent in self.possible_agents[1:]:
            if env.observation_space(agent) != observation_space or env.action_space(agent) != action_space:
                raise ValueError("SharedPolicyVecEnv requires homogeneous observation and action spaces")
        super().__init__(len(self.possible_agents), observation_space, action_space)
        self._obs_keys, shapes, dtypes = obs_space_info(observation_space)
        self._obs_buffer = OrderedDict(
            (key, np.zeros((self.num_envs, *shapes[key]), dtype=dtypes[key]))
            for key in self._obs_keys
        )
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.scenario_index = -1
        self._actions = None
        self._has_reset = False
        self._closed = False

    def _batch_observations(self, observations, active_agents):
        for buffer in self._obs_buffer.values():
            buffer.fill(0)
        for agent in active_agents:
            if agent not in observations:
                raise ValueError(f"Missing observation for active agent {agent!r}")
            self._save_observation(self._agent_indices[agent], observations[agent])
        return dict_to_obs(self.observation_space, deepcopy(self._obs_buffer))

    def _save_observation(self, index, observation):
        for key in self._obs_keys:
            self._obs_buffer[key][index] = observation if key is None else observation[key]

    def _check_agents(self, agents):
        unknown = set(agents).difference(self._agent_indices)
        if unknown:
            raise ValueError(f"Agents outside possible_agents: {unknown!r}")
        if len(set(agents)) != len(agents):
            raise ValueError("env.agents contains duplicate agent IDs")

    def _reset_scenario(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._check_agents(self.env.agents)
        if not self.env.agents:
            raise ValueError("The environment must have active agents after reset")
        self.scenario_index += 1
        self._episode_rewards.fill(0)
        self._episode_lengths.fill(0)
        self.reset_infos = [deepcopy(infos.get(agent, {})) for agent in self.possible_agents]
        self._has_reset = True
        return self._batch_observations(observations, self.env.agents)

    def reset(self):
        if self._closed:
            raise RuntimeError("Cannot reset a closed SharedPolicyVecEnv")
        self._actions = None
        observations = self._reset_scenario(seed=self._seeds[0], options=self._options[0] or None)
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions):
        if self._closed or not self._has_reset:
            raise RuntimeError("Call reset() on an open SharedPolicyVecEnv before stepping")
        if self._actions is not None:
            raise RuntimeError("A step is already pending")
        actions = np.asarray(actions)
        if actions.ndim == 0 or len(actions) != self.num_envs:
            raise ValueError(f"Expected one action for each of {self.num_envs} agent slots")
        self._actions = actions.copy()

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("Call step_async() before step_wait()")
        step_agents = tuple(self.env.agents)
        actions = {agent: self._actions[self._agent_indices[agent]] for agent in step_agents}
        self._actions = None
        observations, rewards, terminations, truncations, source_infos = self.env.step(actions)
        remaining_agents = tuple(self.env.agents)
        self._check_agents(remaining_agents)
        if not set(remaining_agents).issubset(step_agents):
            raise ValueError("Agents cannot join or reappear before the scenario resets")

        # Departed slots remain absorbing, and never repeat their real metrics.
        batch_rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.ones(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in self.possible_agents]
        for agent in step_agents:
            index = self._agent_indices[agent]
            terminated = bool(terminations[agent])
            truncated = bool(truncations[agent])
            done = terminated or truncated
            if done == (agent in remaining_agents):
                raise ValueError(f"Agent {agent!r} membership disagrees with termination/truncation flags")
            reward = float(rewards[agent])
            batch_rewards[index] = reward
            dones[index] = done
            self._episode_rewards[index] += reward
            self._episode_lengths[index] += 1
            info = deepcopy(source_infos.get(agent, {}))
            # This aircraft was active BEFORE the step. Keep its arrival/timeout
            # transition even though its next observation may already be padding.
            info[ACTIVE_TRANSITION_KEY] = True
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                if agent not in observations:
                    raise ValueError(f"Missing terminal observation for agent {agent!r}")
                info["terminal_observation"] = deepcopy(observations[agent])
                info["final_metrics"] = deepcopy(source_infos.get(agent, {}))
                get_components = getattr(self.env, "get_reward_components", None)
                if callable(get_components):
                    info["reward_components"] = dict(get_components(agent))
                info["agent_id"] = agent
                info["scenario_index"] = self.scenario_index
                info["episode"] = {
                    "r": float(self._episode_rewards[index]),
                    "l": int(self._episode_lengths[index]),
                }
            infos[index] = info

        if remaining_agents:
            batch_obs = self._batch_observations(observations, remaining_agents)
        else:
            # Preserve the ending step's infos and bootstrap observations while
            # returning the next scenario's initial observations, as SB3 expects.
            batch_obs = self._reset_scenario()
        return batch_obs, batch_rewards, dones, infos

    def close(self):
        if not self._closed:
            self.env.close()
            self._closed = True
            self._actions = None

    def _indices(self, indices):
        selected = list(self._get_indices(indices))
        for index in selected:
            if index < 0 or index >= self.num_envs:
                raise IndexError(f"Agent slot index {index} is out of range")
        return selected

    def get_attr(self, attr_name, indices=None):
        selected = self._indices(indices)
        if not selected:
            return []
        value = getattr(self.env, attr_name)
        return [value for _ in selected]

    def set_attr(self, attr_name, value, indices=None):
        if self._indices(indices):
            setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        selected = self._indices(indices)
        if not selected:
            return []
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result for _ in selected]

    def env_is_wrapped(self, wrapper_class, indices=None):
        selected = self._indices(indices)
        current = self.env
        wrapped = False
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, wrapper_class):
                wrapped = True
                break
            current = getattr(current, "env", None)
        return [wrapped for _ in selected]

    def get_images(self):
        frame = self.env.render() if self.render_mode == "rgb_array" else None
        return [frame for _ in self.possible_agents]

    def render(self, mode=None):
        # One shared scene needs one frame, rather than a tiled duplicate per agent.
        if (mode or self.render_mode) == "rgb_array" and self.render_mode == "rgb_array":
            return self.env.render()
        return super().render(mode=mode)
