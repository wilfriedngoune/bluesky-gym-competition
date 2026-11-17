"""Centralized-critic (CTDE / MAPPO-style) multi-agent scaffolding.

The multi-agent pipeline shares one PPO ``MultiInputPolicy`` across all aircraft,
but its value network only sees a single aircraft's local observation. This module
adds the missing multi-agent ingredient: a *centralized* critic.

Design
------
* Every per-aircraft observation ``Dict`` gains one extra key, ``CENTRALIZED_KEY``,
  holding a compact, symmetric summary of the whole scenario (mean-pooled local
  features over the live aircraft plus the active-aircraft count). The same vector
  is written to every active aircraft's observation.
* ``CentralizedObservationWrapper`` injects that key around any PettingZoo
  ``ParallelEnv`` so it is present identically during training (through
  ``SharedPolicyVecEnv``), replay and official evaluation.
* ``CentralizedCriticPolicy`` is a PPO policy whose *actor* reads only the local
  keys and whose *critic* reads only the centralized key. This keeps the learned
  action decentralized (each aircraft acts on its own view) while the value/
  advantage estimate is computed from the joint state -- the CTDE / MAPPO recipe.

The centralized key is just another ``Dict`` entry, so the existing
``ActiveAircraftRolloutBuffer`` masks its padding exactly like every other feature;
no changes to GAE, the buffer, the reward, the scoring or the evaluation protocol
are needed.

Only training, replay and evaluation consume this. The scored metrics, scenario
generator, dynamics and thresholds are untouched.
"""

from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import BaseModel, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor, MlpExtractor


CENTRALIZED_KEY = "centralized_state"


def centralized_obs_space(local_obs_space):
    """Box space for the centralized summary of a per-aircraft Dict observation.

    One float per local feature (mean-pooled) plus one active-aircraft count.
    """
    total = sum(int(np.prod(local_obs_space[key].shape)) for key in local_obs_space.spaces)
    return spaces.Box(low=0.0, high=1.0, shape=(total + 1,), dtype=np.float32)


def build_centralized_obs(per_agent_obs, local_keys, n_max_agents):
    """Mean-pool the live aircraft's local features, then append the active count.

    ``per_agent_obs`` maps a live aircraft id to its (local) observation dict. The
    result is a single float32 vector of shape ``centralized_obs_space`` that is
    identical for every aircraft -- the joint state a centralized critic consumes.
    Inactive aircraft simply do not contribute; the trailing count tells the critic
    how many are currently live.
    """
    agents = list(per_agent_obs)
    if not agents:
        raise ValueError("build_centralized_obs needs at least one live aircraft")
    pooled = []
    for key in local_keys:
        stacked = np.stack([np.asarray(per_agent_obs[a][key], dtype=np.float32).reshape(-1) for a in agents], axis=0)
        pooled.append(stacked.mean(axis=0))
    body = np.concatenate([part.reshape(-1) for part in pooled])
    active_count = np.array([len(agents) / float(n_max_agents)], dtype=np.float32)
    return np.concatenate([body, active_count]).astype(np.float32)


def is_centralized_observation_space(space):
    """True when a (possibly saved) observation space carries the centralized key."""
    return isinstance(space, spaces.Dict) and CENTRALIZED_KEY in space.spaces


class CentralizedObservationWrapper:
    """Transparent PettingZoo wrapper that appends the centralized key to each obs.

    Only the observation dictionaries are touched (a copy is made; the inner
    environment is never mutated). Every other attribute and method is forwarded
    to the wrapped environment, so the wrapper is safe to use in place of the raw
    multi-agent environment for training, replay and evaluation alike.
    """

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_possible_agents", tuple(env.possible_agents))
        base = env.observation_space(self._possible_agents[0])
        object.__setattr__(self, "_local_keys", list(base.spaces))
        object.__setattr__(self, "_n_max", len(self._possible_agents))
        object.__setattr__(self, "_global_space", centralized_obs_space(base))
        augmented = spaces.Dict({**base.spaces, CENTRALIZED_KEY: self._global_space})
        object.__setattr__(self, "_observation_spaces", {a: augmented for a in self._possible_agents})
        object.__setattr__(self, "_action_spaces", {a: env.action_space(a) for a in self._possible_agents})
        object.__setattr__(self, "metadata", getattr(env, "metadata", {}))
        object.__setattr__(self, "render_mode", getattr(env, "render_mode", None))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    @property
    def possible_agents(self):
        return self._possible_agents

    @property
    def agents(self):
        return self._env.agents

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _augment(self, observations, active_agents):
        vector = build_centralized_obs({a: observations[a] for a in active_agents}, self._local_keys, self._n_max)
        for agent in observations:
            merged = dict(observations[agent])
            merged[CENTRALIZED_KEY] = vector
            observations[agent] = merged
        return observations

    def reset(self, seed=None, options=None):
        observations, infos = self._env.reset(seed=seed, options=options)
        self._augment(observations, list(self._env.agents))
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self._env.step(actions)
        active = [a for a in self._env.agents]
        if not active:  # scenario just ended; fall back to the final snapshot
            active = list(observations)
        self._augment(observations, active)
        return observations, rewards, terminations, truncations, infos

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


class CentralizedCriticPolicy(MultiInputActorCriticPolicy):
    """PPO policy with a decentralized actor and a centralized critic.

    The actor (policy net) is built from every observation key except
    ``global_key``; the critic (value net) is built from ``global_key`` alone. Both
    networks are independent ``MlpExtractor``s because their input dimensions
    differ, which the stock shared ``MlpExtractor`` cannot express. The
    actor/critic split happens at the features-extractor level, so the rollout
    buffer, GAE and active-aircraft masking all work unchanged.
    """

    def __init__(self, observation_space, action_space, lr_schedule, global_key,
                 net_arch=None, activation_fn=nn.Tanh, ortho_init=True, use_sde=False,
                 log_std_init=0.0, full_std=True, use_expln=False, squash_output=False,
                 features_extractor_class=CombinedExtractor, features_extractor_kwargs=None,
                 share_features_extractor=False, normalize_images=True,
                 optimizer_class=th.optim.Adam, optimizer_kwargs=None):
        self._centralized_global_key = global_key
        self._centralized_local_keys = [k for k in observation_space.spaces if k != global_key]
        if not self._centralized_local_keys:
            raise ValueError("A centralized critic needs at least one local observation key")
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn,
            ortho_init, use_sde, log_std_init, full_std, use_expln, squash_output,
            features_extractor_class, features_extractor_kwargs, share_features_extractor,
            normalize_images, optimizer_class, optimizer_kwargs,
        )

    def _build(self, lr_schedule):
        local_space = spaces.Dict({k: self.observation_space[k] for k in self._centralized_local_keys})
        global_space = spaces.Dict({self._centralized_global_key: self.observation_space[self._centralized_global_key]})
        self.pi_features_extractor = self._build_extractor(local_space)
        self.vf_features_extractor = self._build_extractor(global_space)
        # Replace the placeholder all-key extractor the base __init__ created.
        self.features_extractor = self.pi_features_extractor
        self.features_dim = self.pi_features_extractor.features_dim

        if isinstance(self.net_arch, list):
            arch = {"pi": list(self.net_arch), "vf": list(self.net_arch)}
        else:
            arch = self.net_arch
        self.mlp_extractor = MlpExtractor(self.pi_features_extractor.features_dim,
                                           {"pi": arch.get("pi", []), "vf": []}, self.activation_fn, self.device)
        self.mlp_extractor_vf = MlpExtractor(self.vf_features_extractor.features_dim,
                                              {"vf": arch.get("vf", []), "pi": []}, self.activation_fn, self.device)
        latent_pi = self.mlp_extractor.latent_dim_pi
        latent_vf = self.mlp_extractor_vf.latent_dim_vf

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_pi, log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_pi, latent_sde_dim=latent_pi, log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution,
                                           BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(latent_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.pi_features_extractor: np.sqrt(2),
                self.vf_features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.mlp_extractor_vf: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_extractor(self, subset_space):
        return self.features_extractor_class(subset_space, **self.features_extractor_kwargs)

    def _actor_latent(self, obs):
        # Route through BaseModel.extract_features so observations get the same
        # preprocessing (and float64 -> float32 cast) as the stock policy path.
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        return self.mlp_extractor.forward_actor(features)

    def _critic_latent(self, obs):
        features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
        return self.mlp_extractor_vf.forward_critic(features)

    def forward(self, obs, deterministic=False):
        latent_pi = self._actor_latent(obs)
        latent_vf = self._critic_latent(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi = self._actor_latent(obs)
        latent_vf = self._critic_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        return self._get_action_dist_from_latent(self._actor_latent(obs))

    def predict_values(self, obs):
        return self.value_net(self._critic_latent(obs))

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(dict(global_key=self._centralized_global_key))
        return data
