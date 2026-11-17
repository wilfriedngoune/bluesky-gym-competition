"""Centralized-critic (CTDE) checks without initializing BlueSky's global simulator.

Mirrors the fake-env approach of ``test_sb3_vec_env``: a 3-agent PettingZoo-style
environment with departures, wrapped by ``CentralizedObservationAdapter`` and driven
through ``SharedPolicyVecEnv`` + the active-aircraft PPO buffer with the
``CentralizedCriticPolicy``. No real BlueSky environment is created.
"""

from copy import deepcopy
import tempfile
import unittest

import numpy as np
from gymnasium import spaces


class FakeParallelEnv:
    possible_agents = ["A", "B", "C"]
    render_mode = "rgb_array"
    action_frequency = 5
    center = (0.0, 0.0)

    def __init__(self):
        self.agents = []
        self._observation_space = spaces.Dict({
            "position": spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32),
            "visible": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })
        self._action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.step_number = 0
        self.custom = 0

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def _observation(self, agent):
        value = 10 * self.step_number + self.possible_agents.index(agent)
        return {"position": np.array([value, -value], dtype=np.float32),
                "visible": np.ones(1, dtype=np.float32)}

    def reset(self, seed=None, options=None):
        self.step_number = 0
        self.agents = list(self.possible_agents)
        return ({a: self._observation(a) for a in self.agents},
                {a: {"reset_agent": a} for a in self.agents})

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.step_number += 1
        step_agents = list(self.agents)
        departing = self.possible_agents[self.step_number - 1]
        observations = {a: self._observation(a) for a in step_agents}
        rewards = {a: float(self.possible_agents.index(a) + 1) for a in step_agents}
        terminations = {a: (a == departing and a != "C") for a in step_agents}
        truncations = {a: (a == departing and a == "C") for a in step_agents}
        infos = {a: {"flight_time": float(self.step_number)} for a in step_agents}
        self.agents.remove(departing)
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def close(self):
        pass


class FakeParallelEnv64(FakeParallelEnv):
    """Same env but with float64 observation spaces/values, like the real env."""

    def __init__(self):
        super().__init__()
        self._observation_space = spaces.Dict({
            "position": spaces.Box(-1000, 1000, shape=(2,), dtype=np.float64),
            "visible": spaces.Box(0, 1, shape=(1,), dtype=np.float64),
        })

    def _observation(self, agent):
        value = 10 * self.step_number + self.possible_agents.index(agent)
        return {"position": np.array([value, -value], dtype=np.float64),
                "visible": np.ones(1, dtype=np.float64)}


def _arr(tensor):
    return tensor.detach().cpu().numpy()


class CentralizedCriticTests(unittest.TestCase):
    def make_env(self):
        return FakeParallelEnv()

    # ------------------------------------------------------------ pure helpers
    def test_observation_space_shape_is_local_features_plus_count(self):
        from bluesky_zoo.centralized_critic import centralized_obs_space
        env = self.make_env()
        space = centralized_obs_space(env.observation_space("A"))
        # position(2) + visible(1) + active count(1)
        self.assertEqual(space.shape, (4,))

    def test_build_centralized_obs_pools_and_counts(self):
        from bluesky_zoo.centralized_critic import build_centralized_obs
        local_keys = ["position", "visible"]
        # Two live aircraft with distinct positions; mean pooling and a 2/3 count.
        per_agent = {
            "A": {"position": np.array([0.0, 0.0], dtype=np.float32), "visible": np.ones(1, dtype=np.float32)},
            "B": {"position": np.array([2.0, 2.0], dtype=np.float32), "visible": np.ones(1, dtype=np.float32)},
        }
        vec = build_centralized_obs(per_agent, local_keys, n_max_agents=3)
        self.assertEqual(vec.shape, (4,))
        np.testing.assert_allclose(vec[0], 1.0)   # mean of 0 and 2
        np.testing.assert_allclose(vec[1], 1.0)
        np.testing.assert_allclose(vec[2], 1.0)   # mean of the visible features
        np.testing.assert_allclose(vec[3], 2.0 / 3.0)  # active count

    def test_is_centralized_detection(self):
        from bluesky_zoo.centralized_critic import CENTRALIZED_KEY, is_centralized_observation_space
        plain = spaces.Box(0, 1, shape=(4,), dtype=np.float32)
        self.assertFalse(is_centralized_observation_space(plain))
        augmented = spaces.Dict({"a": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                                 CENTRALIZED_KEY: spaces.Box(0, 1, shape=(2,), dtype=np.float32)})
        self.assertTrue(is_centralized_observation_space(augmented))

    # --------------------------------------------------------------- wrapper
    def test_wrapper_augments_obs_and_forwards_attributes(self):
        from bluesky_zoo.centralized_critic import CENTRALIZED_KEY, CentralizedObservationWrapper
        env = self.make_env()
        wrapped = CentralizedObservationWrapper(env)
        self.addCleanup(wrapped.close)

        obs, _ = wrapped.reset(seed=0)
        self.assertEqual(set(obs), set(wrapped.possible_agents))
        self.assertEqual(obs["A"][CENTRALIZED_KEY].shape, (4,))
        # Identical joint-state vector across every live aircraft.
        np.testing.assert_array_equal(obs["A"][CENTRALIZED_KEY], obs["B"][CENTRALIZED_KEY])
        # Active count 3/3.
        np.testing.assert_allclose(obs["A"][CENTRALIZED_KEY][-1], 1.0)
        # The inner environment is never mutated (its obs dict has no centralized key).
        self.assertNotIn(CENTRALIZED_KEY, env._observation("A"))

        # Departure drops the count to 2/3 and keeps the vector consistent.
        actions = {a: np.zeros(2, dtype=np.float32) for a in wrapped.agents}
        obs, _, term, trunc, _ = wrapped.step(actions)
        self.assertEqual(set(wrapped.agents), {"B", "C"})
        np.testing.assert_allclose(obs["B"][CENTRALIZED_KEY][-1], 2.0 / 3.0)

        # Attribute forwarding (read and write) reaches the inner environment.
        self.assertEqual(wrapped.action_frequency, 5)
        wrapped.custom = 42
        self.assertEqual(env.custom, 42)
        # Observation/action spaces carry the centralized key and match the model.
        self.assertIn(CENTRALIZED_KEY, wrapped.observation_space("A").spaces)
        self.assertEqual(wrapped.action_space("A"), env._action_space)

    # ------------------------------------------------------------ PPO smoke
    def test_ppo_trains_with_centralized_critic_and_active_masking(self):
        from stable_baselines3 import PPO
        from bluesky_zoo.active_rollout_buffer import ActiveAircraftCallback, ActiveAircraftRolloutBuffer
        from bluesky_zoo.centralized_critic import CENTRALIZED_KEY, CentralizedCriticPolicy, CentralizedObservationWrapper
        from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv

        venv = SharedPolicyVecEnv(CentralizedObservationWrapper(self.make_env()))
        self.addCleanup(venv.close)
        policy_kwargs = dict(net_arch=dict(pi=[16, 16], vf=[16, 16]), log_std_init=-0.5,
                             global_key=CENTRALIZED_KEY)
        model = PPO(CentralizedCriticPolicy, venv, learning_rate=0.001, n_steps=4, batch_size=4,
                    n_epochs=2, gamma=0.99, seed=7, device="cpu", policy_kwargs=policy_kwargs,
                    rollout_buffer_class=ActiveAircraftRolloutBuffer, verbose=0)
        model.learn(total_timesteps=32, callback=ActiveAircraftCallback(), progress_bar=False)
        self.assertGreater(model.num_timesteps, 0)
        # Departing aircraft padding is excluded: fewer active than slot timesteps.
        self.assertLess(model.competition_active_timesteps, model.num_timesteps)

    def test_ppo_trains_with_float64_observations(self):
        # The real BlueSky navigation observations are float64; the policy must still
        # run (its weights are float32), so observations must be cast on the way in.
        from stable_baselines3 import PPO
        from bluesky_zoo.centralized_critic import CENTRALIZED_KEY, CentralizedCriticPolicy, CentralizedObservationWrapper
        from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv

        venv = SharedPolicyVecEnv(CentralizedObservationWrapper(FakeParallelEnv64()))
        self.addCleanup(venv.close)
        policy_kwargs = dict(net_arch=dict(pi=[8], vf=[8]), log_std_init=-0.5, global_key=CENTRALIZED_KEY)
        model = PPO(CentralizedCriticPolicy, venv, learning_rate=0.001, n_steps=4, batch_size=4,
                    n_epochs=1, seed=7, device="cpu", policy_kwargs=policy_kwargs, verbose=0)
        model.learn(total_timesteps=16, progress_bar=False)
        self.assertGreater(model.num_timesteps, 0)
        wrapped = CentralizedObservationWrapper(FakeParallelEnv64())
        self.addCleanup(wrapped.close)
        obs, _ = wrapped.reset(seed=0)
        agents = list(wrapped.agents)
        batch = {k: np.stack([obs[a][k] for a in agents]) for k in obs[agents[0]]}
        actions, _ = model.predict(batch, deterministic=True)
        self.assertEqual(actions.shape, (len(agents), 2))

    # ------------------------------------------------- actor/critic separation
    def test_actor_reads_local_only_critic_reads_centralized_only(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import obs_as_tensor
        from bluesky_zoo.centralized_critic import CENTRALIZED_KEY, CentralizedCriticPolicy, CentralizedObservationWrapper
        from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv

        venv = SharedPolicyVecEnv(CentralizedObservationWrapper(self.make_env()))
        self.addCleanup(venv.close)
        policy_kwargs = dict(net_arch=dict(pi=[8], vf=[8]), log_std_init=-0.5, global_key=CENTRALIZED_KEY)
        model = PPO(CentralizedCriticPolicy, venv, learning_rate=0.001, n_steps=4, batch_size=4,
                    n_epochs=1, seed=7, device="cpu", policy_kwargs=policy_kwargs, verbose=0)
        model.learn(total_timesteps=8, progress_bar=False)
        policy = model.policy

        wrapped = CentralizedObservationWrapper(self.make_env())
        self.addCleanup(wrapped.close)
        obs, _ = wrapped.reset(seed=0)
        agents = list(wrapped.agents)
        batch = {k: np.stack([obs[a][k] for a in agents]) for k in obs[agents[0]]}
        b = obs_as_tensor(batch, model.device)

        actor0, critic0 = _arr(policy._actor_latent(b)), _arr(policy._critic_latent(b))

        # Perturb only the centralized key: critic changes, actor is invariant.
        b_global = {k: v.clone() for k, v in b.items()}
        b_global[CENTRALIZED_KEY] = b_global[CENTRALIZED_KEY] + 0.25
        self.assertTrue(np.allclose(_arr(policy._actor_latent(b_global)), actor0, atol=1e-5))
        self.assertFalse(np.allclose(_arr(policy._critic_latent(b_global)), critic0))

        # Perturb only a local key: actor changes, critic is invariant.
        b_local = {k: v.clone() for k, v in b.items()}
        b_local["position"] = b_local["position"] + 5.0
        self.assertFalse(np.allclose(_arr(policy._actor_latent(b_local)), actor0, atol=1e-5))
        self.assertTrue(np.allclose(_arr(policy._critic_latent(b_local)), critic0, atol=1e-5))

    # -------------------------------------------------------- save / load
    def test_save_load_round_trip_preserves_centralized_policy(self):
        from stable_baselines3 import PPO
        from bluesky_zoo.centralized_critic import CENTRALIZED_KEY, CentralizedCriticPolicy, CentralizedObservationWrapper
        from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv

        venv = SharedPolicyVecEnv(CentralizedObservationWrapper(self.make_env()))
        self.addCleanup(venv.close)
        policy_kwargs = dict(net_arch=dict(pi=[8], vf=[8]), log_std_init=-0.5, global_key=CENTRALIZED_KEY)
        model = PPO(CentralizedCriticPolicy, venv, learning_rate=0.001, n_steps=4, batch_size=4,
                    n_epochs=1, seed=7, device="cpu", policy_kwargs=policy_kwargs, verbose=0)
        model.learn(total_timesteps=8, progress_bar=False)

        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/ctde"
            model.save(path)
            loaded = PPO.load(path, device="cpu")
            self.assertEqual(type(loaded.policy).__name__, "CentralizedCriticPolicy")
            self.assertIn(CENTRALIZED_KEY, loaded.observation_space.spaces)
            # Deterministic outputs must match the freshly trained policy.
            wrapped = CentralizedObservationWrapper(self.make_env())
            self.addCleanup(wrapped.close)
            obs, _ = wrapped.reset(seed=1)
            agents = list(wrapped.agents)
            batch = {k: np.stack([obs[a][k] for a in agents]) for k in obs[agents[0]]}
            a1, _ = model.predict(batch, deterministic=True)
            a2, _ = loaded.predict(batch, deterministic=True)
            np.testing.assert_allclose(a1, a2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
