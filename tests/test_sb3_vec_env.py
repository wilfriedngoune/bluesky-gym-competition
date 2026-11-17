"""SB3 lifecycle checks without initializing BlueSky's global simulator."""

from copy import deepcopy
import unittest

from gymnasium import spaces
import numpy as np

from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv


class FakeParallelEnv:
    possible_agents = ["A", "B", "C"]
    render_mode = "rgb_array"

    def __init__(self, dict_observations=True):
        self.agents = []
        position_space = spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32)
        self._observation_space = (
            spaces.Dict({
                "position": position_space,
                "visible": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            })
            if dict_observations else position_space
        )
        self._action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.reset_calls = []
        self.action_calls = []
        self.close_calls = 0
        self.render_calls = 0
        self.method_calls = 0
        self.value = 3

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def _observation(self, agent):
        value = 100 * len(self.reset_calls) + 10 * self.step_number + self.possible_agents.index(agent)
        position = np.array([value, -value], dtype=np.float32)
        if isinstance(self._observation_space, spaces.Dict):
            return {"position": position, "visible": np.ones(1, dtype=np.float32)}
        return position

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, deepcopy(options)))
        self.step_number = 0
        self.agents = list(self.possible_agents)
        return (
            {agent: self._observation(agent) for agent in self.agents},
            {agent: {"reset_agent": agent} for agent in self.agents},
        )

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.action_calls.append(deepcopy(actions))
        self.step_number += 1
        step_agents = list(self.agents)
        departing_agent = self.possible_agents[self.step_number - 1]
        observations = {agent: self._observation(agent) for agent in step_agents}
        rewards = {agent: float(self.possible_agents.index(agent) + 1) for agent in step_agents}
        terminations = {agent: agent == departing_agent and agent != "C" for agent in step_agents}
        truncations = {agent: agent == departing_agent and agent == "C" for agent in step_agents}
        infos = {
            agent: {"flight_time": float(self.step_number), "waypoint_reached": int(terminations[agent])}
            for agent in step_agents
        }
        self.agents.remove(departing_agent)
        return observations, rewards, terminations, truncations, infos

    def add(self, increment):
        self.method_calls += 1
        self.value += increment
        return self.value

    def render(self):
        self.render_calls += 1
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def close(self):
        self.close_calls += 1


class SharedPolicyVecEnvTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = FakeParallelEnv(**kwargs)
        vec = SharedPolicyVecEnv(env)
        self.addCleanup(vec.close)
        return env, vec

    @staticmethod
    def actions():
        return np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

    def test_individual_departure_absorbs_until_whole_scenario_finishes(self):
        env, vec = self.make_env()
        initial = vec.reset()
        np.testing.assert_array_equal(initial["position"], [[100, -100], [101, -101], [102, -102]])
        self.assertEqual(vec.reset_infos, [{"reset_agent": agent} for agent in env.possible_agents])

        observations, rewards, dones, infos = vec.step(self.actions())
        np.testing.assert_array_equal(rewards, [1, 2, 3])
        np.testing.assert_array_equal(dones, [True, False, False])
        np.testing.assert_array_equal(observations["position"][0], [0, 0])
        np.testing.assert_array_equal(observations["visible"][0], [0])
        np.testing.assert_array_equal(infos[0]["terminal_observation"]["position"], [110, -110])
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertEqual(infos[0]["agent_id"], "A")
        self.assertEqual(infos[0]["scenario_index"], 0)
        self.assertEqual(infos[0]["episode"], {"r": 1, "l": 1})
        self.assertEqual(infos[0]["final_metrics"], {"flight_time": 1, "waypoint_reached": 1})
        self.assertEqual(len(env.reset_calls), 1)

        observations, rewards, dones, infos = vec.step(self.actions())
        np.testing.assert_array_equal(rewards, [0, 2, 3])
        np.testing.assert_array_equal(dones, [True, True, False])
        np.testing.assert_array_equal(observations["position"][:2], [[0, 0], [0, 0]])
        self.assertEqual(infos[0], {})
        self.assertEqual(infos[1]["episode"], {"r": 4, "l": 2})
        self.assertEqual(set(env.action_calls[-1]), {"B", "C"})
        self.assertEqual(len(env.reset_calls), 1)

        observations, rewards, dones, infos = vec.step(self.actions())
        np.testing.assert_array_equal(rewards, [0, 0, 3])
        np.testing.assert_array_equal(dones, [True, True, True])
        self.assertEqual(infos[:2], [{}, {}])
        self.assertTrue(infos[2]["TimeLimit.truncated"])
        self.assertEqual(infos[2]["episode"], {"r": 9, "l": 3})
        self.assertEqual(infos[2]["scenario_index"], 0)
        np.testing.assert_array_equal(infos[2]["terminal_observation"]["position"], [132, -132])
        np.testing.assert_array_equal(observations["position"], [[200, -200], [201, -201], [202, -202]])
        self.assertEqual(set(env.action_calls[-1]), {"C"})
        self.assertEqual(len(env.reset_calls), 2)
        self.assertEqual(vec.scenario_index, 1)

        _, _, _, next_infos = vec.step(self.actions())
        self.assertEqual(next_infos[0]["scenario_index"], 1)
        self.assertEqual(next_infos[0]["episode"], {"r": 1, "l": 1})

    def test_seed_and_options_are_consumed_only_by_explicit_reset(self):
        env, vec = self.make_env()
        self.assertEqual(vec.seed(42), [42, 43, 44])
        options = {"variant": [1]}
        vec.set_options(options)
        options["variant"].append(2)
        vec.reset()
        self.assertEqual(env.reset_calls, [(42, {"variant": [1]})])
        self.assertEqual(vec._seeds, [None, None, None])
        self.assertEqual(vec._options, [{}, {}, {}])

        vec.seed(84)
        vec.set_options({"next_explicit": True})
        for _ in range(3):
            vec.step(self.actions())
        self.assertEqual(env.reset_calls[-1], (None, None))
        vec.reset()
        self.assertEqual(env.reset_calls[-1], (84, {"next_explicit": True}))
        vec.reset()
        self.assertEqual(env.reset_calls[-1], (None, None))

    def test_box_batching_terminal_snapshots_and_buffer_ownership(self):
        _, vec = self.make_env(dict_observations=False)
        initial = vec.reset()
        initial.fill(999)
        observations, _, _, infos = vec.step(self.actions())
        self.assertEqual(observations.shape, (3, 2))
        self.assertEqual(observations.dtype, np.float32)
        np.testing.assert_array_equal(observations[1], [111, -111])
        terminal = infos[0]["terminal_observation"]
        vec.step(self.actions())
        vec.step(self.actions())
        np.testing.assert_array_equal(terminal, [110, -110])
        np.testing.assert_array_equal(observations[1], [111, -111])

    def test_attributes_methods_and_render_operate_once_on_shared_environment(self):
        env, vec = self.make_env()
        self.assertEqual(vec.get_attr("value", indices=[2, 0]), [3, 3])
        vec.set_attr("value", 10, indices=1)
        self.assertEqual(vec.get_attr("value"), [10, 10, 10])
        self.assertEqual(vec.env_method("add", 2, indices=[0, 2]), [12, 12])
        self.assertEqual(env.method_calls, 1)
        self.assertEqual(vec.env_method("add", 2, indices=[]), [])
        self.assertEqual(env.method_calls, 1)
        self.assertEqual(vec.env_is_wrapped(FakeParallelEnv, indices=[0, 2]), [True, True])
        self.assertEqual(vec.env_is_wrapped(str, indices=1), [False])
        with self.assertRaises(IndexError):
            vec.get_attr("value", indices=3)
        self.assertEqual(vec.render().shape, (2, 3, 3))
        self.assertEqual(env.render_calls, 1)
        self.assertEqual(len(vec.get_images()), 3)
        self.assertEqual(env.render_calls, 2)
        vec.close()
        vec.close()
        self.assertEqual(env.close_calls, 1)

    def test_async_actions_are_copied_and_reset_cancels_pending_step(self):
        env, vec = self.make_env()
        with self.assertRaises(RuntimeError):
            vec.step(self.actions())
        vec.reset()
        actions = self.actions()
        vec.step_async(actions)
        actions.fill(0)
        with self.assertRaises(RuntimeError):
            vec.step_async(actions)
        vec.step_wait()
        np.testing.assert_array_equal(env.action_calls[0]["A"], [np.float32(0.1)])
        vec.step_async(self.actions())
        vec.reset()
        with self.assertRaises(RuntimeError):
            vec.step_wait()
        with self.assertRaises(ValueError):
            vec.step_async(np.zeros((2, 1)))

    def test_sb3_ppo_can_train_with_dict_observations_and_truncated_slots(self):
        from stable_baselines3 import PPO

        _, vec = self.make_env()
        model = PPO(
            "MultiInputPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
            learning_rate=0.001, seed=7, device="cpu",
            policy_kwargs={"net_arch": [8]},
        )
        model.learn(total_timesteps=24)
        self.assertEqual(model.num_timesteps, 24)
        # SB3 records only real completion infos, rather than absorbing slots.
        self.assertEqual(len(model.ep_info_buffer), 8)
        self.assertEqual([item["l"] for item in model.ep_info_buffer], [1, 2, 3, 1, 2, 3, 1, 2])


if __name__ == "__main__":
    unittest.main()
