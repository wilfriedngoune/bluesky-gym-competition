"""Active-only PPO batches must retain arrivals, timeout returns and checkpoints."""

import io
import unittest
from unittest.mock import patch

from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO

from bluesky_zoo.active_rollout_buffer import ActiveAircraftCallback, ActiveAircraftRolloutBuffer
from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv
from test_sb3_vec_env import FakeParallelEnv


class ActiveRolloutTests(unittest.TestCase):
    def test_filter_preserves_gae_and_alignment_across_epochs(self):
        buffer = ActiveAircraftRolloutBuffer(
            3, spaces.Dict({"position": spaces.Box(-1000, 1000, (1,), dtype=np.float32)}),
            spaces.Box(-1, 1, (1,), dtype=np.float32), n_envs=2, gamma=0.9, gae_lambda=1,
        )
        # First aircraft arrives immediately. Large padding values would corrupt
        # normalization/losses if filtering used next-state rather than pre-step activity.
        for step in range(3):
            buffer.set_active_mask([step == 0, True])
            buffer.add(
                {"position": np.array([[10 if step == 0 else 999], [20 + step]])},
                np.array([[0.1 if step == 0 else 0.9], [0.2 + step / 10]]),
                np.array([1 if step == 0 else 999, 1 if step < 2 else 5.5]),
                np.array([True, step == 0]),
                torch.tensor([0 if step == 0 else 999, 0]), torch.zeros(2),
            )
        buffer.compute_returns_and_advantage(torch.zeros(2), np.ones(2, dtype=bool))
        expected_returns = {10: 1, 20: 6.355, 21: 5.95, 22: 5.5}
        for _ in range(2):
            records = {}
            for batch in buffer.get(batch_size=3):
                for position, action, target in zip(batch.observations["position"], batch.actions, batch.returns):
                    position = int(position.item())
                    records[position] = float(target)
                    self.assertAlmostEqual(float(action.item()), 0.1 if position == 10 else (position - 18) / 10)
            self.assertEqual(set(records), set(expected_returns))
            for position, target in expected_returns.items():
                self.assertAlmostEqual(records[position], target, places=5)
        self.assertEqual(len(buffer.values), 4)
        buffer.reset()
        self.assertEqual(buffer.active_masks.shape, (3, 2))
        with self.assertRaisesRegex(RuntimeError, "Missing active-aircraft mask"):
            buffer.add(None)
        with self.assertRaises(ValueError):
            buffer.set_active_mask([False, False])

    def test_real_ppo_updates_use_live_samples_and_bootstrap_timeout_once(self):
        torch.set_num_threads(1)
        vec = SharedPolicyVecEnv(FakeParallelEnv())
        self.addCleanup(vec.close)
        model = PPO(
            "MultiInputPolicy", vec, n_steps=3, batch_size=3, n_epochs=2,
            gamma=0.9, gae_lambda=1, seed=7, device="cpu",
            policy_kwargs={"net_arch": [8]}, rollout_buffer_class=ActiveAircraftRolloutBuffer,
        )
        with patch.object(model.policy, "predict_values", side_effect=lambda obs: torch.full((len(obs["visible"]), 1), 5.0)), \
                patch.object(model.policy, "evaluate_actions", wraps=model.policy.evaluate_actions) as evaluate:
            model.learn(total_timesteps=9, callback=ActiveAircraftCallback())
        self.assertEqual(model.num_timesteps, 9)
        self.assertEqual(model.competition_active_timesteps, 6)
        self.assertEqual(len(model.rollout_buffer.returns), 6)
        for call in evaluate.call_args_list:
            self.assertTrue(torch.all(call.args[0]["visible"] == 1))
        # Fake aircraft A/B arrive after 1/2 steps; C times out after 3.
        targets = dict(zip(model.rollout_buffer.observations["position"][:, 0],
                           model.rollout_buffer.returns.ravel()))
        for position, expected in {100: 1, 101: 3.8, 111: 2, 102: 11.775, 112: 9.75, 122: 7.5}.items():
            self.assertAlmostEqual(float(targets[position]), expected, places=4)
        self.assertEqual([episode["l"] for episode in model.ep_info_buffer], [1, 2, 3])

        observation = vec.reset()
        action = model.predict(observation, deterministic=True)[0]
        with io.BytesIO() as checkpoint:
            model.save(checkpoint)
            checkpoint.seek(0)
            restored = PPO.load(checkpoint, device="cpu")
        np.testing.assert_array_equal(restored.predict(observation, deterministic=True)[0], action)
        self.assertIsInstance(restored.rollout_buffer, ActiveAircraftRolloutBuffer)


if __name__ == "__main__":
    unittest.main()
