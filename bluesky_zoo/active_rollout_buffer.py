"""Exclude departed aircraft's padding from PPO updates, not from simulation.

GAE is computed on the original time-by-aircraft grid using SB3's termination
and timeout handling. Only then are inactive transitions removed, before PPO
normalizes advantages or computes policy/value/entropy losses and diagnostics.
"""

import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback


ACTIVE_TRANSITION_KEY = "shared_policy_active"


class ActiveAircraftRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer requiring an explicit mask for every vector step."""

    def reset(self):
        super().reset()
        self.active_masks = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self._pending_active_mask = None

    def set_active_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_envs,) or not mask.any():
            raise ValueError("Expected one active-aircraft flag per slot and at least one active aircraft")
        if self._pending_active_mask is not None:
            raise RuntimeError("The preceding active-aircraft mask has not been consumed")
        self._pending_active_mask = mask.copy()

    def add(self, *args, **kwargs):
        if self._pending_active_mask is None:
            raise RuntimeError("Missing active-aircraft mask; train with ActiveAircraftCallback")
        self.active_masks[self.pos] = self._pending_active_mask
        super().add(*args, **kwargs)
        self._pending_active_mask = None

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Collect a complete rollout before sampling")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.generator_ready:
            # Match SB3's environment-major flattening so the mask stays aligned
            # with observations, actions and already-computed GAE targets.
            active = self.swap_and_flatten(self.active_masks).ravel()
            self.observations = {
                key: self.swap_and_flatten(observation)[active]
                for key, observation in self.observations.items()
            }
            for name in ("actions", "values", "log_probs", "advantages", "returns"):
                setattr(self, name, self.swap_and_flatten(getattr(self, name))[active])
            self.generator_ready = True

        count = len(self.returns)
        if not count:
            raise RuntimeError("Rollout contains no active-aircraft transitions")
        indices = np.random.permutation(count)
        batch_size = count if batch_size is None else batch_size
        for start in range(0, count, batch_size):
            yield self._get_samples(indices[start:start + batch_size])


class ActiveAircraftCallback(BaseCallback):
    """Pass pre-step activity to the buffer and report real sample counts."""

    def _on_training_start(self):
        self.model.competition_active_timesteps = 0

    def _on_rollout_start(self):
        self._active_samples = 0
        self._slot_samples = 0

    def _on_step(self):
        infos = self.locals["infos"]
        mask = np.array([info.get(ACTIVE_TRANSITION_KEY, False) for info in infos], dtype=bool)
        if mask.shape != (self.training_env.num_envs,) or not mask.any():
            raise ValueError("ActiveAircraftCallback requires SharedPolicyVecEnv activity flags")
        buffer = self.locals["rollout_buffer"]
        if isinstance(buffer, ActiveAircraftRolloutBuffer):
            buffer.set_active_mask(mask)
        count = int(mask.sum())
        self._active_samples += count
        self._slot_samples += len(mask)
        self.model.competition_active_timesteps += count
        return True

    def _on_rollout_end(self):
        self.logger.record("rollout/active_samples", self._active_samples)
        self.logger.record("rollout/padding_fraction", 1 - self._active_samples / self._slot_samples)
        self.logger.record("time/active_timesteps", self.model.competition_active_timesteps)
