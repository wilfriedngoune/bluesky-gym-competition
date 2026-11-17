"""MA navigation v3: all traffic, explicit masks and episode-time awareness."""

import numpy as np
from gymnasium import spaces

from bluesky_zoo.multi_agent_reward_env import MultiAgentRewardEnv, _RewardNavigation
from bluesky_zoo.navigation_reward import make_reward_config


class _TrafficNavigation(_RewardNavigation):
    def __init__(self, env, ac_id):
        super().__init__(env, ac_id)
        # Fixed across 2-to-10-agent curriculum stages: checkpoint shapes match.
        self.n_tracked_intruders = 9
        self.intruder_mask = np.zeros(9, dtype=np.float64)
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "intruders": spaces.Box(-3.0, 3.0, shape=(9, 7), dtype=np.float64),
            "intruder_mask": spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float64),
            "remaining_time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
        })

    def _reset_navigation(self):
        super()._reset_navigation()
        self.intruder_mask = np.zeros(9, dtype=np.float64)

    def _get_intruder_features(self, own_idx):
        candidates = self._get_intruder_candidates(own_idx)
        # Keep existing losses of separation visible first, even when receding.
        # Relative distance is normalized by 100 NM in the shared feature code.
        candidates.sort(key=lambda item: (
            item[1][4] * 100.0 >= self.env.intrusion_distance, item[0],
        ))
        features = np.zeros((9, 7), dtype=np.float64)
        self.intruder_mask.fill(0)
        for slot, (_, feature) in enumerate(candidates[:9]):
            features[slot] = feature
            self.intruder_mask[slot] = 1
        return features

    def _get_obs(self, ac_id):
        observation = super()._get_obs(ac_id)
        observation["intruder_mask"] = self.intruder_mask.copy()
        observation["remaining_time"] = np.array([
            np.clip(1.0 - self.env.sim_time / self.env.episode_time_limit, 0.0, 1.0)
        ], dtype=np.float64)
        return observation


class MultiAgentTrafficEnv(MultiAgentRewardEnv):
    """Same world/actions/scoring, with a larger MA-specific observation.

    V2/SA checkpoints cannot load directly into the changed input shape. Start
    fresh or use a v3 checkpoint. All v3 curriculum stages share identical spaces.
    """

    navigation_class = _TrafficNavigation
    metadata = {**MultiAgentRewardEnv.metadata, "name": "navigation_v3"}

    def __init__(self, reward_config=None, **kwargs):
        scenario = kwargs.get("scenario")
        count = len(scenario.agents) if scenario is not None else kwargs.get("n_agents", 10)
        if not 2 <= count <= 10:
            raise ValueError("navigation-v3 supports 2 to 10 aircraft, with nine traffic slots")
        if reward_config is None:
            reward_config = make_reward_config("navigation-v3")
        super().__init__(reward_config=reward_config, **kwargs)
