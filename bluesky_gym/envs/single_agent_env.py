import numpy as np
import bluesky as bs

from bluesky_gym.envs.competition_env import CompetitionEnv
from core.tools import kwikqdrdist


NM2KM = 1.852


class SingleAgentEnv(CompetitionEnv):

    def __init__(self, **kwargs):

        super().__init__(
            d_heading=15.0,
            d_speed=10.0,
            action_frequency=5,
            **kwargs
        )

        # Previous goal distance
        self.previous_goal_distance_km = None

    # RESET
    def reset(self, seed=None, options=None):

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        ac_idx = bs.traf.id2idx(self.agent)

        _, distance_nm = kwikqdrdist(
            bs.traf.lat[ac_idx],
            bs.traf.lon[ac_idx],
            self.goal_lat,
            self.goal_lon
        )

        self.previous_goal_distance_km = distance_nm * NM2KM

        return obs, info

    # OBSERVATION
    def _get_obs(self, ac_id):

        inside_sector = not self._outside_sector

        obs = {
            # Goal position
            **self.waypoint_obs.observe(
                ac_id,
                [self.goal_lat],
                [self.goal_lon]
            ),

            # Own speed
            **self.airspeed_obs.observe(ac_id),

            # Dynamic obstacles: aircraft
            **self.intruder_obs.observe(ac_id),

            # Static obstacles
            **self.obstacle_obs.observe(
                ac_id,
                self.obstacle_centre_lat,
                self.obstacle_centre_lon,
                self.obstacle_radius
            ),

            # Sector boundary
            **self.sector_obs.observe(
                ac_id,
                self.scenario.sector,
                inside_sector
            ),
        }

        return obs

    # ACTION
    def _get_action(self, ac_id, action):

        action = np.asarray(action, dtype=np.float64)

        # Safety clip
        action = np.clip(action, -1.0, 1.0)

        # action[0] -> heading
        # action[1] -> speed
        self.heading_action.execute(
            ac_id,
            action[0]
        )

        self.speed_action.execute(
            ac_id,
            action[1]
        )

    # REWARD
    def _get_reward(self, ac_id):

        reward = 0.0

        m = self.metrics[ac_id]
        previous = self._reward_baseline

        ac_idx = bs.traf.id2idx(ac_id)

        # 1. DISTANCE TO GOAL
        _, distance_nm = kwikqdrdist(
            bs.traf.lat[ac_idx],
            bs.traf.lon[ac_idx],
            self.goal_lat,
            self.goal_lon
        )

        distance_km = distance_nm * NM2KM

        if self.previous_goal_distance_km is not None:

            progress = (
                self.previous_goal_distance_km
                - distance_km
            )

            # Positive if moving toward goal
            reward += 0.5 * progress

        self.previous_goal_distance_km = distance_km

        # 2. GOAL REACHED
        reached = (
            m["waypoint_reached"]
            - previous["waypoint_reached"]
        )

        if reached > 0:
            reward += 100.0

        # 3. LOSS OF SEPARATION
        intrusion_events = (
            m["intrusion_events"]
            - previous["intrusion_events"]
        )

        intrusion_time = (
            m["intrusion_time"]
            - previous["intrusion_time"]
        )

        reward -= 50.0 * intrusion_events
        reward -= 5.0 * intrusion_time

        # 4. RESTRICTED AREAS
        restricted_events = (
            m["restricted_area_events"]
            - previous["restricted_area_events"]
        )

        restricted_time = (
            m["time_in_restricted_area"]
            - previous["time_in_restricted_area"]
        )

        reward -= 40.0 * restricted_events
        reward -= 4.0 * restricted_time

        # 5. SECTOR EXIT
        sector_exit_events = (
            m["sector_exit_events"]
            - previous["sector_exit_events"]
        )

        outside_time = (
            m["time_outside_sector"]
            - previous["time_outside_sector"]
        )

        reward -= 50.0 * sector_exit_events
        reward -= 5.0 * outside_time

        # 6. TIME / TRAJECTORY EFFICIENCY
        elapsed_time = (
            m["flight_time"]
            - previous["flight_time"]
        )

        # Small time penalty
        reward -= 0.02 * elapsed_time


        # 7. DIRECTION TOWARD GOAL
        goal_heading, _ = kwikqdrdist(
            bs.traf.lat[ac_idx],
            bs.traf.lon[ac_idx],
            self.goal_lat,
            self.goal_lon
        )

        current_heading = bs.traf.hdg[ac_idx]

        heading_error = (
            (current_heading - goal_heading + 180)
            % 360
            - 180
        )

        heading_error = abs(heading_error) / 180.0

        # Small penalty only
        reward -= 0.1 * heading_error

        return float(reward)