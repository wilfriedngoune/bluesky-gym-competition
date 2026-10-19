import numpy as np
import bluesky as bs

from gymnasium import spaces

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_gym.envs.common import functions as fn

from core.tools import kwikqdrdist


M_PER_NM = 1852.0
MPS_TO_KT = 1.94384


class SingleAgentEnv(CompetitionEnv):

    def __init__(self, **kwargs):

        kwargs.setdefault(
            "action_frequency",
            5
        )

        super().__init__(
            **kwargs
        )

        # =====================================================
        # Configuration
        # =====================================================

        self.n_tracked_intruders = 5

        # Maximum offset around the navigation reference.
        self.max_heading_offset = 70.0

        # Absolute target-speed range.
        self.min_target_speed = 120.0
        self.max_target_speed = 220.0

        # =====================================================
        # Continuous action space
        # =====================================================

        # action[0]:
        #   -1 -> reference heading - 70 deg
        #    0 -> reference heading
        #   +1 -> reference heading + 70 deg
        #
        # action[1]:
        #   -1 -> 120 kt
        #   +1 -> 220 kt

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

        # =====================================================
        # Observation space
        # =====================================================

        self.observation_space = spaces.Dict({

            # Direction of the goal in body frame.
            "goal_direction": spaces.Box(
                -1.0,
                1.0,
                shape=(2,),
                dtype=np.float64,
            ),

            # Normalized distance to goal.
            "goal_distance": spaces.Box(
                0.0,
                3.0,
                shape=(1,),
                dtype=np.float64,
            ),

            # Current speed.
            "own_speed": spaces.Box(
                0.0,
                2.0,
                shape=(1,),
                dtype=np.float64,
            ),

            # Navigation mode:
            # 0 = goal
            # 1 = escape restricted area
            # 2 = return inside sector
            "navigation_mode": spaces.Box(
                0.0,
                2.0,
                shape=(1,),
                dtype=np.float64,
            ),

            # One row per obstacle:
            #
            # 0 nearest boundary x
            # 1 nearest boundary y
            # 2 centroid x
            # 3 centroid y
            # 4 clearance
            # 5 approximate radius
            # 6 inside flag
            "obstacles": spaces.Box(
                -3.0,
                3.0,
                shape=(
                    self.n_obstacles,
                    7,
                ),
                dtype=np.float64,
            ),

            # Most relevant intruders:
            #
            # 0 relative x
            # 1 relative y
            # 2 relative vx
            # 3 relative vy
            # 4 distance
            # 5 TCPA
            # 6 DCPA
            "intruders": spaces.Box(
                -3.0,
                3.0,
                shape=(
                    self.n_tracked_intruders,
                    7,
                ),
                dtype=np.float64,
            ),

            # Sector:
            #
            # 0 nearest boundary x
            # 1 nearest boundary y
            # 2 clearance
            # 3 inside flag
            "sector": spaces.Box(
                -3.0,
                3.0,
                shape=(4,),
                dtype=np.float64,
            ),

            # Previous policy action.
            "previous_action": spaces.Box(
                -1.0,
                1.0,
                shape=(2,),
                dtype=np.float64,
            ),
        })

        # =====================================================
        # Episode state
        # =====================================================

        self.previous_goal_distance = None
        self.previous_escape_clearance = None

        self.was_inside_restricted = False

        self.last_action = np.zeros(
            2,
            dtype=np.float64
        )

        self.previous_action = np.zeros(
            2,
            dtype=np.float64
        )

        # Debug information.
        self.last_reference_heading = 0.0
        self.last_target_heading = 0.0
        self.last_target_speed = 0.0
        self.last_navigation_mode = 0

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        self.last_action[:] = 0.0
        self.previous_action[:] = 0.0

        self.previous_escape_clearance = None
        self.was_inside_restricted = False

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        own_idx = bs.traf.id2idx(
            self.agent
        )

        _, distance_nm = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        self.previous_goal_distance = (
            distance_nm
        )

        return obs, info

    # =========================================================
    # ANGLE
    # =========================================================

    def _wrap_angle(
        self,
        angle
    ):

        return (
            (
                angle + 180.0
            )
            % 360.0
        ) - 180.0

    # =========================================================
    # BODY POSITION
    # =========================================================

    def _relative_position(
        self,
        own_idx,
        lat,
        lon,
        norm=100.0,
    ):

        qdr, distance_nm = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            lat,
            lon,
        )

        relative_angle = np.deg2rad(
            qdr
            - bs.traf.hdg[own_idx]
        )

        x = (
            distance_nm
            * np.cos(relative_angle)
        )

        y = (
            distance_nm
            * np.sin(relative_angle)
        )

        return np.clip(
            np.array([
                x / norm,
                y / norm,
            ]),
            -3.0,
            3.0,
        )

    # =========================================================
    # SEGMENT GEOMETRY
    # =========================================================

    def _nearest_point_segment(
        self,
        point,
        start,
        end,
    ):

        segment = (
            end - start
        )

        length_squared = np.dot(
            segment,
            segment
        )

        if length_squared <= 1e-12:

            return start.copy()

        t = np.dot(
            point - start,
            segment
        ) / length_squared

        t = np.clip(
            t,
            0.0,
            1.0
        )

        return (
            start
            + t * segment
        )

    # =========================================================
    # POLYGON GEOMETRY
    # =========================================================

    def _polygon_info(
        self,
        own_idx,
        vertices,
    ):
        """
        Compute nearest boundary point, clearance,
        centroid and approximate radius.
        """

        center = np.array(
            self.scenario.center
        )

        aircraft_nm = fn.latlong_to_nm(
            center,
            np.array([
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
            ])
        )

        vertices_nm = np.array([
            fn.latlong_to_nm(
                center,
                np.array(vertex)
            )
            for vertex in vertices
        ])

        nearest_point = None
        minimum_distance = np.inf

        for i in range(
            len(vertices_nm)
        ):

            start = vertices_nm[i]

            end = vertices_nm[
                (i + 1)
                % len(vertices_nm)
            ]

            candidate = (
                self._nearest_point_segment(
                    aircraft_nm,
                    start,
                    end,
                )
            )

            distance = np.linalg.norm(
                aircraft_nm
                - candidate
            )

            if (
                distance
                < minimum_distance
            ):

                minimum_distance = (
                    distance
                )

                nearest_point = (
                    candidate
                )

        centroid_nm = np.mean(
            vertices_nm,
            axis=0
        )

        radius = np.max(
            np.linalg.norm(
                vertices_nm
                - centroid_nm,
                axis=1
            )
        )

        nearest_latlon = (
            fn.nm_to_latlong(
                center,
                nearest_point
            )
        )

        centroid_latlon = (
            fn.nm_to_latlong(
                center,
                centroid_nm
            )
        )

        return (
            float(minimum_distance),
            nearest_latlon,
            centroid_latlon,
            float(radius),
        )

    # =========================================================
    # OBSTACLE STATE
    # =========================================================

    def _get_obstacle_states(
        self,
        own_idx,
    ):

        states = []

        for obstacle_index, obstacle in enumerate(
            self.scenario.obstacles
        ):

            (
                clearance,
                nearest_point,
                centroid,
                radius,
            ) = self._polygon_info(
                own_idx,
                obstacle.vertices,
            )

            inside = self._check_inside(
                self._obstacle_area_names[
                    obstacle_index
                ],
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
                bs.traf.alt[own_idx],
            )

            nearest_relative = (
                self._relative_position(
                    own_idx,
                    nearest_point[0],
                    nearest_point[1],
                    norm=50.0,
                )
            )

            centroid_relative = (
                self._relative_position(
                    own_idx,
                    centroid[0],
                    centroid[1],
                    norm=100.0,
                )
            )

            states.append({
                "index": obstacle_index,
                "clearance": clearance,
                "inside": inside,
                "nearest_point": nearest_point,
                "nearest_relative":
                    nearest_relative,
                "centroid_relative":
                    centroid_relative,
                "radius": radius,
            })

        states.sort(
            key=lambda item:
                item["clearance"]
        )

        return states

    # =========================================================
    # SECTOR STATE
    # =========================================================

    def _get_sector_state(
        self,
        own_idx,
    ):

        (
            clearance,
            nearest_point,
            _,
            _,
        ) = self._polygon_info(
            own_idx,
            self.scenario.sector,
        )

        inside = self._check_inside(
            "sector",
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            bs.traf.alt[own_idx],
        )

        nearest_relative = (
            self._relative_position(
                own_idx,
                nearest_point[0],
                nearest_point[1],
                norm=50.0,
            )
        )

        return {
            "clearance": clearance,
            "inside": inside,
            "nearest_point": nearest_point,
            "nearest_relative":
                nearest_relative,
        }

    # =========================================================
    # CPA
    # =========================================================

    def _compute_cpa(
        self,
        own_idx,
        intruder_idx,
    ):

        qdr, distance_nm = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            bs.traf.lat[intruder_idx],
            bs.traf.lon[intruder_idx],
        )

        bearing = np.deg2rad(
            qdr
        )

        relative_position = np.array([
            distance_nm
            * np.cos(bearing),

            distance_nm
            * np.sin(bearing),
        ])

        own_heading = np.deg2rad(
            bs.traf.hdg[own_idx]
        )

        intruder_heading = np.deg2rad(
            bs.traf.hdg[intruder_idx]
        )

        own_velocity = np.array([
            np.cos(own_heading),
            np.sin(own_heading),
        ]) * (
            bs.traf.tas[own_idx]
            / M_PER_NM
        )

        intruder_velocity = np.array([
            np.cos(intruder_heading),
            np.sin(intruder_heading),
        ]) * (
            bs.traf.tas[intruder_idx]
            / M_PER_NM
        )

        relative_velocity = (
            intruder_velocity
            - own_velocity
        )

        velocity_squared = np.dot(
            relative_velocity,
            relative_velocity
        )

        if velocity_squared < 1e-10:

            return (
                600.0,
                distance_nm,
                distance_nm,
            )

        tcpa = -np.dot(
            relative_position,
            relative_velocity
        ) / velocity_squared

        if tcpa < 0.0:

            return (
                600.0,
                distance_nm,
                distance_nm,
            )

        position_cpa = (
            relative_position
            + tcpa
            * relative_velocity
        )

        dcpa = np.linalg.norm(
            position_cpa
        )

        return (
            float(tcpa),
            float(dcpa),
            float(distance_nm),
        )

    # =========================================================
    # INTRUDER OBSERVATION
    # =========================================================

    def _get_intruder_features(
        self,
        own_idx,
    ):

        candidates = []

        own_heading = np.deg2rad(
            bs.traf.hdg[own_idx]
        )

        own_vn = (
            np.cos(own_heading)
            * bs.traf.tas[own_idx]
        )

        own_ve = (
            np.sin(own_heading)
            * bs.traf.tas[own_idx]
        )

        for intruder_id in (
            self.intruder_ids
        ):

            idx = bs.traf.id2idx(
                intruder_id
            )

            if idx < 0:
                continue

            relative_position = (
                self._relative_position(
                    own_idx,
                    bs.traf.lat[idx],
                    bs.traf.lon[idx],
                    norm=100.0,
                )
            )

            intruder_heading = np.deg2rad(
                bs.traf.hdg[idx]
            )

            int_vn = (
                np.cos(intruder_heading)
                * bs.traf.tas[idx]
            )

            int_ve = (
                np.sin(intruder_heading)
                * bs.traf.tas[idx]
            )

            relative_vn = (
                int_vn
                - own_vn
            )

            relative_ve = (
                int_ve
                - own_ve
            )

            vx = (
                relative_vn
                * np.cos(own_heading)
                + relative_ve
                * np.sin(own_heading)
            )

            vy = (
                -relative_vn
                * np.sin(own_heading)
                + relative_ve
                * np.cos(own_heading)
            )

            tcpa, dcpa, distance = (
                self._compute_cpa(
                    own_idx,
                    idx,
                )
            )

            # Lower score means more urgent.
            if tcpa < 300.0:

                urgency = (
                    dcpa
                    + 0.02 * tcpa
                )

            else:

                urgency = (
                    1000.0
                    + distance
                )

            candidates.append(
                (
                    urgency,
                    np.array([
                        relative_position[0],
                        relative_position[1],

                        np.clip(
                            vx / 150.0,
                            -3.0,
                            3.0
                        ),

                        np.clip(
                            vy / 150.0,
                            -3.0,
                            3.0
                        ),

                        np.clip(
                            distance / 100.0,
                            0.0,
                            3.0
                        ),

                        np.clip(
                            tcpa / 300.0,
                            0.0,
                            2.0
                        ),

                        np.clip(
                            dcpa / 10.0,
                            0.0,
                            3.0
                        ),
                    ])
                )
            )

        candidates.sort(
            key=lambda item:
                item[0]
        )

        features = np.zeros(
            (
                self.n_tracked_intruders,
                7,
            )
        )

        for slot, (
            _,
            feature,
        ) in enumerate(
            candidates[
                :self.n_tracked_intruders
            ]
        ):

            features[slot] = (
                feature
            )

        return features

    # =========================================================
    # NAVIGATION REFERENCE
    # =========================================================

    def _get_navigation_reference(
        self,
        own_idx,
    ):

        obstacle_states = (
            self._get_obstacle_states(
                own_idx
            )
        )

        # Restricted area has local priority.
        for obstacle in (
            obstacle_states
        ):

            if obstacle["inside"]:

                qdr, _ = kwikqdrdist(
                    bs.traf.lat[own_idx],
                    bs.traf.lon[own_idx],
                    obstacle[
                        "nearest_point"
                    ][0],
                    obstacle[
                        "nearest_point"
                    ][1],
                )

                return (
                    qdr,
                    1,
                )

        sector_state = (
            self._get_sector_state(
                own_idx
            )
        )

        # If outside the sector, return inside.
        if not sector_state["inside"]:

            qdr, _ = kwikqdrdist(
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
                sector_state[
                    "nearest_point"
                ][0],
                sector_state[
                    "nearest_point"
                ][1],
            )

            return (
                qdr,
                2,
            )

        # Normal mode: fly toward the goal.
        goal_heading, _ = (
            kwikqdrdist(
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
                self.goal_lat,
                self.goal_lon,
            )
        )

        return (
            goal_heading,
            0,
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(
        self,
        ac_id,
    ):

        own_idx = bs.traf.id2idx(
            ac_id
        )

        # -----------------------------------------------------
        # Goal
        # -----------------------------------------------------

        goal_heading, goal_distance = (
            kwikqdrdist(
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
                self.goal_lat,
                self.goal_lon,
            )
        )

        goal_error = np.deg2rad(
            self._wrap_angle(
                goal_heading
                - bs.traf.hdg[own_idx]
            )
        )

        goal_direction = np.array([
            np.cos(goal_error),
            np.sin(goal_error),
        ])

        goal_distance_obs = np.array([
            np.clip(
                goal_distance / 100.0,
                0.0,
                3.0
            )
        ])

        # -----------------------------------------------------
        # Own speed
        # -----------------------------------------------------

        own_speed_knots = (
            bs.traf.cas[own_idx]
            * MPS_TO_KT
        )

        own_speed = np.array([
            np.clip(
                own_speed_knots / 200.0,
                0.0,
                2.0
            )
        ])

        # -----------------------------------------------------
        # Obstacles
        # -----------------------------------------------------

        obstacle_states = (
            self._get_obstacle_states(
                own_idx
            )
        )

        obstacle_features = np.zeros(
            (
                self.n_obstacles,
                7,
            )
        )

        for slot, obstacle in enumerate(
            obstacle_states
        ):

            obstacle_features[
                slot
            ] = np.array([

                obstacle[
                    "nearest_relative"
                ][0],

                obstacle[
                    "nearest_relative"
                ][1],

                obstacle[
                    "centroid_relative"
                ][0],

                obstacle[
                    "centroid_relative"
                ][1],

                np.clip(
                    obstacle[
                        "clearance"
                    ] / 20.0,
                    0.0,
                    3.0
                ),

                np.clip(
                    obstacle[
                        "radius"
                    ] / 20.0,
                    0.0,
                    3.0
                ),

                1.0
                if obstacle["inside"]
                else 0.0,
            ])

        # -----------------------------------------------------
        # Sector
        # -----------------------------------------------------

        sector_state = (
            self._get_sector_state(
                own_idx
            )
        )

        sector_features = np.array([

            sector_state[
                "nearest_relative"
            ][0],

            sector_state[
                "nearest_relative"
            ][1],

            np.clip(
                sector_state[
                    "clearance"
                ] / 20.0,
                0.0,
                3.0
            ),

            1.0
            if sector_state["inside"]
            else 0.0,
        ])

        # -----------------------------------------------------
        # Navigation mode
        # -----------------------------------------------------

        _, navigation_mode = (
            self._get_navigation_reference(
                own_idx
            )
        )

        # -----------------------------------------------------
        # Intruders
        # -----------------------------------------------------

        intruder_features = (
            self._get_intruder_features(
                own_idx
            )
        )

        return {

            "goal_direction":
                goal_direction,

            "goal_distance":
                goal_distance_obs,

            "own_speed":
                own_speed,

            "navigation_mode":
                np.array([
                    float(
                        navigation_mode
                    )
                ]),

            "obstacles":
                obstacle_features,

            "intruders":
                intruder_features,

            "sector":
                sector_features,

            "previous_action":
                self.last_action.copy(),
        }

    # =========================================================
    # ACTION
    # =========================================================

    def _get_action(
        self,
        ac_id,
        action,
    ):
        """
        Map the policy output to an absolute target heading
        and an absolute target speed.
        """

        action = np.asarray(
            action,
            dtype=np.float64
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        self.previous_action = (
            self.last_action.copy()
        )

        self.last_action = (
            action.copy()
        )

        own_idx = bs.traf.id2idx(
            ac_id
        )

        (
            reference_heading,
            navigation_mode,
        ) = self._get_navigation_reference(
            own_idx
        )

        heading_offset = (
            action[0]
            * self.max_heading_offset
        )

        target_heading = (
            self._wrap_angle(
                reference_heading
                + heading_offset
            )
        )

        speed_fraction = (
            action[1]
            + 1.0
        ) / 2.0

        target_speed = (
            self.min_target_speed
            + speed_fraction
            * (
                self.max_target_speed
                - self.min_target_speed
            )
        )

        # Direct absolute BlueSky commands.
        bs.stack.stack(
            f"HDG {ac_id} "
            f"{target_heading}"
        )

        bs.stack.stack(
            f"SPD {ac_id} "
            f"{target_speed}"
        )

        self.last_reference_heading = (
            reference_heading
        )

        self.last_target_heading = (
            target_heading
        )

        self.last_target_speed = (
            target_speed
        )

        self.last_navigation_mode = (
            navigation_mode
        )

    # =========================================================
    # REWARD
    # =========================================================

    def _get_reward(
        self,
        ac_id,
    ):

        reward = 0.0

        metrics = self.metrics[
            ac_id
        ]

        previous = (
            self._reward_baseline
        )

        own_idx = bs.traf.id2idx(
            ac_id
        )

        elapsed_time = (
            metrics["flight_time"]
            - previous["flight_time"]
        )

        # =====================================================
        # CURRENT GEOMETRY
        # =====================================================

        obstacle_states = (
            self._get_obstacle_states(
                own_idx
            )
        )

        inside_obstacles = [
            obstacle
            for obstacle
            in obstacle_states
            if obstacle["inside"]
        ]

        inside_restricted = (
            len(
                inside_obstacles
            )
            > 0
        )

        nearest_clearance = min(
            obstacle[
                "clearance"
            ]
            for obstacle
            in obstacle_states
        )

        # =====================================================
        # 1. GOAL
        # =====================================================

        _, goal_distance = (
            kwikqdrdist(
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
                self.goal_lat,
                self.goal_lon,
            )
        )

        if (
            self.previous_goal_distance
            is not None
        ):

            progress = (
                self.previous_goal_distance
                - goal_distance
            )

            # Normal mode:
            # strongly reward goal progress.
            if not inside_restricted:

                if progress >= 0.0:

                    reward += (
                        3.0
                        * progress
                    )

                else:

                    # Moving away may be necessary for avoidance,
                    # therefore use only a weak penalty.
                    reward += (
                        0.5
                        * progress
                    )

        self.previous_goal_distance = (
            goal_distance
        )

        reached = (
            metrics["waypoint_reached"]
            - previous[
                "waypoint_reached"
            ]
        )

        if reached > 0:

            reward += 100.0

        # =====================================================
        # 2. RESTRICTED AREAS
        # =====================================================

        restricted_events = (
            metrics[
                "restricted_area_events"
            ]
            - previous[
                "restricted_area_events"
            ]
        )

        restricted_time = (
            metrics[
                "time_in_restricted_area"
            ]
            - previous[
                "time_in_restricted_area"
            ]
        )

        reward -= (
            10.0
            * restricted_events
        )

        reward -= (
            0.5
            * restricted_time
        )

        if inside_restricted:

            current_escape_clearance = min(
                obstacle[
                    "clearance"
                ]
                for obstacle
                in inside_obstacles
            )

            if (
                self.previous_escape_clearance
                is not None
            ):

                escape_progress = (
                    self.previous_escape_clearance
                    - current_escape_clearance
                )

                # Positive when approaching the boundary.
                reward += (
                    5.0
                    * escape_progress
                )

            self.previous_escape_clearance = (
                current_escape_clearance
            )

        else:

            self.previous_escape_clearance = (
                None
            )

        # Bonus when the aircraft successfully leaves
        # a restricted area.
        if (
            self.was_inside_restricted
            and not inside_restricted
        ):

            reward += 10.0

        self.was_inside_restricted = (
            inside_restricted
        )

        # Penalize getting close before entering.
        if (
            not inside_restricted
            and nearest_clearance < 8.0
        ):

            risk = (
                1.0
                - nearest_clearance
                / 8.0
            )

            reward -= (
                0.15
                * risk
                * elapsed_time
            )

        # =====================================================
        # 3. AIRCRAFT CONFLICTS
        # =====================================================

        intrusion_events = (
            metrics[
                "intrusion_events"
            ]
            - previous[
                "intrusion_events"
            ]
        )

        intrusion_time = (
            metrics[
                "intrusion_time"
            ]
            - previous[
                "intrusion_time"
            ]
        )

        reward -= (
            20.0
            * intrusion_events
        )

        reward -= (
            1.0
            * intrusion_time
        )

        highest_risk = 0.0

        for intruder_id in (
            self.intruder_ids
        ):

            idx = bs.traf.id2idx(
                intruder_id
            )

            if idx < 0:
                continue

            (
                tcpa,
                dcpa,
                _,
            ) = self._compute_cpa(
                own_idx,
                idx,
            )

            if (
                0.0
                < tcpa
                < 120.0
                and dcpa < 8.0
            ):

                time_risk = (
                    1.0
                    - tcpa / 120.0
                )

                separation_risk = (
                    1.0
                    - dcpa / 8.0
                )

                risk = (
                    time_risk
                    * separation_risk
                )

                highest_risk = max(
                    highest_risk,
                    risk,
                )

        reward -= (
            0.5
            * highest_risk
            * elapsed_time
        )

        # =====================================================
        # 4. SECTOR
        # =====================================================

        sector_exit_events = (
            metrics[
                "sector_exit_events"
            ]
            - previous[
                "sector_exit_events"
            ]
        )

        outside_time = (
            metrics[
                "time_outside_sector"
            ]
            - previous[
                "time_outside_sector"
            ]
        )

        reward -= (
            15.0
            * sector_exit_events
        )

        reward -= (
            0.5
            * outside_time
        )

        # =====================================================
        # 5. FLIGHT TIME
        # =====================================================

        reward -= (
            0.002
            * elapsed_time
        )

        # =====================================================
        # 6. ACTION SMOOTHNESS
        # =====================================================

        action_change = np.linalg.norm(
            self.last_action
            - self.previous_action
        )

        reward -= (
            0.02
            * action_change
        )

        return float(
            reward
        )