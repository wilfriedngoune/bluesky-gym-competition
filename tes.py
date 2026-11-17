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

        kwargs.setdefault("action_frequency", 5)
        super().__init__(**kwargs)

        # =====================================================
        # CONFIGURATION
        # =====================================================

        self.n_tracked_intruders = 5

        # Absolute target-speed range.
        self.min_target_speed = 180.0
        self.max_target_speed = 300.0

        # Maximum policy heading offset in normal navigation.
        self.max_heading_offset = 45.0

        # -----------------------------------------------------
        # Restricted areas
        # -----------------------------------------------------

        # Nominal safety buffer.
        self.restricted_buffer_nm = 5.0

        # Minimum buffer even under strong deadline pressure.
        self.min_restricted_buffer_nm = 4.0

        # Avoidance starts before reaching the actual buffer.
        self.obstacle_trigger_extra_nm = 2.0

        # Hysteresis prevents repeated mode switching.
        self.obstacle_release_extra_nm = 3.0

        self.active_buffer_obstacle = None

        # -----------------------------------------------------
        # Deadline management
        # -----------------------------------------------------

        self.arrival_reserve_s = 300.0
        self.time_pressure_horizon_s = 900.0
        self.hard_deadline_margin_s = 150.0

        # -----------------------------------------------------
        # Sector
        # -----------------------------------------------------

        self.max_sector_excursion_nm = 8.0

        # -----------------------------------------------------
        # Aircraft conflicts
        # -----------------------------------------------------

        # Predict conflicts earlier than before.
        self.conflict_prediction_time_s = 150.0
        self.conflict_prediction_dcpa_nm = 9.0

        # Start reacting early.
        self.conflict_activation_risk = 0.20

        # Strong intervention.
        self.conflict_emergency_risk = 0.50

        # -----------------------------------------------------
        # Persistent obstacle side
        # -----------------------------------------------------

        self.avoidance_side = 0
        self.avoidance_side_obstacle = None

        # -----------------------------------------------------
        # Stuck detection
        # -----------------------------------------------------

        self.stuck_counter = 0
        self.stuck_side_switched = False

        self.stuck_progress_threshold_nm = 0.03
        self.stuck_side_switch_steps = 30
        self.stuck_direct_steps = 70

        # -----------------------------------------------------
        # Direct-goal rescue
        # -----------------------------------------------------

        self.goal_rescue_until = 0.0
        self.goal_rescue_duration_s = 120.0

        self.failure_penalty_applied = False

        # -----------------------------------------------------
        # Debug
        # -----------------------------------------------------

        self.last_remaining_time = self.episode_time_limit
        self.last_time_to_goal = 0.0
        self.last_time_slack = self.episode_time_limit
        self.last_time_pressure = 0.0
        self.last_dynamic_buffer = self.restricted_buffer_nm

        # =====================================================
        # ACTION SPACE
        # =====================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        self.observation_space = spaces.Dict({

            "goal_direction": spaces.Box(
                -1.0, 1.0,
                shape=(2,),
                dtype=np.float64,
            ),

            "goal_distance": spaces.Box(
                0.0, 3.0,
                shape=(1,),
                dtype=np.float64,
            ),

            "remaining_time": spaces.Box(
                0.0, 1.0,
                shape=(1,),
                dtype=np.float64,
            ),

            "time_to_goal": spaces.Box(
                0.0, 2.0,
                shape=(1,),
                dtype=np.float64,
            ),

            "time_slack": spaces.Box(
                -1.0, 1.0,
                shape=(1,),
                dtype=np.float64,
            ),

            "own_speed": spaces.Box(
                0.0, 2.0,
                shape=(1,),
                dtype=np.float64,
            ),

            # 0 = goal
            # 1 = restricted-area avoidance
            # 2 = sector return
            # 3 = aircraft conflict avoidance
            # 4 = direct-goal rescue
            "navigation_mode": spaces.Box(
                0.0, 4.0,
                shape=(1,),
                dtype=np.float64,
            ),

            "obstacles": spaces.Box(
                -3.0, 3.0,
                shape=(self.n_obstacles, 7),
                dtype=np.float64,
            ),

            "intruders": spaces.Box(
                -3.0, 3.0,
                shape=(self.n_tracked_intruders, 7),
                dtype=np.float64,
            ),

            "sector": spaces.Box(
                -3.0, 3.0,
                shape=(4,),
                dtype=np.float64,
            ),

            "previous_action": spaces.Box(
                -1.0, 1.0,
                shape=(2,),
                dtype=np.float64,
            ),
        })

        # =====================================================
        # EPISODE STATE
        # =====================================================

        self.previous_goal_distance = None
        self.previous_escape_clearance = None
        self.was_inside_restricted = False

        self.last_action = np.zeros(2, dtype=np.float64)
        self.previous_action = np.zeros(2, dtype=np.float64)

        self.last_reference_heading = 0.0
        self.last_target_heading = 0.0
        self.last_target_speed = 0.0
        self.last_navigation_mode = 0

    # =========================================================
    # RESET
    # =========================================================

    def reset(self, seed=None, options=None):

        self.last_action[:] = 0.0
        self.previous_action[:] = 0.0

        self.previous_escape_clearance = None
        self.was_inside_restricted = False

        self.active_buffer_obstacle = None
        self.avoidance_side = 0
        self.avoidance_side_obstacle = None

        self.stuck_counter = 0
        self.stuck_side_switched = False

        self.goal_rescue_until = 0.0
        self.failure_penalty_applied = False

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        own_idx = bs.traf.id2idx(self.agent)

        _, distance_nm = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        self.previous_goal_distance = distance_nm

        return obs, info

    # =========================================================
    # BASIC GEOMETRY
    # =========================================================

    def _wrap_angle(self, angle):
        return ((angle + 180.0) % 360.0) - 180.0

    def _angle_distance(self, heading_a, heading_b):
        return abs(self._wrap_angle(heading_a - heading_b))

    def _heading_vector(self, heading_deg):

        rad = np.deg2rad(heading_deg)

        return np.array([
            np.cos(rad),
            np.sin(rad),
        ])

    def _vector_heading(
        self,
        vector,
        fallback_heading,
    ):

        norm = np.linalg.norm(vector)

        if norm <= 1e-8:
            return float(fallback_heading)

        vector = vector / norm

        return self._wrap_angle(
            np.rad2deg(
                np.arctan2(
                    vector[1],
                    vector[0],
                )
            )
        )

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

        angle = np.deg2rad(
            qdr - bs.traf.hdg[own_idx]
        )

        x = distance_nm * np.cos(angle)
        y = distance_nm * np.sin(angle)

        return np.clip(
            np.array([
                x / norm,
                y / norm,
            ]),
            -3.0,
            3.0,
        )

    def _nearest_point_segment(
        self,
        point,
        start,
        end,
    ):

        segment = end - start
        length_squared = np.dot(
            segment,
            segment,
        )

        if length_squared <= 1e-12:
            return start.copy()

        t = np.dot(
            point - start,
            segment,
        ) / length_squared

        t = np.clip(
            t,
            0.0,
            1.0,
        )

        return start + t * segment

    def _polygon_info(
        self,
        own_idx,
        vertices,
    ):

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
                np.array(vertex),
            )
            for vertex in vertices
        ])

        nearest_point = None
        minimum_distance = np.inf

        for i in range(len(vertices_nm)):

            start = vertices_nm[i]
            end = vertices_nm[
                (i + 1) % len(vertices_nm)
            ]

            candidate = (
                self._nearest_point_segment(
                    aircraft_nm,
                    start,
                    end,
                )
            )

            distance = np.linalg.norm(
                aircraft_nm - candidate
            )

            if distance < minimum_distance:
                minimum_distance = distance
                nearest_point = candidate

        centroid_nm = np.mean(
            vertices_nm,
            axis=0,
        )

        radius = np.max(
            np.linalg.norm(
                vertices_nm - centroid_nm,
                axis=1,
            )
        )

        nearest_latlon = fn.nm_to_latlong(
            center,
            nearest_point,
        )

        centroid_latlon = fn.nm_to_latlong(
            center,
            centroid_nm,
        )

        return (
            float(minimum_distance),
            nearest_latlon,
            centroid_latlon,
            float(radius),
        )

    # =========================================================
    # OBSTACLES
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

            states.append({
                "index": obstacle_index,
                "clearance": clearance,
                "inside": inside,
                "nearest_point": nearest_point,

                "nearest_relative":
                    self._relative_position(
                        own_idx,
                        nearest_point[0],
                        nearest_point[1],
                        norm=50.0,
                    ),

                "centroid_relative":
                    self._relative_position(
                        own_idx,
                        centroid[0],
                        centroid[1],
                        norm=100.0,
                    ),

                "radius": radius,
            })

        states.sort(
            key=lambda item:
                item["clearance"]
        )

        return states

    def _get_buffer_avoidance_heading(
        self,
        own_idx,
        obstacle_state,
    ):

        boundary_point = (
            obstacle_state["nearest_point"]
        )

        qdr_to_boundary, _ = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            boundary_point[0],
            boundary_point[1],
        )

        # When inside, leave through the nearest boundary.
        if obstacle_state["inside"]:
            return qdr_to_boundary

        # When outside, move away from the boundary.
        return self._wrap_angle(
            qdr_to_boundary + 180.0
        )

    # =========================================================
    # SECTOR
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

        return {
            "clearance": clearance,
            "inside": inside,
            "nearest_point": nearest_point,
            "nearest_relative":
                self._relative_position(
                    own_idx,
                    nearest_point[0],
                    nearest_point[1],
                    norm=50.0,
                ),
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

        bearing = np.deg2rad(qdr)

        relative_position = np.array([
            distance_nm * np.cos(bearing),
            distance_nm * np.sin(bearing),
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
            relative_velocity,
        )

        if velocity_squared < 1e-10:

            return (
                600.0,
                distance_nm,
                distance_nm,
            )

        tcpa = -np.dot(
            relative_position,
            relative_velocity,
        ) / velocity_squared

        if tcpa < 0.0:

            return (
                600.0,
                distance_nm,
                distance_nm,
            )

        position_cpa = (
            relative_position
            + tcpa * relative_velocity
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
    # CONFLICT NAVIGATION
    # =========================================================

    def _get_conflict_state(
        self,
        own_idx,
    ):

        best_risk = 0.0
        best_heading = None

        goal_heading, _ = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        for intruder_id in self.intruder_ids:

            idx = bs.traf.id2idx(
                intruder_id
            )

            if idx < 0:
                continue

            (
                tcpa,
                dcpa,
                distance,
            ) = self._compute_cpa(
                own_idx,
                idx,
            )

            qdr_to_intruder, _ = kwikqdrdist(
                bs.traf.lat[own_idx],
                bs.traf.lon[own_idx],
                bs.traf.lat[idx],
                bs.traf.lon[idx],
            )

            # Immediate loss of separation.
            if distance < 5.0:

                risk = 1.0

                escape_heading = (
                    self._wrap_angle(
                        qdr_to_intruder
                        + 180.0
                    )
                )

            elif (
                0.0 < tcpa
                < self.conflict_prediction_time_s
                and dcpa
                < self.conflict_prediction_dcpa_nm
            ):

                time_risk = (
                    1.0
                    - tcpa
                    / self.conflict_prediction_time_s
                )

                separation_risk = (
                    1.0
                    - dcpa
                    / self.conflict_prediction_dcpa_nm
                )

                risk = (
                    time_risk
                    * separation_risk
                )

                left = self._wrap_angle(
                    qdr_to_intruder - 90.0
                )

                right = self._wrap_angle(
                    qdr_to_intruder + 90.0
                )

                # Prefer the lateral solution that preserves
                # more goal progress.
                if (
                    self._angle_distance(
                        left,
                        goal_heading,
                    )
                    <=
                    self._angle_distance(
                        right,
                        goal_heading,
                    )
                ):

                    escape_heading = left

                else:

                    escape_heading = right

            else:

                continue

            if risk > best_risk:

                best_risk = risk
                best_heading = escape_heading

        return (
            best_heading,
            float(best_risk),
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

        for intruder_id in self.intruder_ids:

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

            relative_vn = int_vn - own_vn
            relative_ve = int_ve - own_ve

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

            (
                tcpa,
                dcpa,
                distance,
            ) = self._compute_cpa(
                own_idx,
                idx,
            )

            # Sort primarily by predicted conflict relevance.
            if (
                0.0 < tcpa
                < self.conflict_prediction_time_s
            ):

                urgency = (
                    dcpa
                    + 0.015 * tcpa
                )

            else:

                urgency = (
                    1000.0
                    + distance
                )

            feature = np.array([
                relative_position[0],
                relative_position[1],

                np.clip(
                    vx / 150.0,
                    -3.0,
                    3.0,
                ),

                np.clip(
                    vy / 150.0,
                    -3.0,
                    3.0,
                ),

                np.clip(
                    distance / 100.0,
                    0.0,
                    3.0,
                ),

                np.clip(
                    tcpa / 300.0,
                    0.0,
                    2.0,
                ),

                np.clip(
                    dcpa / 10.0,
                    0.0,
                    3.0,
                ),
            ])

            candidates.append(
                (
                    urgency,
                    feature,
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

            features[slot] = feature

        return features

    # =========================================================
    # TIME
    # =========================================================

    def _get_time_state(
        self,
        own_idx,
    ):

        flight_time = self.metrics[
            self.agent
        ]["flight_time"]

        remaining_time = max(
            0.0,
            self.episode_time_limit
            - flight_time,
        )

        _, goal_distance_nm = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        time_to_goal = (
            goal_distance_nm
            / self.max_target_speed
            * 3600.0
        )

        time_slack = (
            remaining_time
            - time_to_goal
        )

        effective_slack = (
            time_slack
            - self.arrival_reserve_s
        )

        time_pressure = np.clip(
            1.0
            - effective_slack
            / self.time_pressure_horizon_s,
            0.0,
            1.0,
        )

        return (
            float(remaining_time),
            float(time_to_goal),
            float(time_slack),
            float(time_pressure),
        )

    def _get_dynamic_buffer(
        self,
        time_pressure,
    ):

        # Buffer remains between 4 and 5 NM.
        buffer_nm = (
            self.min_restricted_buffer_nm
            + (
                self.restricted_buffer_nm
                - self.min_restricted_buffer_nm
            )
            * (
                1.0 - time_pressure
            )
        )

        trigger_nm = (
            buffer_nm
            + self.obstacle_trigger_extra_nm
        )

        release_nm = (
            buffer_nm
            + self.obstacle_release_extra_nm
        )

        return (
            float(buffer_nm),
            float(trigger_nm),
            float(release_nm),
        )

    # =========================================================
    # OBSTACLE NAVIGATION
    # =========================================================

    def _get_goal_biased_avoidance_heading(
        self,
        own_idx,
        obstacle_state,
        time_pressure,
    ):

        goal_heading, _ = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        escape_heading = (
            self._get_buffer_avoidance_heading(
                own_idx,
                obstacle_state,
            )
        )

        # Inside restricted area: leave immediately.
        if obstacle_state["inside"]:
            return escape_heading

        tangent_left = (
            self._wrap_angle(
                escape_heading - 90.0
            )
        )

        tangent_right = (
            self._wrap_angle(
                escape_heading + 90.0
            )
        )

        obstacle_index = (
            obstacle_state["index"]
        )

        # Keep a persistent avoidance side.
        if (
            self.avoidance_side_obstacle
            != obstacle_index
        ):

            self.avoidance_side_obstacle = (
                obstacle_index
            )

            self.stuck_side_switched = False

            if (
                self._angle_distance(
                    tangent_left,
                    goal_heading,
                )
                <=
                self._angle_distance(
                    tangent_right,
                    goal_heading,
                )
            ):

                self.avoidance_side = -1

            else:

                self.avoidance_side = 1

        # If progress stops, try the other side once.
        if (
            self.stuck_counter
            >= self.stuck_side_switch_steps
            and not self.stuck_side_switched
        ):

            self.avoidance_side *= -1
            self.stuck_side_switched = True

        tangent_heading = (
            tangent_left
            if self.avoidance_side < 0
            else tangent_right
        )

        (
            buffer_nm,
            _,
            _,
        ) = self._get_dynamic_buffer(
            time_pressure
        )

        # Very close to the buffer:
        # strongly prioritize moving away.
        if (
            obstacle_state["clearance"]
            <= buffer_nm
        ):

            goal_weight = 0.02
            tangent_weight = 0.33
            escape_weight = 0.65

        else:

            goal_weight = (
                0.25
                + 0.10 * time_pressure
            )

            tangent_weight = (
                0.55
                - 0.05 * time_pressure
            )

            escape_weight = (
                1.0
                - goal_weight
                - tangent_weight
            )

        vector = (
            goal_weight
            * self._heading_vector(
                goal_heading
            )
            + tangent_weight
            * self._heading_vector(
                tangent_heading
            )
            + escape_weight
            * self._heading_vector(
                escape_heading
            )
        )

        return self._vector_heading(
            vector,
            tangent_heading,
        )

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

        (
            remaining_time,
            time_to_goal,
            time_slack,
            time_pressure,
        ) = self._get_time_state(
            own_idx
        )

        (
            dynamic_buffer,
            trigger_distance,
            release_distance,
        ) = self._get_dynamic_buffer(
            time_pressure
        )

        self.last_remaining_time = remaining_time
        self.last_time_to_goal = time_to_goal
        self.last_time_slack = time_slack
        self.last_time_pressure = time_pressure
        self.last_dynamic_buffer = dynamic_buffer

        flight_time = self.metrics[
            self.agent
        ]["flight_time"]

        goal_heading, _ = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        (
            conflict_heading,
            conflict_risk,
        ) = self._get_conflict_state(
            own_idx
        )

        # =====================================================
        # 1. EMERGENCY CONFLICT
        # =====================================================

        if (
            conflict_heading is not None
            and conflict_risk
            >= self.conflict_emergency_risk
        ):

            return (
                conflict_heading,
                3,
            )

        # =====================================================
        # 2. PHYSICALLY INSIDE RESTRICTED AREA
        # =====================================================

        for obstacle in obstacle_states:

            if obstacle["inside"]:

                self.active_buffer_obstacle = (
                    obstacle["index"]
                )

                return (
                    self._get_buffer_avoidance_heading(
                        own_idx,
                        obstacle,
                    ),
                    1,
                )

        # =====================================================
        # 3. PREDICTED AIRCRAFT CONFLICT
        # =====================================================

        # React before obstacle navigation and before goal rescue.
        if (
            conflict_heading is not None
            and conflict_risk
            >= self.conflict_activation_risk
        ):

            return (
                conflict_heading,
                3,
            )

        # =====================================================
        # 4. DIRECT-GOAL RESCUE
        # =====================================================

        hard_deadline = (
            remaining_time
            <= time_to_goal
            + self.hard_deadline_margin_s
        )

        stuck_rescue = (
            self.active_buffer_obstacle
            is not None
            and self.stuck_counter
            >= self.stuck_direct_steps
        )

        if stuck_rescue:

            self.goal_rescue_until = max(
                self.goal_rescue_until,
                flight_time
                + self.goal_rescue_duration_s,
            )

        # Do not use stuck rescue while extremely close
        # to an obstacle unless the hard deadline is active.
        nearest_clearance = min(
            obstacle["clearance"]
            for obstacle in obstacle_states
        )

        rescue_safe_enough = (
            nearest_clearance
            > self.min_restricted_buffer_nm
        )

        if hard_deadline:

            return (
                goal_heading,
                4,
            )

        if (
            flight_time
            < self.goal_rescue_until
            and rescue_safe_enough
        ):

            return (
                goal_heading,
                4,
            )

        # =====================================================
        # 5. CONTINUE ACTIVE OBSTACLE AVOIDANCE
        # =====================================================

        if (
            self.active_buffer_obstacle
            is not None
        ):

            active_obstacle = None

            for obstacle in obstacle_states:

                if (
                    obstacle["index"]
                    == self.active_buffer_obstacle
                ):

                    active_obstacle = obstacle
                    break

            if (
                active_obstacle is not None
                and active_obstacle[
                    "clearance"
                ] < release_distance
            ):

                return (
                    self._get_goal_biased_avoidance_heading(
                        own_idx,
                        active_obstacle,
                        time_pressure,
                    ),
                    1,
                )

            self.active_buffer_obstacle = None
            self.avoidance_side = 0
            self.avoidance_side_obstacle = None
            self.stuck_side_switched = False

        # =====================================================
        # 6. START OBSTACLE AVOIDANCE EARLY
        # =====================================================

        nearest_obstacle = min(
            obstacle_states,
            key=lambda item:
                item["clearance"],
        )

        if (
            nearest_obstacle["clearance"]
            < trigger_distance
        ):

            self.active_buffer_obstacle = (
                nearest_obstacle["index"]
            )

            return (
                self._get_goal_biased_avoidance_heading(
                    own_idx,
                    nearest_obstacle,
                    time_pressure,
                ),
                1,
            )

        # =====================================================
        # 7. SECTOR
        # =====================================================

        sector_state = (
            self._get_sector_state(
                own_idx
            )
        )

        if not sector_state["inside"]:

            if (
                sector_state["clearance"]
                < self.max_sector_excursion_nm
            ):

                return (
                    goal_heading,
                    0,
                )

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

        # =====================================================
        # 8. NORMAL GOAL NAVIGATION
        # =====================================================

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

        (
            goal_heading,
            goal_distance,
        ) = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
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
                3.0,
            )
        ])

        (
            remaining_time_s,
            time_to_goal_s,
            time_slack_s,
            _,
        ) = self._get_time_state(
            own_idx
        )

        remaining_time_obs = np.array([
            np.clip(
                remaining_time_s
                / self.episode_time_limit,
                0.0,
                1.0,
            )
        ])

        time_to_goal_obs = np.array([
            np.clip(
                time_to_goal_s
                / self.episode_time_limit,
                0.0,
                2.0,
            )
        ])

        time_slack_obs = np.array([
            np.clip(
                time_slack_s
                / self.episode_time_limit,
                -1.0,
                1.0,
            )
        ])

        own_speed_knots = (
            bs.traf.cas[own_idx]
            * MPS_TO_KT
        )

        own_speed = np.array([
            np.clip(
                own_speed_knots / 250.0,
                0.0,
                2.0,
            )
        ])

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

            obstacle_features[slot] = np.array([
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
                    3.0,
                ),

                np.clip(
                    obstacle[
                        "radius"
                    ] / 20.0,
                    0.0,
                    3.0,
                ),

                1.0
                if obstacle["inside"]
                else 0.0,
            ])

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
                3.0,
            ),

            1.0
            if sector_state["inside"]
            else 0.0,
        ])

        _, navigation_mode = (
            self._get_navigation_reference(
                own_idx
            )
        )

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

            "remaining_time":
                remaining_time_obs,

            "time_to_goal":
                time_to_goal_obs,

            "time_slack":
                time_slack_obs,

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

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
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

        # -----------------------------------------------------
        # Policy freedom by navigation mode
        # -----------------------------------------------------

        if navigation_mode == 1:
            heading_limit = 8.0

        elif navigation_mode == 2:
            heading_limit = 20.0

        elif navigation_mode == 3:
            heading_limit = 8.0

        elif navigation_mode == 4:
            heading_limit = 10.0

        else:
            heading_limit = (
                self.max_heading_offset
            )

        heading_offset = (
            action[0]
            * heading_limit
        )

        target_heading = (
            self._wrap_angle(
                reference_heading
                + heading_offset
            )
        )

        speed_fraction = (
            action[1] + 1.0
        ) / 2.0

        target_speed = (
            self.min_target_speed
            + speed_fraction
            * (
                self.max_target_speed
                - self.min_target_speed
            )
        )

        (
            _,
            _,
            _,
            time_pressure,
        ) = self._get_time_state(
            own_idx
        )

        # -----------------------------------------------------
        # Speed management
        # -----------------------------------------------------

        minimum_urgent_speed = (
            220.0
            + 70.0 * time_pressure
        )

        target_speed = max(
            target_speed,
            minimum_urgent_speed,
        )

        # Do not force maximum speed during conflict resolution.
        # A moderate speed gives the aircraft more time to maneuver.
        if navigation_mode == 3:

            target_speed = np.clip(
                target_speed,
                200.0,
                250.0,
            )

        elif navigation_mode == 4:

            target_speed = max(
                target_speed,
                290.0,
            )

        target_speed = np.clip(
            target_speed,
            self.min_target_speed,
            self.max_target_speed,
        )

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

        (
            remaining_time,
            _,
            _,
            time_pressure,
        ) = self._get_time_state(
            own_idx
        )

        (
            dynamic_buffer,
            _,
            _,
        ) = self._get_dynamic_buffer(
            time_pressure
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
            len(inside_obstacles) > 0
        )

        nearest_clearance = min(
            obstacle["clearance"]
            for obstacle
            in obstacle_states
        )

        # =====================================================
        # 1. GOAL
        # =====================================================

        _, goal_distance = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            self.goal_lat,
            self.goal_lon,
        )

        if (
            self.previous_goal_distance
            is not None
        ):

            progress = (
                self.previous_goal_distance
                - goal_distance
            )

            if (
                progress
                < self.stuck_progress_threshold_nm
            ):

                self.stuck_counter += 1

            else:

                self.stuck_counter = 0
                self.stuck_side_switched = False

            # Preserve strong goal motivation.
            progress_weight = (
                6.0
                + 12.0 * time_pressure
            )

            away_weight = (
                0.5
                + 3.0 * time_pressure
            )

            if progress >= 0.0:

                reward += (
                    progress_weight
                    * progress
                )

            else:

                reward += (
                    away_weight
                    * progress
                )

        self.previous_goal_distance = (
            goal_distance
        )

        reached = (
            metrics[
                "waypoint_reached"
            ]
            - previous[
                "waypoint_reached"
            ]
        )

        if reached > 0:
            reward += 600.0

        if (
            remaining_time <= 5.0
            and metrics[
                "waypoint_reached"
            ] == 0
            and not
            self.failure_penalty_applied
        ):

            reward -= 600.0

            self.failure_penalty_applied = (
                True
            )

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

        # Entering a restricted area is now very expensive.
        reward -= (
            55.0
            * restricted_events
        )

        # Remaining inside is also strongly discouraged.
        reward -= (
            3.5
            * restricted_time
        )

        if inside_restricted:

            current_escape_clearance = min(
                obstacle["clearance"]
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

                reward += (
                    15.0
                    * escape_progress
                )

            self.previous_escape_clearance = (
                current_escape_clearance
            )

        else:

            self.previous_escape_clearance = (
                None
            )

        if (
            self.was_inside_restricted
            and not inside_restricted
        ):

            reward += 30.0

        self.was_inside_restricted = (
            inside_restricted
        )

        # -----------------------------------------------------
        # Proximity shaping
        # -----------------------------------------------------

        if (
            not inside_restricted
            and nearest_clearance < 9.0
        ):

            proximity_risk = (
                1.0
                - nearest_clearance / 9.0
            )

            reward -= (
                0.30
                * proximity_risk
                * elapsed_time
            )

        if (
            not inside_restricted
            and nearest_clearance
            < dynamic_buffer
        ):

            buffer_risk = (
                1.0
                - nearest_clearance
                / max(
                    dynamic_buffer,
                    1e-6,
                )
            )

            reward -= (
                2.5
                * buffer_risk
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

        # Prevent entering an intrusion.
        reward -= (
            90.0
            * intrusion_events
        )

        # Leave existing intrusions quickly.
        reward -= (
            5.0
            * intrusion_time
        )

        highest_risk = 0.0
        nearest_intruder_distance = np.inf

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
                distance,
            ) = self._compute_cpa(
                own_idx,
                idx,
            )

            nearest_intruder_distance = min(
                nearest_intruder_distance,
                distance,
            )

            if distance < 5.0:

                risk = 1.0

            elif (
                0.0
                < tcpa
                < self.conflict_prediction_time_s
                and dcpa
                < self.conflict_prediction_dcpa_nm
            ):

                time_risk = (
                    1.0
                    - tcpa
                    / self.conflict_prediction_time_s
                )

                separation_risk = (
                    1.0
                    - dcpa
                    / self.conflict_prediction_dcpa_nm
                )

                risk = (
                    time_risk
                    * separation_risk
                )

            else:

                risk = 0.0

            highest_risk = max(
                highest_risk,
                risk,
            )

        # Strong dense penalty before the official intrusion.
        reward -= (
            3.0
            * highest_risk
            * elapsed_time
        )

        # Additional current-distance safety margin.
        if (
            nearest_intruder_distance
            < 8.0
        ):

            distance_risk = (
                1.0
                - nearest_intruder_distance
                / 8.0
            )

            reward -= (
                1.5
                * distance_risk
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

        # Keep sector penalties moderate.
        if (
            highest_risk
            >= self.conflict_activation_risk
        ):

            sector_event_weight = 0.5
            outside_time_weight = 0.02

        else:

            sector_event_weight = 2.0
            outside_time_weight = 0.08

        reward -= (
            sector_event_weight
            * sector_exit_events
        )

        reward -= (
            outside_time_weight
            * outside_time
        )

        # =====================================================
        # 5. FLIGHT TIME
        # =====================================================

        reward -= (
            0.01
            * elapsed_time
        )

        speed_target = (
            240.0
            + 50.0 * time_pressure
        )

        # Do not apply the high-speed reward during conflicts.
        if (
            self.last_navigation_mode != 3
            and self.last_target_speed
            < speed_target
        ):

            speed_deficit = (
                speed_target
                - self.last_target_speed
            ) / 100.0

            reward -= (
                0.03
                * speed_deficit
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
            0.01
            * action_change
        )

        return float(reward)