"""RL-guided search using copied horizontal state and an approximate predictor.

BlueSky remains the authoritative simulator. This module never steps, resets,
commands or restores it. PPO learns on its own sampled actions; this optional
inference adapter uses the learned Gaussian policy as search priors and the
learned critic at leaves. Other aircraft follow their snapshot policy commands
in forecasts, so predictions are not promises of conflict-free flight.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from shapely import intersects_xy, prepare
from shapely.geometry import Point, Polygon

from bluesky_gym.envs.common import functions as fn
from bluesky_gym.envs.common.decision_timing import forecast_segment_seconds
from bluesky_zoo.lookahead import (
    AircraftState, PolicyExpansion, Transition, gaussian_action_candidates,
    policy_value_search, predict_aircraft,
)
from bluesky_zoo.navigation_reward import (
    aircraft_buffer_risk, distance_potential, predicted_conflict_risk, sector_buffer_risk,
)
from bluesky_zoo.passage_navigation import RAY_OFFSETS_DEG, ray_distances
from bluesky_zoo.strategic_navigation import zone_penalty
from bluesky_zoo.separation_filter import SeparationForecast, select_separation_candidate


def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def _position(motion):
    return np.array([motion.north_nm, motion.east_nm])


def _heading(vector):
    return math.degrees(math.atan2(vector[1], vector[0])) % 360.0


def _body(vector, heading, scale):
    angle = math.radians(heading)
    return np.clip(np.array([
        vector[0] * math.cos(angle) + vector[1] * math.sin(angle),
        -vector[0] * math.sin(angle) + vector[1] * math.cos(angle),
    ]) / scale, -3.0, 3.0)


def _velocity(motion):
    angle = math.radians(motion.heading_deg)
    return motion.speed_kt / 3600.0 * np.array([math.cos(angle), math.sin(angle)])


def _cpa(own, other):
    relative = _position(other) - _position(own)
    velocity = _velocity(other) - _velocity(own)
    distance = float(np.linalg.norm(relative))
    squared = float(np.dot(velocity, velocity))
    tcpa = -float(np.dot(relative, velocity)) / squared if squared >= 1e-10 else 600.0
    if tcpa < 0:
        tcpa = 600.0
    dcpa = distance if tcpa == 600.0 else float(np.linalg.norm(relative + tcpa * velocity))
    return tcpa, dcpa, distance


@dataclass(frozen=True)
class ForecastAircraft:
    agent: str
    motion: AircraftState
    goal: tuple
    action: tuple
    active_obstacle: object
    target_heading: float
    target_speed: float
    has_command: bool
    tas_cas_ratio: float
    schedule: object
    arrived: bool = False


@dataclass(frozen=True)
class ForecastScene:
    aircraft: tuple
    time: float


class LookaheadPolicy:
    """A bounded PUCT search for each live aircraft, from one common snapshot."""

    checkpoint_mdp = "navigation-v5"
    traffic_slots = 9

    def __init__(self, model, env, config):
        if "centralized_state" in model.observation_space.spaces:
            raise ValueError("v5 lookahead currently requires a decentralized policy/value checkpoint")
        if getattr(model, "competition_mdp", None) != self.checkpoint_mdp:
            raise ValueError(f"Lookahead requires a learned {self.checkpoint_mdp} checkpoint")
        if not math.isclose(config.discount, model.gamma, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Lookahead discount must match the learned PPO discount")
        self.model, self.env, self.config = model, env, config
        self.policy = model.policy
        self.reward_config = env.reward_config
        self.zone_config = env.zone_config
        self.passage = env.passage_config
        self.command_config = env.action_config
        self.last_diagnostics = {}
        self.last_safety_diagnostics = {}
        self.observer = None
        self._scenario = None
        self._reset_forecast_cache()

    def _reset_forecast_cache(self):
        # Exact copied-state results only; never reuse a plan across decisions.
        # Bounds prevent large user-selected search budgets growing memory
        # without limit. Clearing a cache affects work, not forecast results.
        self._area_cache = {}
        self._inside_cache = {}
        self._path_cache = {}

    def _geometry(self):
        if self._scenario is self.env.scenario:
            return
        self._scenario = self.env.scenario
        center = np.array(self.env.scenario.center)
        vertices = [np.array([fn.latlong_to_nm(center, np.array(v)) for v in polygon])
                    for polygon in [self.env.scenario.sector,
                                    *(o.vertices for o in self.env.scenario.obstacles)]]
        self.polygons = tuple(Polygon(v) for v in vertices)
        prepare(self.polygons)
        self._reset_forecast_cache()
        self.centers = tuple(v.mean(axis=0) for v in vertices)
        self.radii = tuple(float(np.linalg.norm(v - c, axis=1).max())
                           for v, c in zip(vertices, self.centers))
        self.starts = np.concatenate(vertices)
        self.ends = np.concatenate([np.roll(v, -1, axis=0) for v in vertices])

    def snapshot(self):
        """Copy only horizontal traffic, static geometry and navigation history."""
        import bluesky as bs
        self._geometry()
        self._reset_forecast_cache()
        center = np.array(self.env.scenario.center)
        aircraft = []
        for agent in self.env.agents:
            idx = bs.traf.id2idx(agent)
            state = self.env.navigation_states[agent]
            if state.zone_schedule is None:
                raise RuntimeError("Reset the v5 environment before requesting lookahead")
            position = fn.latlong_to_nm(center, np.array([bs.traf.lat[idx], bs.traf.lon[idx]]))
            goal = fn.latlong_to_nm(center, np.array(self.env._goal[agent]))
            cas = float(bs.traf.cas[idx]) * 1.94384
            tas = float(bs.traf.tas[idx]) * 1.94384
            aircraft.append(ForecastAircraft(
                agent, AircraftState(float(position[0]), float(position[1]), float(bs.traf.hdg[idx]), tas),
                tuple(goal), tuple(state.last_action), state.active_buffer_obstacle,
                float(state.last_target_heading), float(state.last_target_speed),
                bool(state._has_target_command), tas / max(cas, 1e-6), state.zone_schedule,
            ))
        return ForecastScene(tuple(aircraft), float(self.env.sim_time))

    def _area_state(self, position, index):
        key = (float(position[0]), float(position[1]), index)
        cached = self._area_cache.get(key)
        if cached is not None:
            return cached
        point = Point(position)
        polygon = self.polygons[index]
        nearest = polygon.exterior.interpolate(polygon.exterior.project(point))
        nearest_xy = np.array(nearest.coords[0])
        nearest_xy.setflags(write=False)
        result = (float(np.linalg.norm(position - nearest_xy)),
                  self._inside_areas(position)[index], nearest_xy)
        if len(self._area_cache) >= 4096:
            self._area_cache.clear()
        self._area_cache[key] = result
        return result

    def _inside_areas(self, position):
        """Polygon/point intersection includes edges, exactly like covers(Point).

        Reward occupancy needs these booleans, not nearest boundary points.
        The vectorized predicate avoids constructing/projecting six Points for
        every predicted second. Geometry belongs to this copied forecast only.
        """
        key = (float(position[0]), float(position[1]))
        cached = self._inside_cache.get(key)
        if cached is None:
            cached = tuple(bool(value) for value in intersects_xy(self.polygons, *key))
            if len(self._inside_cache) >= 8192:
                self._inside_cache.clear()
            self._inside_cache[key] = cached
        return cached

    def _forecast_path(self, aircraft, heading, speed, duration):
        target_speed = speed * aircraft.tas_cas_ratio
        key = (aircraft.motion, heading, target_speed, duration, self.config)
        cached = self._path_cache.get(key)
        if cached is None:
            cached = predict_aircraft(aircraft.motion, heading, target_speed, duration, self.config)
            if len(self._path_cache) >= 2048:
                self._path_cache.clear()
            self._path_cache[key] = cached
        return cached

    def _reference(self, aircraft):
        position = _position(aircraft.motion)
        obstacles = [self._area_state(position, i + 1) for i in range(len(self.polygons) - 1)]
        order = sorted(range(len(obstacles)), key=lambda i: obstacles[i][0])
        active = next((i for i in order if obstacles[i][1]), None)
        if active is None and aircraft.active_obstacle is not None:
            i = aircraft.active_obstacle
            if 0 <= i < len(obstacles) and obstacles[i][0] < self.passage.obstacle_release_nm:
                active = i
        if active is None and order and obstacles[order[0]][0] < self.passage.obstacle_buffer_nm:
            active = order[0]
        if active is not None:
            _, inside, nearest = obstacles[active]
            heading = _heading(nearest - position) + (0 if inside else 180)
            return _wrap(heading), 1, active
        _, inside, nearest = self._area_state(position, 0)
        if not inside:
            return _heading(nearest - position), 2, None
        return _heading(np.array(aircraft.goal) - position), 0, None

    def _command(self, aircraft, action):
        reference, _, active = self._reference(aircraft)
        heading = _wrap(reference + float(action[0]) * self.passage.max_heading_offset_deg)
        speed = 120.0 + (float(action[1]) + 1) * 50.0
        if aircraft.has_command:
            if abs(_wrap(heading - aircraft.target_heading)) < self.command_config.heading_deadband_deg:
                heading = aircraft.target_heading
            if abs(speed - aircraft.target_speed) < self.command_config.speed_deadband_kt:
                speed = aircraft.target_speed
        return heading, speed, active

    def observation(self, scene, index):
        """Construct future policy inputs without calling any live MDP hooks."""
        aircraft = scene.aircraft[index]
        motion = aircraft.motion
        position = _position(motion)
        goal_vector = np.array(aircraft.goal) - position
        distance = float(np.linalg.norm(goal_vector))
        goal_angle = math.radians(_wrap(_heading(goal_vector) - motion.heading_deg))
        obstacle_rows = []
        for i in range(1, len(self.polygons)):
            clearance, inside, nearest = self._area_state(position, i)
            obstacle_rows.append((clearance, np.r_[
                _body(nearest - position, motion.heading_deg, 50),
                _body(self.centers[i] - position, motion.heading_deg, 100),
                np.clip(clearance / 20, 0, 3), np.clip(self.radii[i] / 20, 0, 3), float(inside),
            ]))
        obstacle_rows.sort(key=lambda row: row[0])
        clearance, inside, nearest = self._area_state(position, 0)
        intruders = []
        for other in scene.aircraft:
            if other.agent == aircraft.agent or other.arrived:
                continue
            tcpa, dcpa, separation = _cpa(motion, other.motion)
            urgency = dcpa + .02 * tcpa if tcpa < 300 else 1000 + separation
            relative_velocity = (_velocity(other.motion) - _velocity(motion)) * 1852
            features = np.r_[
                _body(_position(other.motion) - position, motion.heading_deg, 100),
                _body(relative_velocity, motion.heading_deg, 150),
                np.clip(separation / 100, 0, 3), np.clip(tcpa / 300, 0, 2), np.clip(dcpa / 10, 0, 3),
            ]
            intruders.append(((separation >= self.env.intrusion_distance, urgency), features))
        intruders.sort(key=lambda row: row[0])
        traffic = np.zeros((self.traffic_slots, 7))
        mask = np.zeros(self.traffic_slots)
        for slot, (_, features) in enumerate(intruders[:self.traffic_slots]):
            traffic[slot], mask[slot] = features, 1
        _, mode, _ = self._reference(aircraft)
        obstacles = np.array([row for _, row in obstacle_rows]).reshape((-1, 7))
        rays = (ray_distances(position, motion.heading_deg + RAY_OFFSETS_DEG, self.starts, self.ends,
                              self.passage.lookahead_nm) / self.passage.lookahead_nm
                if inside and not np.any(obstacles[:, 6]) else np.zeros(len(RAY_OFFSETS_DEG)))
        return dict(
            goal_direction=np.array([math.cos(goal_angle), math.sin(goal_angle)]),
            goal_distance=np.array([np.clip(distance / 100, 0, 3)]),
            own_speed=np.array([np.clip(motion.speed_kt / aircraft.tas_cas_ratio / 200, 0, 2)]),
            navigation_mode=np.array([float(mode)]), obstacles=obstacles, intruders=traffic,
            sector=np.r_[_body(nearest - position, motion.heading_deg, 50), np.clip(clearance / 20, 0, 3), float(inside)],
            previous_action=np.array(aircraft.action), intruder_mask=mask,
            remaining_time=np.array([np.clip(1 - scene.time / self.env.episode_time_limit, 0, 1)]),
            clearance_rays=rays,
            zone=aircraft.schedule.observation(distance, scene.time, self.env.episode_time_limit),
        )

    def _expansion(self, scene, index, root_observation=None):
        aircraft = scene.aircraft[index]
        obs = root_observation if root_observation is not None else self.observation(scene, index)
        tensor, _ = self.policy.obs_to_tensor(obs)
        with torch.no_grad():
            distribution = self.policy.get_distribution(tensor).distribution
            mean = distribution.mean.cpu().numpy()[0]
            std = distribution.stddev.cpu().numpy()[0]
            value = float(self.policy.predict_values(tensor).cpu().item())
        actions, priors = gaussian_action_candidates(mean, std, count=self.config.candidate_count, seed=0)
        return PolicyExpansion(actions, priors, value, terminal=aircraft.arrived)

    def transition(self, scene, index, action, opponent_commands, sample_observer=None):
        """Forecast a held manoeuvre, sampling hazards at one-second intervals."""
        own = scene.aircraft[index]
        heading, speed, active = self._command(own, action)
        commands = dict(opponent_commands)
        commands[own.agent] = (heading, speed)
        duration = min(forecast_segment_seconds(self.config.rollout_seconds, self.env.action_frequency),
                       self.env.episode_time_limit - scene.time)
        if duration <= 0:
            # A time-limit boundary keeps its critic value, matching PPO's
            # truncation bootstrap. Only an actual goal has zero continuation.
            return Transition(scene, 0.0, terminal=False, discount=1.0)
        # Arrived forecast traffic stays frozen and is excluded from hazards;
        # no path is consumed for it by the guarded motion update below.
        paths = [None if a.arrived else self._forecast_path(a, *commands[a.agent], duration)
                 for a in scene.aircraft]
        aircraft = list(scene.aircraft)
        before_distance = float(np.linalg.norm(np.array(own.goal) - _position(own.motion)))
        previous_intrusions = {a.agent for a in aircraft if a.agent != own.agent and not a.arrived
                               and _cpa(own.motion, a.motion)[2] < self.env.intrusion_distance}
        previous_areas = self._inside_areas(_position(own.motion))
        previous_obstacles = {i for i in range(1, len(previous_areas)) if previous_areas[i]}
        previous_outside = not previous_areas[0]
        reward, elapsed = 0.0, 0.0
        cfg = self.reward_config
        for sample in range(len(paths[index])):
            dt = min(1.0, duration - elapsed)
            for i, a in enumerate(aircraft):
                if not a.arrived:
                    aircraft[i] = replace(a, motion=paths[i][sample])
            current = aircraft[index]
            position = _position(current.motion)
            # Each encounter is used for actual intrusion, CPA warning and the
            # aircraft buffer. Calculate its unchanged CPA once, not three times.
            encounters = [(a.agent, _cpa(current.motion, a.motion)) for a in aircraft
                          if a.agent != own.agent and not a.arrived]
            intrusions = {agent for agent, encounter in encounters
                          if encounter[2] < self.env.intrusion_distance}
            area_flags = self._inside_areas(position)
            occupied = {i for i in range(1, len(area_flags)) if area_flags[i]}
            inside = area_flags[0]
            outside = not inside
            risk = max((predicted_conflict_risk(*encounter, self.env.intrusion_distance, cfg)
                        for _, encounter in encounters), default=0.0)
            aircraft_risk = sum(aircraft_buffer_risk(encounter[2],
                                                    self.env.intrusion_distance, cfg)
                                for _, encounter in encounters)
            sector_risk = 0.0
            if cfg.sector_buffer_second_cost:
                clearance, _, _ = self._area_state(position, 0)
                sector_risk = sector_buffer_risk(clearance, inside, cfg)
            costs = (
                cfg.intrusion_event_cost * len(intrusions - previous_intrusions)
                + cfg.intrusion_second_cost * len(intrusions) * dt
                + cfg.restricted_event_cost * len(occupied - previous_obstacles)
                + cfg.restricted_second_cost * bool(occupied) * dt
                + cfg.sector_event_cost * (outside and not previous_outside)
                + cfg.sector_second_cost * outside * dt + cfg.flight_second_cost * dt
                + cfg.conflict_risk_second_cost * risk * dt
                + cfg.aircraft_buffer_second_cost * aircraft_risk * dt
                + cfg.sector_buffer_second_cost * sector_risk * dt
            )
            reward -= self.model.gamma ** (elapsed / self.env.action_frequency) * costs
            elapsed += dt
            if sample_observer is not None:
                # Immutable copied aircraft, sampled before forecast arrivals
                # are removed, matching this second's separation calculation.
                sample_observer(ForecastScene(tuple(aircraft), scene.time + elapsed))
            previous_intrusions, previous_obstacles, previous_outside = intrusions, occupied, outside
            # Remove forecast arrivals after this second's separation sampling.
            for i, a in enumerate(aircraft):
                if a.arrived:
                    continue
                distance = float(np.linalg.norm(np.array(a.goal) - _position(a.motion)))
                if distance * 1.852 < self.env.distance_margin:
                    aircraft[i] = replace(a, arrived=True)
            if aircraft[index].arrived:
                break
        after = aircraft[index]
        distance = float(np.linalg.norm(np.array(after.goal) - _position(after.motion)))
        discount = self.model.gamma ** (elapsed / self.env.action_frequency)
        reward += cfg.progress_weight * (
            discount * (0.0 if after.arrived else distance_potential(distance, self.env.distance_margin, cfg))
            - distance_potential(before_distance, self.env.distance_margin, cfg))
        reward += discount * cfg.goal_bonus * after.arrived
        reward += zone_penalty(distance, own.schedule.radius_nm(scene.time + elapsed), elapsed, self.zone_config)
        change = np.asarray(action) - np.array(own.action)
        reward -= cfg.action_change_cost * float(np.dot(change, change))
        aircraft[index] = replace(after, action=tuple(action), active_obstacle=active,
                                  target_heading=heading, target_speed=speed, has_command=True)
        return Transition(ForecastScene(tuple(aircraft), scene.time + elapsed), float(reward),
                          terminal=after.arrived, discount=discount)

    def separation_forecast(self, scene, index, action, opponent_commands):
        """Screen a held root command over one fixed, bounded horizon.

        Like the reward predictor, sample at <=1 second and stop counting an
        aircraft after its forecast arrival. No critic or live hooks are used.
        """
        own = scene.aircraft[index]
        commands = dict(opponent_commands)
        commands[own.agent] = self._command(own, action)[:2]
        duration = min(self.config.safety_horizon_seconds,
                       self.env.episode_time_limit - scene.time)
        if duration <= 0:
            return SeparationForecast(math.inf, 0., 0., 0.)
        paths = [self._forecast_path(a, *commands[a.agent], duration) for a in scene.aircraft]
        active = [not a.arrived for a in scene.aircraft]
        minimum, intrusion, buffer, elapsed = math.inf, 0., 0., 0.
        threshold = self.env.intrusion_distance + self.config.safety_buffer_nm
        for sample, motion in enumerate(paths[index]):
            dt = min(1., duration - elapsed)
            for other, path in enumerate(paths):
                if other == index or not active[other]:
                    continue
                distance = math.hypot(motion.north_nm - path[sample].north_nm,
                                      motion.east_nm - path[sample].east_nm)
                minimum = min(minimum, distance)
                intrusion += max(0., self.env.intrusion_distance - distance) * dt
                buffer += max(0., threshold - distance) * dt
            elapsed += dt
            for other, a in enumerate(scene.aircraft):
                position = paths[other][sample]
                if math.hypot(position.north_nm - a.goal[0], position.east_nm - a.goal[1]) * 1.852 < self.env.distance_margin:
                    active[other] = False
            if not active[index]:
                break
        return SeparationForecast(minimum, intrusion, buffer, elapsed)

    def actions(self, observations, agents):
        scene = self.snapshot()
        agents = list(agents)
        if set(agents) != {a.agent for a in scene.aircraft}:
            raise ValueError("Lookahead must plan all live aircraft from the same snapshot")
        batch = {key: np.stack([observations[a][key] for a in agents]) for key in observations[agents[0]]}
        baseline, _ = self.model.predict(batch, deterministic=True)
        baseline = dict(zip(agents, baseline))
        assumed_actions = baseline
        self.last_pass_diagnostics = []
        self.last_safety_diagnostics = {}
        budgets = [self.config.simulations // self.config.coordination_passes
                   + int(p < self.config.simulations % self.config.coordination_passes)
                   for p in range(self.config.coordination_passes)]
        expanded = dict.fromkeys(agents, 0)
        for pass_index, budget in enumerate(budgets):
            # Freeze one complete joint proposal for the entire pass. Aircraft
            # never see earlier aircraft's refinements from this same pass.
            opponent_commands = {a.agent: self._command(a, assumed_actions[a.agent])[:2]
                                 for a in scene.aircraft}
            results = {}
            pass_records = {}
            self.active_pass, self.active_pass_budget = pass_index, budget
            tree_config = replace(self.config, simulations=budget, coordination_passes=1)
            for index, a in enumerate(scene.aircraft):
                root = scene
                def evaluate(candidate):
                    return self._expansion(candidate, index,
                                           root_observation=observations[a.agent] if candidate is root else None)
                tracing = (pass_index == self.config.coordination_passes - 1
                           and self.observer is not None and self.observer.wants(a.agent, scene.time))
                samples = {} if tracing else None

                def transition(candidate, action):
                    if not tracing:
                        return self.transition(candidate, index, action, opponent_commands)
                    frames = []
                    predicted = self.transition(candidate, index, action, opponent_commands,
                                                sample_observer=frames.append)
                    samples[(id(candidate), tuple(float(v) for v in action))] = tuple(frames)
                    return predicted

                def observe(diagnostics):
                    self.observer.record_search(self, root, index, diagnostics, samples, opponent_commands)

                def select_action(actions, visits, values, priors):
                    profiles = tuple(self.separation_forecast(root, index, action, opponent_commands)
                                     for action in actions)
                    threshold = self.env.intrusion_distance + self.config.safety_buffer_nm
                    selected = select_separation_candidate(profiles, visits, values, priors, threshold)
                    original = max(range(len(visits)), key=lambda i: (visits[i], values[i], priors[i]))
                    self.last_safety_diagnostics[a.agent] = dict(
                        candidates=profiles, threshold_nm=threshold, selected_index=selected,
                        original_index=original, overridden=selected != original,
                        safe_candidates=sum(p.minimum_nm >= threshold for p in profiles),
                    )
                    return selected

                result = policy_value_search(
                    root, evaluate=evaluate, transition=transition, config=tree_config,
                    **({"observer": observe} if tracing else {}),
                    **({"action_selector": select_action} if self.config.safety_filter else {}),
                )
                results[a.agent] = np.array(result.action, copy=True)
                expanded[a.agent] += result.expanded_nodes
                pass_records[a.agent] = dict(action=results[a.agent].tolist(),
                                             assumed_commands=dict(opponent_commands), simulations=budget)
                self.last_diagnostics[a.agent] = dict(simulations=sum(budgets[:pass_index + 1]),
                                                      expanded_nodes=expanded[a.agent],
                                                      visit_counts=result.visit_counts.tolist())
                if self.config.coordination_passes > 1:
                    self.last_diagnostics[a.agent].update(coordination_passes=pass_index + 1,
                                                          simulations_per_pass=budgets[:pass_index + 1])
            self.last_pass_diagnostics.append(pass_records)
            assumed_actions = results
        return results
