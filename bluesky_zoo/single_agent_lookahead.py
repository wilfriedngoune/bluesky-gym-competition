"""PPO-guided lookahead for the shared single-agent navigation contract.

Search uses the same bounded PUCT, learned Gaussian priors and learned critic
as the MA adapter. Only current horizontal traffic state, static geometry, the
own goal and copied navigation history are read. Scripted intruders retain
their observed heading/TAS in forecasts: their future route is deliberately
unknown. Live BlueSky dynamics, commands, RNGs and scoring are never modified.

An edge holds an absolute command, using the existing approximate turn/speed
predictor. Rewards and navigation memory advance at the real decision cadence
within that edge; safety occupancy is sampled every second. Replanning after
each live decision limits (but does not eliminate) motion-model errors.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from bluesky_gym.envs.common import functions as fn
from bluesky_gym.envs.common.decision_timing import forecast_segment_seconds
from bluesky_zoo.lookahead import AircraftState, Transition, policy_value_search
from bluesky_zoo.lookahead_policy import (
    ForecastAircraft, ForecastScene, LookaheadPolicy, _cpa, _heading, _position, _wrap,
)
from bluesky_zoo.navigation_reward import (
    aircraft_buffer_risk, predicted_conflict_risk, reward_components, sector_buffer_risk,
)
from bluesky_zoo.strategic_navigation import zone_penalty
from bluesky_zoo.transfer_navigation import SingleAgentTransferEnv, deadline_features


@dataclass(frozen=True)
class NavigationMemory:
    progress_anchor_time: float | None
    progress_anchor_distance: float | None
    recent_progress_rate: float
    stalled_seconds: float
    avoidance_obstacle: int | None
    avoidance_side: int
    avoidance_started: float
    avoidance_switched: bool


@dataclass(frozen=True)
class TransferForecastAircraft(ForecastAircraft):
    memory: NavigationMemory | None = None


class SingleAgentLookaheadPolicy(LookaheadPolicy):
    """Plan only the learning aircraft; scripted traffic is never controlled."""

    checkpoint_mdp = "navigation-transfer"
    traffic_slots = 10

    def __init__(self, model, env, config):
        if not isinstance(env, SingleAgentTransferEnv):
            raise ValueError("Single-agent lookahead requires SingleAgentTransferEnv")
        if config.safety_filter:
            raise ValueError("Single-agent lookahead does not support the optional safety filter")
        if config.coordination_passes != 1:
            raise ValueError("Single-agent lookahead requires one coordination pass")
        super().__init__(model, env, config)
        self.transfer_config = env.transfer_config
        self.lookahead_config = config

    def snapshot(self):
        """Read observed motion and own history, never scripted route targets."""
        import bluesky as bs

        self._geometry()
        self._reset_forecast_cache()
        center = np.array(self.env.scenario.center)
        state = self.env.navigation_state
        if state.zone_schedule is None:
            raise RuntimeError("Reset the single-agent environment before requesting lookahead")
        own_id = self.env.agent
        aircraft = []
        memory = NavigationMemory(**{name: getattr(state, name)
                                     for name in NavigationMemory.__dataclass_fields__})
        for agent in [own_id, *self.env.intruder_ids]:
            idx = bs.traf.id2idx(agent)
            if idx < 0:
                if agent == own_id:
                    raise RuntimeError("The learning aircraft is absent from the live simulator")
                continue
            position = fn.latlong_to_nm(center, np.array([bs.traf.lat[idx], bs.traf.lon[idx]]))
            cas = float(bs.traf.cas[idx]) * 1.94384
            tas = float(bs.traf.tas[idx]) * 1.94384
            motion = AircraftState(float(position[0]), float(position[1]), float(bs.traf.hdg[idx]), tas)
            if agent == own_id:
                goal = tuple(fn.latlong_to_nm(center, np.array([self.env.goal_lat, self.env.goal_lon])))
                # The live anchor uses kwikqdrdist, while copied motion uses
                # local planar distances. Translate the current observation's
                # anchor into that same plane; mixing the two metrics can
                # dwarf five seconds of real progress and reverse stall state.
                if memory.progress_anchor_time == self.env.sim_time:
                    memory = replace(memory, progress_anchor_distance=float(
                        np.linalg.norm(np.asarray(goal) - position)))
                aircraft.append(TransferForecastAircraft(
                    agent, motion, goal, tuple(state.last_action), state.active_buffer_obstacle,
                    float(state.last_target_heading), float(state.last_target_speed),
                    bool(state._has_target_command), tas / max(cas, 1e-6), state.zone_schedule,
                    memory=memory,
                ))
            else:
                # No future route, goal, autopilot target or other policy read.
                aircraft.append(ForecastAircraft(
                    agent, motion, (), (), None, motion.heading_deg, cas,
                    False, tas / max(cas, 1e-6), None,
                ))
        return ForecastScene(tuple(aircraft), float(self.env.sim_time))

    def _refresh_memory(self, aircraft, time):
        memory = aircraft.memory
        distance = float(np.linalg.norm(np.asarray(aircraft.goal) - _position(aircraft.motion)))
        if memory.progress_anchor_time is not None and time > memory.progress_anchor_time:
            rate = (memory.progress_anchor_distance - distance) / (time - memory.progress_anchor_time)
            progressing = rate >= self.transfer_config.stagnant_rate_nm_s
            memory = replace(memory, recent_progress_rate=rate,
                             stalled_seconds=0. if progressing else memory.stalled_seconds + time - memory.progress_anchor_time,
                             avoidance_switched=False if progressing else memory.avoidance_switched)
        if memory.progress_anchor_time is None or time > memory.progress_anchor_time:
            memory = replace(memory, progress_anchor_time=time, progress_anchor_distance=distance)
        return replace(aircraft, memory=memory)

    def _navigation_reference(self, aircraft, time):
        """Pure counterpart of the transfer MDP's obstacle/sector/goal reference."""
        heading, mode, active = super()._reference(aircraft)
        aircraft = replace(aircraft, active_obstacle=active)
        memory = aircraft.memory
        if active is not None:
            _, inside, _ = self._area_state(_position(aircraft.motion), active + 1)
            if self.transfer_config.side_commitment and not inside:
                goal_heading = _heading(np.asarray(aircraft.goal) - _position(aircraft.motion))
                left, right = heading - 90., heading + 90.
                if memory.avoidance_obstacle != active:
                    side = -1 if abs(_wrap(left - goal_heading)) <= abs(_wrap(right - goal_heading)) else 1
                    memory = replace(memory, avoidance_obstacle=active, avoidance_side=side,
                                     avoidance_started=time, avoidance_switched=False)
                if (min(memory.stalled_seconds, time - memory.avoidance_started)
                        >= self.transfer_config.side_switch_after_s and not memory.avoidance_switched):
                    memory = replace(memory, avoidance_side=-memory.avoidance_side, avoidance_switched=True)
                tangent = left if memory.avoidance_side < 0 else right
                angles = np.radians([goal_heading, tangent, heading])
                weights = np.array([.15, .6, .25])
                north = float(np.dot(weights, np.cos(angles)))
                east = float(np.dot(weights, np.sin(angles)))
                if math.hypot(north, east) > 1e-12:
                    heading = math.degrees(math.atan2(east, north))
        else:
            # _TransferNavigation clears these fields at the end of _get_obs.
            memory = replace(memory, avoidance_obstacle=None, avoidance_side=0, avoidance_switched=False)
        return replace(aircraft, memory=memory), _wrap(heading), mode

    def _reference(self, aircraft):
        time = aircraft.memory.progress_anchor_time
        updated, heading, mode = self._navigation_reference(aircraft, 0. if time is None else time)
        return heading, mode, updated.active_obstacle

    def _command(self, aircraft, action):
        reference, _, active = self._reference(aircraft)
        heading = _wrap(reference + float(action[0]) * self.passage.max_heading_offset_deg)
        cfg = self.transfer_config
        speed = cfg.min_target_speed_kt + (float(action[1]) + 1.) / 2. * (cfg.max_target_speed_kt - cfg.min_target_speed_kt)
        if aircraft.has_command:
            if abs(_wrap(heading - aircraft.target_heading)) < self.command_config.heading_deadband_deg:
                heading = aircraft.target_heading
            if abs(speed - aircraft.target_speed) < self.command_config.speed_deadband_kt:
                speed = aircraft.target_speed
        return heading, speed, active

    def observation(self, scene, index=0):
        if index != 0:
            raise ValueError("Only the learning aircraft has a policy observation")
        aircraft = self._refresh_memory(scene.aircraft[0], scene.time)
        aircraft, _, _ = self._navigation_reference(aircraft, scene.time)
        normalized = replace(scene, aircraft=(aircraft, *scene.aircraft[1:]))
        obs = super().observation(normalized, 0)
        distance = float(np.linalg.norm(np.asarray(aircraft.goal) - _position(aircraft.motion)))
        estimate, slack = deadline_features(distance, self.env.distance_margin,
            self.transfer_config.max_target_speed_kt, scene.time, self.env.episode_time_limit)
        memory = aircraft.memory
        obs.update(time_to_goal=estimate, time_slack=slack, navigation_memory=np.array([
            float(memory.avoidance_side),
            np.clip(memory.stalled_seconds / self.env.episode_time_limit, 0., 1.),
            np.clip(memory.recent_progress_rate * 3600. / self.transfer_config.max_target_speed_kt, -3., 3.),
            float(memory.avoidance_switched),
        ]))
        return obs

    def transition(self, scene, index, action, opponent_commands=None, sample_observer=None):
        """Hold own command; integrate scored hazards at 1s, rewards per decision."""
        if index != 0:
            raise ValueError("Only the learning aircraft can be searched")
        duration = min(forecast_segment_seconds(self.config.rollout_seconds, self.env.action_frequency),
                       self.env.episode_time_limit - scene.time)
        if duration <= 0:
            return Transition(scene, 0., terminal=False, discount=1.)
        own, _, _ = self._navigation_reference(scene.aircraft[0], scene.time)
        heading, speed, _ = self._command(own, action)
        paths = [self._forecast_path(own, heading, speed, duration)]
        # Each intruder is propagated at current heading and TAS, without a
        # hidden turn command, speed target, capture radius or removal time.
        paths.extend(self._forecast_path(a, a.motion.heading_deg,
                                        a.motion.speed_kt / a.tas_cas_ratio, duration)
                     for a in scene.aircraft[1:])
        aircraft = [own, *scene.aircraft[1:]]
        previous_intrusions = {a.agent for a in aircraft[1:]
                               if _cpa(own.motion, a.motion)[2] < self.env.intrusion_distance}
        flags = self._inside_areas(_position(own.motion))
        previous_obstacles = {i for i in range(1, len(flags)) if flags[i]}
        previous_outside = not flags[0]
        before_distance = float(np.linalg.norm(np.asarray(own.goal) - _position(own.motion)))
        elapsed, interval_elapsed, reward, discount = 0., 0., 0., 1.
        action = np.asarray(action)
        previous_action = np.asarray(own.action)
        totals = dict(intrusion_events=0., intrusion_time=0., restricted_area_events=0.,
                      time_in_restricted_area=0., sector_exit_events=0., time_outside_sector=0.)
        for sample in range(len(paths[0])):
            dt = min(1., duration - elapsed)
            aircraft = [replace(a, motion=paths[i][sample]) for i, a in enumerate(aircraft)]
            current = aircraft[0]
            position = _position(current.motion)
            encounters = [_cpa(current.motion, a.motion) for a in aircraft[1:]]
            intrusions = {a.agent for a, encounter in zip(aircraft[1:], encounters)
                          if encounter[2] < self.env.intrusion_distance}
            flags = self._inside_areas(position)
            obstacles = {i for i in range(1, len(flags)) if flags[i]}
            outside = not flags[0]
            totals['intrusion_events'] += len(intrusions - previous_intrusions)
            totals['intrusion_time'] += len(intrusions) * dt
            totals['restricted_area_events'] += len(obstacles - previous_obstacles)
            totals['time_in_restricted_area'] += bool(obstacles) * dt
            totals['sector_exit_events'] += outside and not previous_outside
            totals['time_outside_sector'] += outside * dt
            elapsed += dt
            interval_elapsed += dt
            previous_intrusions, previous_obstacles, previous_outside = intrusions, obstacles, outside
            distance = float(np.linalg.norm(np.asarray(own.goal) - position))
            arrived = distance * 1.852 < self.env.distance_margin
            decision_end = (interval_elapsed >= self.env.action_frequency or arrived or elapsed >= duration)
            if decision_end:
                current = self._refresh_memory(current, scene.time + elapsed)
                current, _, _ = self._navigation_reference(current, scene.time + elapsed)
                current = replace(current, action=tuple(action), target_heading=heading,
                                  target_speed=speed, has_command=True, arrived=arrived)
                aircraft[0] = current
                cfg = self.reward_config
                risk = max((predicted_conflict_risk(*e, self.env.intrusion_distance, cfg)
                            for e in encounters), default=0.)
                buffer_risk = sum(aircraft_buffer_risk(e[2], self.env.intrusion_distance, cfg)
                                  for e in encounters)
                sector_risk = 0.
                if cfg.sector_buffer_second_cost:
                    clearance, inside, _ = self._area_state(position, 0)
                    sector_risk = sector_buffer_risk(clearance, inside, cfg)
                metrics = dict(totals, flight_time=interval_elapsed, waypoint_reached=int(arrived))
                components = reward_components(cfg, previous_distance_nm=before_distance,
                    distance_nm=distance, capture_radius_km=self.env.distance_margin,
                    metrics=metrics, baseline=dict.fromkeys(metrics, 0.), risk=risk,
                    action=action, previous_action=previous_action,
                    sector_risk=sector_risk, aircraft_risk=buffer_risk)
                step_reward = sum(components.values()) + zone_penalty(
                    distance, own.schedule.radius_nm(scene.time + elapsed), interval_elapsed, self.zone_config)
                reward += discount * step_reward
                discount *= self.model.gamma
                before_distance, previous_action, interval_elapsed = distance, action, 0.
                totals = dict.fromkeys(totals, 0.)
            if sample_observer is not None:
                sample_observer(ForecastScene(tuple(aircraft), scene.time + elapsed))
            if arrived:
                break
        return Transition(ForecastScene(tuple(aircraft), scene.time + elapsed), float(reward),
                          terminal=arrived, discount=discount)

    def action(self, observation):
        """Search from one copied snapshot and the exact current policy input."""
        scene = self.snapshot()
        result = policy_value_search(scene,
            evaluate=lambda candidate: self._expansion(candidate, 0,
                root_observation=observation if candidate is scene else None),
            transition=lambda candidate, action: self.transition(candidate, 0, action),
            config=self.config)
        self.last_diagnostics = dict(simulations=result.simulations, expanded_nodes=result.expanded_nodes,
            deepest_depth=result.deepest_depth, visit_counts=result.visit_counts.tolist())
        return np.array(result.action, copy=True)
