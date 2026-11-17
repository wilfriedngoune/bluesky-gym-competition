"""Optional copied-forecast diagnostics; never changes actions or scored info."""

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import bluesky as bs
import numpy as np

from bluesky_gym.envs.common import functions as fn
from bluesky_gym.envs.common.decision_timing import forecast_segment_seconds
from bluesky_zoo.lookahead_policy import _cpa


class SearchTrace:
    candidate_fields = [
        "episode_index", "time", "root_agent", "candidate_index", "selected", "action_heading", "action_speed",
        "target_heading_deg", "target_speed_kt", "actor_prior", "visits", "mean_return", "root_value",
        "first_reward", "first_discount", "first_leaf_value", "explored_depth", "principal_path_depth",
        "forecast_duration_seconds",
        "coordination_pass", "simulations_this_pass", "simulations_total",
        "safety_filter_enabled", "safety_min_separation_nm", "safety_intrusion_deficit_nm_seconds",
        "safety_buffer_deficit_nm_seconds", "safety_forecast_seconds", "safety_eligible",
    ]
    forecast_fields = [
        "episode_index", "decision_time", "root_agent", "candidate_index", "other_agent", "segment_index",
        "time", "offset_seconds", "own_north_nm", "own_east_nm", "other_north_nm", "other_east_nm",
        "predicted_separation_nm", "predicted_tcpa_seconds", "predicted_dcpa_nm",
        "target_heading_deg", "target_speed_kt", "assumed_other_heading_deg", "assumed_other_speed_kt",
    ]
    actual_fields = [
        "episode_index", "time", "own_agent", "other_agent", "own_lat", "own_lon", "other_lat", "other_lon",
        "own_north_nm", "own_east_nm", "other_north_nm", "other_east_nm", "own_heading_deg", "other_heading_deg",
        "own_tas_kt", "other_tas_kt", "separation_nm", "tcpa_seconds", "dcpa_nm",
    ]
    command_fields = [
        "episode_index", "time", "agent", "action_heading", "action_speed", "target_heading_deg",
        "target_speed_kt", "reference_heading_deg", "navigation_mode",
    ]

    def __init__(self, prefix, env, act, *, model_path, seed, scenario_start,
                 start=40., end=400., agents=None, neighbours=None):
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
            raise ValueError("Search trace window must be finite with 0 <= start <= end")
        self.prefix, self.env, self.start, self.end = Path(prefix), env, float(start), float(end)
        self.agents = frozenset(agents or env.possible_agents)
        self.neighbours = frozenset(neighbours or env.possible_agents)
        unknown = (self.agents | self.neighbours) - set(env.possible_agents)
        if unknown:
            raise ValueError(f"Unknown search trace aircraft: {sorted(unknown)}")
        self.enabled = not getattr(act, "disable_lookahead", False)
        config = getattr(act, "lookahead_config", None)
        self.actual_end = self.end + (config.max_depth * forecast_segment_seconds(
            config.rollout_seconds, env.action_frequency) if config is not None else 0)
        self.episode = 0
        self._last_actual = None
        self.files, self.writers = {}, {}
        self.paths = {name: Path(str(self.prefix) + f".{name}.csv")
                      for name in ("candidates", "forecasts", "actual", "commands")}
        self.settings_path = Path(str(self.prefix) + ".settings.json")
        for path in [*self.paths.values(), self.settings_path]:
            if path.exists():
                raise FileExistsError(f"Diagnostic file already exists: {path}; choose another --search-trace-prefix")
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        source = Path(model_path)
        if not source.is_file():
            source = Path(str(source) + ".zip")
        self.settings = dict(
            model=str(source.resolve()), model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            seed=seed, scenario_start=scenario_start, window_start=self.start, window_end=self.end,
            actual_window_end=self.actual_end, root_agents=sorted(self.agents), neighbours=sorted(self.neighbours),
            lookahead_enabled=self.enabled, lookahead_config=asdict(config) if config is not None else None,
            decision_seconds=env.action_frequency, completed=False,
            files={name: str(path) for name, path in self.paths.items()},
            semantics=("Forecasts are the completed search tree's most-visited continuations for each root candidate, "
                       "not extra rollouts. Only the final coordination pass is logged. Other aircraft follow "
                       "root actor commands in pass one, or the previous pass's joint proposals in pass two. "
                       "Actual encounters are sampled at live decision boundaries; official scoring stays at 1 second. "
                       "Predicted geometry uses local NM; actual separation/CPA use the existing geodesic helper. "
                       "No simulation, scoring, search objective, budget or policy is overridden."),
        )
        try:
            for name, path in self.paths.items():
                file = path.open("x", newline="", encoding="utf-8")
                self.files[name] = file
                writer = csv.DictWriter(file, fieldnames=getattr(self, name[:-1] + "_fields")
                                        if name in {"candidates", "forecasts", "commands"} else self.actual_fields)
                self.writers[name] = writer
                writer.writeheader()
            self._save_settings()
        except Exception:
            for file in self.files.values():
                file.close()
            raise

    def _save_settings(self):
        self.settings_path.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")

    def begin_episode(self, episode):
        self.episode, self._last_actual = episode, None

    def wants(self, agent, sim_time):
        return self.enabled and agent in self.agents and self.start <= sim_time <= self.end

    def record_search(self, planner, scene, index, diagnostics, samples, opponent_commands):
        own = scene.aircraft[index]
        safety = planner.last_safety_diagnostics.get(own.agent)
        for candidate in diagnostics.candidates:
            profile = safety["candidates"][candidate.index] if safety else None
            heading, speed, _ = planner._command(own, candidate.action)
            path = candidate.principal_path
            duration = path[-1].state_after.time - scene.time if path else 0.
            self.writers["candidates"].writerow(dict(
                episode_index=self.episode, time=scene.time, root_agent=own.agent,
                candidate_index=candidate.index, selected=candidate.index == diagnostics.selected_index,
                action_heading=candidate.action[0], action_speed=candidate.action[1],
                target_heading_deg=heading, target_speed_kt=speed, actor_prior=candidate.prior,
                visits=candidate.visits, mean_return=candidate.mean_return, root_value=diagnostics.root_value,
                first_reward=candidate.first_reward, first_discount=candidate.first_discount,
                first_leaf_value=candidate.first_leaf_value, explored_depth=candidate.explored_depth,
                principal_path_depth=len(path), forecast_duration_seconds=duration,
                coordination_pass=planner.active_pass + 1, simulations_this_pass=planner.active_pass_budget,
                simulations_total=planner.config.simulations,
                safety_filter_enabled=profile is not None,
                safety_min_separation_nm=profile.minimum_nm if profile else None,
                safety_intrusion_deficit_nm_seconds=profile.intrusion_deficit_nm_seconds if profile else None,
                safety_buffer_deficit_nm_seconds=profile.buffer_deficit_nm_seconds if profile else None,
                safety_forecast_seconds=profile.duration_seconds if profile else None,
                safety_eligible=profile.minimum_nm >= safety["threshold_nm"] if profile else None,
            ))
            for segment, step in enumerate(path):
                heading, speed, _ = planner._command(step.state_before.aircraft[index], step.action)
                for frame in samples.get((id(step.state_before), step.action), ()):
                    current = frame.aircraft[index]
                    for other in frame.aircraft:
                        if other.agent == own.agent or other.agent not in self.neighbours or other.arrived:
                            continue
                        tcpa, dcpa, distance = _cpa(current.motion, other.motion)
                        self.writers["forecasts"].writerow(dict(
                            episode_index=self.episode, decision_time=scene.time, root_agent=own.agent,
                            candidate_index=candidate.index, other_agent=other.agent, segment_index=segment,
                            time=frame.time, offset_seconds=frame.time - scene.time,
                            own_north_nm=current.motion.north_nm, own_east_nm=current.motion.east_nm,
                            other_north_nm=other.motion.north_nm, other_east_nm=other.motion.east_nm,
                            predicted_separation_nm=distance, predicted_tcpa_seconds=tcpa, predicted_dcpa_nm=dcpa,
                            target_heading_deg=heading, target_speed_kt=speed,
                            assumed_other_heading_deg=opponent_commands[other.agent][0],
                            assumed_other_speed_kt=opponent_commands[other.agent][1],
                        ))

    def capture_actual(self):
        env = self.env
        now = float(env.sim_time)
        key = self.episode, now
        if key == self._last_actual or not self.start <= now <= self.actual_end:
            return
        self._last_actual = key
        center = np.array(env.scenario.center)
        for own in env.agents:
            if own not in self.agents:
                continue
            i = bs.traf.id2idx(own)
            own_xy = fn.latlong_to_nm(center, np.array([bs.traf.lat[i], bs.traf.lon[i]]))
            for other in env.agents:
                if other == own or other not in self.neighbours:
                    continue
                j = bs.traf.id2idx(other)
                tcpa, dcpa, distance = env.navigation_states[own]._compute_cpa(i, j)
                other_xy = fn.latlong_to_nm(center, np.array([bs.traf.lat[j], bs.traf.lon[j]]))
                self.writers["actual"].writerow(dict(
                    episode_index=self.episode, time=now, own_agent=own, other_agent=other,
                    own_lat=bs.traf.lat[i], own_lon=bs.traf.lon[i], other_lat=bs.traf.lat[j], other_lon=bs.traf.lon[j],
                    own_north_nm=own_xy[0], own_east_nm=own_xy[1], other_north_nm=other_xy[0], other_east_nm=other_xy[1],
                    own_heading_deg=bs.traf.hdg[i], other_heading_deg=bs.traf.hdg[j],
                    own_tas_kt=bs.traf.tas[i] * 1.94384, other_tas_kt=bs.traf.tas[j] * 1.94384,
                    separation_nm=distance, tcpa_seconds=tcpa, dcpa_nm=dcpa,
                ))

    def record_commands(self, sim_time, actions):
        if not self.start <= sim_time <= self.actual_end:
            return
        for agent, action in actions.items():
            if agent not in self.agents | self.neighbours:
                continue
            state = self.env.navigation_states[agent]
            self.writers["commands"].writerow(dict(
                episode_index=self.episode, time=sim_time, agent=agent,
                action_heading=action[0], action_speed=action[1], target_heading_deg=state.last_target_heading,
                target_speed_kt=state.last_target_speed, reference_heading_deg=state.last_reference_heading,
                navigation_mode=state.last_navigation_mode,
            ))
        for file in self.files.values():
            file.flush()

    def close(self, completed=False):
        for file in self.files.values():
            file.close()
        self.settings["completed"] = bool(completed)
        self.settings["recorded_until_seconds"] = float(self.env.sim_time)
        self.settings["requested_window_covered"] = self.env.sim_time >= self.end
        self.settings["maximum_forecast_window_covered"] = self.env.sim_time >= self.actual_end
        self._save_settings()
