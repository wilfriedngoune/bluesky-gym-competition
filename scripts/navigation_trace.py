"""Optional replay diagnostics; does not modify simulator state or scoring."""

import csv
from pathlib import Path

import bluesky as bs
from core.tools import kwikqdrdist


class NavigationTrace:
    fields = ["episode_index", "agent_id", "time_before", "time_after", "lat_before", "lon_before",
              "heading_before_deg", "speed_before_kt", "goal_distance_before_nm", "nearest_obstacle",
              "nearest_clearance_nm", "inside_obstacle", "sector_clearance_nm", "inside_sector",
              "policy_heading", "policy_speed", "reference_heading_deg", "target_heading_deg",
              "target_speed_kt", "navigation_mode", "active_obstacle", "waypoint_reached",
              "intrusion_events", "restricted_area_events", "sector_exit_events",
              "intrusion_time", "time_in_restricted_area", "time_outside_sector",
              "previous_target_heading_deg", "previous_target_speed_kt",
              "target_heading_change_deg", "target_speed_change_kt"]

    def __init__(self, path=None):
        self.file = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()

    def before(self, env, agents):
        if self.file is None:
            return {}
        if not hasattr(env, "navigation_states"):
            raise ValueError("Trajectory logging requires a navigation MDP")
        rows = {}
        for agent in agents:
            state = env.navigation_states[agent]
            idx = bs.traf.id2idx(agent)
            obstacles = state._get_obstacle_states(idx)
            nearest = obstacles[0] if obstacles else None
            sector = state._get_sector_state(idx)
            _, goal_distance = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], state.goal_lat, state.goal_lon)
            rows[agent] = dict(
                agent_id=agent, time_before=env.sim_time, lat_before=bs.traf.lat[idx], lon_before=bs.traf.lon[idx],
                heading_before_deg=bs.traf.hdg[idx], speed_before_kt=bs.traf.cas[idx] * 1.94384,
                goal_distance_before_nm=goal_distance,
                nearest_obstacle=nearest["index"] if nearest else "",
                nearest_clearance_nm=nearest["clearance"] if nearest else "",
                inside_obstacle=any(obstacle["inside"] for obstacle in obstacles),
                sector_clearance_nm=sector["clearance"], inside_sector=sector["inside"],
                previous_target_heading_deg=state.last_target_heading if state._has_target_command else None,
                previous_target_speed_kt=state.last_target_speed if state._has_target_command else None,
            )
        return rows

    def after(self, env, rows, actions, infos, episode):
        if self.file is None:
            return
        for agent, row in rows.items():
            state = env.navigation_states[agent]
            info = infos[agent]
            previous_heading = row["previous_target_heading_deg"]
            previous_speed = row["previous_target_speed_kt"]
            self.writer.writerow({**row, "episode_index": episode, "time_after": info["flight_time"],
                "policy_heading": actions[agent][0], "policy_speed": actions[agent][1],
                "reference_heading_deg": state.last_reference_heading,
                "target_heading_deg": state.last_target_heading, "target_speed_kt": state.last_target_speed,
                "navigation_mode": state.last_navigation_mode, "active_obstacle": state.last_command_obstacle,
                "target_heading_change_deg": (
                    (state.last_target_heading - previous_heading + 180) % 360 - 180
                    if previous_heading is not None else None),
                "target_speed_change_kt": state.last_target_speed - previous_speed if previous_speed is not None else None,
                **{key: info[key] for key in ("waypoint_reached", "intrusion_events",
                                               "restricted_area_events", "sector_exit_events",
                                               "intrusion_time", "time_in_restricted_area", "time_outside_sector")}})
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
