# Competition results

MULTI-AGENT: **TRAINING STATISTICS - NOT EVALUATION**.

Aircraft records: 451; aircraft per scenario: 4; seed: 1.
Completed scenarios: 112; partially logged scenarios: 1 (their completed aircraft are included).
Model: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model_curriculum\01_4_agents\model`; MDP: `navigation-v3`.
Records: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model_curriculum\01_4_agents\training_metrics.csv`.

| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |
| --- | ---: | ---: | --- |
| scenarios | NA | 112 complete, 1 partial | episodes |
| waypoint_reached | NA | 273 | goals reached |
| intrusion_events | NA | 196 | events |
| intrusion_time | NA | 35,642 | seconds |
| restricted_area_events | NA | 81 | events |
| time_in_restricted_area | NA | 3,743 | seconds |
| sector_exit_events | NA | 47 | events |
| time_outside_sector | NA | 2,010 | seconds |
| flight_time | NA | 1,001,732 | seconds |
| violation-free | NA | 315 | clean flights |
| violation-free_success | NA | 208 | clean arrivals |
| goal completion | NA | 60.532 | % |
| violation-free rate | NA | 69.845 | % |
| violation-free arrival rate | NA | 46.12 | % |

All metric values above are totals, not means. MA intrusions remain per-aircraft counts; a conflict can count for both participants.

`violation-free` means zero safety events AND zero safety occupancy time; a clean timeout counts. `violation-free_success` additionally requires reaching the goal. These are derived statistics; the official scoring layer is unchanged.
