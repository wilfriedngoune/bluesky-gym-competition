# Competition results

MULTI-AGENT: **TRAINING STATISTICS - NOT EVALUATION**.

Aircraft records: 440; aircraft per scenario: 6; seed: 2.
Completed scenarios: 73; partially logged scenarios: 1 (their completed aircraft are included).
Model: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model_curriculum\02_6_agents\model`; MDP: `navigation-v3`.
Records: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model_curriculum\02_6_agents\training_metrics.csv`.

| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |
| --- | ---: | ---: | --- |
| scenarios | NA | 73 complete, 1 partial | episodes |
| waypoint_reached | NA | 313 | goals reached |
| intrusion_events | NA | 228 | events |
| intrusion_time | NA | 36,448 | seconds |
| restricted_area_events | NA | 53 | events |
| time_in_restricted_area | NA | 3,397 | seconds |
| sector_exit_events | NA | 107 | events |
| time_outside_sector | NA | 3,417 | seconds |
| flight_time | NA | 973,580 | seconds |
| violation-free | NA | 238 | clean flights |
| violation-free_success | NA | 172 | clean arrivals |
| goal completion | NA | 71.136 | % |
| violation-free rate | NA | 54.091 | % |
| violation-free arrival rate | NA | 39.091 | % |

All metric values above are totals, not means. MA intrusions remain per-aircraft counts; a conflict can count for both participants.

`violation-free` means zero safety events AND zero safety occupancy time; a clean timeout counts. `violation-free_success` additionally requires reaching the goal. These are derived statistics; the official scoring layer is unchanged.
