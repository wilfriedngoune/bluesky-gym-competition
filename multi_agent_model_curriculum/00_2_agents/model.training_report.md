# Competition results

MULTI-AGENT: **TRAINING STATISTICS - NOT EVALUATION**.

Aircraft records: 496; aircraft per scenario: 2; seed: 0.
Completed scenarios: 248; partially logged scenarios: 0 (their completed aircraft are included).
Model: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model_curriculum\00_2_agents\model`; MDP: `navigation-v3`.
Records: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model_curriculum\00_2_agents\training_metrics.csv`.

| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |
| --- | ---: | ---: | --- |
| scenarios | NA | 248 complete, 0 partial | episodes |
| waypoint_reached | NA | 332 | goals reached |
| intrusion_events | NA | 74 | events |
| intrusion_time | NA | 10,522 | seconds |
| restricted_area_events | NA | 80 | events |
| time_in_restricted_area | NA | 4,582 | seconds |
| sector_exit_events | NA | 211 | events |
| time_outside_sector | NA | 5,458 | seconds |
| flight_time | NA | 1,073,190 | seconds |
| violation-free | NA | 385 | clean flights |
| violation-free_success | NA | 279 | clean arrivals |
| goal completion | NA | 66.935 | % |
| violation-free rate | NA | 77.621 | % |
| violation-free arrival rate | NA | 56.25 | % |

All metric values above are totals, not means. MA intrusions remain per-aircraft counts; a conflict can count for both participants.

`violation-free` means zero safety events AND zero safety occupancy time; a clean timeout counts. `violation-free_success` additionally requires reaching the goal. These are derived statistics; the official scoring layer is unchanged.
