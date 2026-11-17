# Competition results

MULTI-AGENT: **TRAINING STATISTICS - NOT EVALUATION**.

Aircraft records: 430; aircraft per scenario: 10; seed: 3.
Completed scenarios: 43; partially logged scenarios: 0 (their completed aircraft are included).
Model: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model`; MDP: `navigation-v3`.
Records: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_training_metrics.csv`.

| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |
| --- | ---: | ---: | --- |
| scenarios | NA | 43 complete, 0 partial | episodes |
| waypoint_reached | NA | 276 | goals reached |
| intrusion_events | NA | 640 | events |
| intrusion_time | NA | 120,436 | seconds |
| restricted_area_events | NA | 80 | events |
| time_in_restricted_area | NA | 4,724 | seconds |
| sector_exit_events | NA | 102 | events |
| time_outside_sector | NA | 4,664 | seconds |
| flight_time | NA | 968,645 | seconds |
| violation-free | NA | 153 | clean flights |
| violation-free_success | NA | 117 | clean arrivals |
| goal completion | NA | 64.186 | % |
| violation-free rate | NA | 35.581 | % |
| violation-free arrival rate | NA | 27.209 | % |

All metric values above are totals, not means. MA intrusions remain per-aircraft counts; a conflict can count for both participants.

`violation-free` means zero safety events AND zero safety occupancy time; a clean timeout counts. `violation-free_success` additionally requires reaching the goal. These are derived statistics; the official scoring layer is unchanged.
