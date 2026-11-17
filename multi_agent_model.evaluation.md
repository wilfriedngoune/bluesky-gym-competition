# Competition results

MULTI-AGENT: **OFFICIAL PROTOCOL (1000 scenarios, seed 42)**.

Aircraft records: 10,000; aircraft per scenario: 10; seed: 42.
Model: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model.zip`; MDP: `navigation-v3`.
Records: `D:\Software-Entwicklung\bluesky-gym-competition\multi_agent_model.evaluation.csv`.

| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |
| --- | ---: | ---: | --- |
| scenarios | NA | 1,000 episodes | episodes |
| waypoint_reached | NA | 7,007 | goals reached |
| intrusion_events | NA | 12,766 | events |
| intrusion_time | NA | 2,341,680 | seconds |
| restricted_area_events | NA | 2,242 | events |
| time_in_restricted_area | NA | 149,674 | seconds |
| sector_exit_events | NA | 1,269 | events |
| time_outside_sector | NA | 63,161 | seconds |
| flight_time | NA | 20,677,320 | seconds |
| violation-free | NA | 3,623 | clean flights |
| violation-free_success | NA | 2,814 | clean arrivals |
| goal completion | NA | 70.07 | % |
| violation-free rate | NA | 36.23 | % |
| violation-free arrival rate | NA | 28.14 | % |

All metric values above are totals, not means. MA intrusions remain per-aircraft counts; a conflict can count for both participants.

`violation-free` means zero safety events AND zero safety occupancy time; a clean timeout counts. `violation-free_success` additionally requires reaching the goal. These are derived statistics; the official scoring layer is unchanged.
