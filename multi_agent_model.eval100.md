# Competition results

MULTI-AGENT: **NONOFFICIAL / VERIFY EVALUATION PROVENANCE**.

Aircraft records: 1,000; aircraft per scenario: 10; seed: 42.
Model: `multi_agent_model.zip`; MDP: `navigation-v3`.
Records: `multi_agent_model.eval100.csv`.

| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |
| --- | ---: | ---: | --- |
| scenarios | NA | 100 episodes | episodes |
| waypoint_reached | NA | 722 | goals reached |
| intrusion_events | NA | 1,132 | events |
| intrusion_time | NA | 224,106 | seconds |
| restricted_area_events | NA | 207 | events |
| time_in_restricted_area | NA | 14,375 | seconds |
| sector_exit_events | NA | 107 | events |
| time_outside_sector | NA | 6,766 | seconds |
| flight_time | NA | 2,065,213 | seconds |
| violation-free | NA | 385 | clean flights |
| violation-free_success | NA | 296 | clean arrivals |
| goal completion | NA | 72.2 | % |
| violation-free rate | NA | 38.5 | % |
| violation-free arrival rate | NA | 29.6 | % |

All metric values above are totals, not means. MA intrusions remain per-aircraft counts; a conflict can count for both participants.

`violation-free` means zero safety events AND zero safety occupancy time; a clean timeout counts. `violation-free_success` additionally requires reaching the goal. These are derived statistics; the official scoring layer is unchanged.
