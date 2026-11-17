# Optional separation safety filter

This inference experiment uses the existing responsive PPO checkpoint and one-pass
RL-guided MCTS. It requires no retraining and is off by default. The saved model is
never modified. Old configurations without the new fields retain the old behavior.

## Selection rule

After the unchanged search finishes, forecast every root candidate over the same
60-second horizon, holding its heading/speed command and the neighbours' snapshot
actor commands. Use the existing independent turn/acceleration predictor, sampled
at intervals of at most one second. Stop at the episode time limit or own forecast
arrival; remove neighbours after their forecast arrival sample. The horizon is
independent of the unequal explored tree depths.

The target is the official 5 NM minimum plus a configurable buffer, default 3 NM
(8 NM total). If any candidates keep every forecast separation at or above the
target, use the existing search ranking within that safe set: visits, learned
return estimate, learned prior. If every candidate breaches the target, minimize
integrated separation deficit below 5 NM first, then deficit below 8 NM, then use
the search ranking to break ties. Deficits sum `(threshold - separation) * dt`,
clipped at zero, over all neighbours and samples; their unit is NM-seconds. This
prefers avoiding a predicted official intrusion over reducing a buffer-only breach.

All root candidates are screened, even if the search did not visit one. The screen
does not add MCTS visits, model evaluations or tree depth. It adds bounded motion
forecasts, sharing cached neighbour paths. The 24-simulation budget is unchanged.
This mode also works with two passes, but that combination has not been evaluated
and is not recommended by this experiment.

This is a separation-only filter. It does not reject commands for forecast obstacle
or sector entry; those continue to influence the original search returns. A command
that improves separation may worsen other metrics. The approximate neighbour
predictions and simultaneous replanning also mean passing the screen cannot
guarantee real separation. Safety eligibility describes this forecast only.

## Commands

Full scenario 9 development replay:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_navigation_v5_responsive/model.zip --mdp navigation-v5 --seed 42 --scenario-start 9 --episodes 1 --headless --lookahead-passes 1 --safety-filter --safety-buffer-nm 3 --safety-horizon-seconds 60 --trajectory-csv models/ma_navigation_v5_responsive/scenario9_safety_filter.csv
```

Separate 20-scenario evaluation, only after reviewing the diagnostic result:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5_responsive/model.zip --episodes 20 --lookahead-passes 1 --safety-filter --safety-buffer-nm 3 --safety-horizon-seconds 60 --out models/ma_navigation_v5_responsive/safety_filter.evaluation.csv
```

Use unique output paths to preserve previous runs. `--no-safety-filter` explicitly
disables a saved filter configuration. Omit these overrides to retain checkpoint
settings. `--no-lookahead` cannot be combined with safety filter options. Evaluation
writes resolved filter settings beside its CSV. Optional search candidate traces
record eligibility, minimum predicted separation, both deficit totals and forecast
duration, and mark the action actually selected by the filter. Those fields describe
a separate held-command forecast, not the tree's multi-action principal path.

## Rules scope

The [published rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
permit custom heading/speed mapping and RL-based algorithms. This optional screen
operates on the RL-guided search's candidates, keeping its learned ranking among
eligible actions and for fallback ties. It changes no live dynamics, scoring,
separation threshold, capture radius, termination, scenario distribution or seed
protocol, and uses no future simulator state. This appears consistent with the
published design surface; there is no explicit organizer approval of this hybrid.
The prescribed 1000 seed-42 scenarios and 10 aircraft remain necessary for reportable
multi-agent evaluation. Scenario 9 and 20-scenario checks are development results.

## Completed scenario 9 diagnostic

Same responsive checkpoint, 24 simulations, one pass, seed 42, 10 aircraft,
full 3000-second scenario; only the optional filter was enabled.

| Metric | Baseline | Separation filter |
| --- | ---: | ---: |
| Arrivals | 8 | 9 |
| Clean arrivals | 6 | 7 |
| Intrusion events (per aircraft) | 4 | 0 |
| Intrusion time (aircraft-seconds) | 580 | 0 |
| Restricted-area events | 0 | 3 |
| Restricted-area time (seconds) | 0 | 1,089 |
| Sector exits | 0 | 0 |
| Flight time (aircraft-seconds) | 17,703 | 17,591 |

The full filtered replay took 299.3 seconds here, including loading and trajectory
logging; 278.1 seconds were spent planning. The filter overrode 271 of 3,522 aircraft
decisions; 6 decisions had no buffer-safe candidate. There is no comparable measured
full-baseline wall time. On the same initial snapshot, two samples per mode averaged
1.019 seconds without filtering and 1.073 seconds with filtering (about 5% overhead).
The timing included a concurrent replay and is not a general performance guarantee.
Default unfiltered initial actions exactly matched the archived baseline for all
10 aircraft. All 157 tests passed, including unchanged live state and seed-prefix
selection for the parallel development check.

This mixed result warrants a 20-scenario check because clean arrivals and completion
improved, but it does not justify enabling the filter by default. In particular,
separation-only screening can select obstacle-crossing manoeuvres. The broader check
must assess all official safety metrics, rather than claiming success from zero
intrusions on this diagnostic.

Artifacts:

- [Full trajectory](../../models/ma_navigation_v5_responsive/scenario9_safety_filter.csv)
- [Final aircraft metrics and baseline comparison](../../models/ma_navigation_v5_responsive/scenario9_safety_filter.comparison.json)
- [Resolved settings, checkpoint checksum, runtime and intervention counts](../../models/ma_navigation_v5_responsive/scenario9_safety_filter.settings.json)
- [Initial-snapshot timings](../../models/ma_navigation_v5_responsive/safety_filter_root_timing.json)

## Parallel development check

The optional `scripts.evaluate_safety_filter` runner uses isolated subprocesses,
with each worker advancing the original seed-42 RNG by resets to its assigned
scenario. Scenario generation consumes that RNG only at reset. Each worker then
uses the existing `run_multi_agent` scoring loop. Results are collected in scenario
order, with 10 final aircraft records per scenario, and retained separately as JSON.
The runner refuses an existing output prefix. Worker logs and the settings manifest
remain available if a run fails. This runner is for development checks only.

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_safety_filter --model models/ma_navigation_v5_responsive/model.zip --out-prefix models/ma_navigation_v5_responsive/safety_filter.eval20_parallel --episodes 20 --workers 8
```

Use the canonical serial `scripts.evaluate_competition` command above for official
protocol runs, with 1000 episodes instead of 20. The parallel development run does
not alter that evaluator's seed or scoring behavior.

## Completed 20-scenario development check

Same checkpoint and first 20 seed-42 scenarios, 10 aircraft per scenario, all
200 aircraft records completed. The independent scenario 9 worker exactly matched
the earlier serial diagnostic's per-aircraft metrics, including reward.

| Metric | Baseline | Separation filter |
| --- | ---: | ---: |
| Arrivals | 195 / 200 (97.5%) | 196 / 200 (98%) |
| Clean arrivals | 169 / 200 (84.5%) | 176 / 200 (88%) |
| Clean flights (including timeouts) | 172 | 178 |
| Intrusion events (per aircraft) | 16 | 0 |
| Intrusion time (aircraft-seconds) | 1,548 | 0 |
| Restricted-area events | 13 | 29 |
| Restricted-area time (seconds) | 2,664 | 7,068 |
| Sector exits | 0 | 0 |
| Flight time (aircraft-seconds) | 298,952 | 311,045 |

Clean arrival rate rose 3.5 percentage points, but restricted-area occupancy grew
to about 2.65 times baseline and flight time rose about 4%. This is a trade-off,
not an unqualified safety improvement. Keep the filter optional and off by default.
A follow-up experiment should consider obstacle/sector hazards jointly with
separation, rather than promoting this traffic-only screen from a small check.
No retraining or official 1000-scenario evaluation was started.

- [Comparison table](../../models/ma_navigation_v5_responsive/safety_filter.eval20_parallel.comparison.md)
- [Comparison and per-scenario totals](../../models/ma_navigation_v5_responsive/safety_filter.eval20_parallel.comparison.json)
- [Filtered evaluation CSV](../../models/ma_navigation_v5_responsive/safety_filter.eval20_parallel.csv)
- [Evaluation report](../../models/ma_navigation_v5_responsive/safety_filter.eval20_parallel.md)
- [Checkpoint checksum, resolved settings and runtime](../../models/ma_navigation_v5_responsive/safety_filter.eval20_parallel.settings.json)

Validation: the standard serial evaluator completed the first two scenarios with
the same filter settings. All 20 final per-aircraft metric records, including reward,
exactly matched the corresponding parallel worker records. The full final test
suite passed 157 tests. [Serial validation CSV](../../models/ma_navigation_v5_responsive/safety_filter.serial_validation.csv).
