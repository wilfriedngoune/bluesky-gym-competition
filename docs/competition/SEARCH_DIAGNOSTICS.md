# Search diagnostics without changing the policy

Diagnostics are opt-in replay observers. They read the completed search tree
and immutable forecast samples from rollouts the search already performed.
They do not add simulations, visits, candidate actions or policy/value calls.
Live command records and actual encounters are logged after decisions; future
actual data is never supplied to the policy or search. Training, authoritative
BlueSky dynamics, reward, scoring, and evaluation protocol stay unchanged.

## Replay the encounter

The responsive model's seed-42 scenario 9 has the KL001/KL006 and KL001/KL009
encounters. To reproduce the diagnostic window with a new output prefix:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_navigation_v5_responsive/model.zip `
  --mdp navigation-v5 --seed 42 --scenario-start 9 --episodes 1 --max-steps 100 --headless `
  --trajectory-csv models/ma_navigation_v5_responsive/scenario9_debug_trajectory.csv `
  --search-trace-prefix models/ma_navigation_v5_responsive/scenario9_debug `
  --search-trace-start 40 --search-trace-end 400 `
  --search-trace-agents KL001 KL006 KL009 `
  --search-trace-neighbours KL001 KL006 KL009
```

Root decisions are recorded from seconds 40 through 400. The 100-decision cap
replays to second 500, allowing observation of the last decision's maximum
90-second forecast. Existing diagnostic files are protected from overwrite;
choose a different prefix to rerun. This capped replay is not a full evaluation.
Agent filters affect only logging, not the aircraft included in control,
predictions, observations or scored safety checks.

## Files

| Suffix | Contents |
| --- | --- |
| `.candidates.csv` | Candidate heading/speed, learned prior, visits, average backed-up return, immediate transition reward, leaf value, explored depth and selected index |
| `.forecasts.csv` | Retained one-second samples along each candidate's most-visited continuation, predicted separation/CPA, own target, and assumed neighbour command |
| `.actual.csv` | Live horizontal positions, true airspeeds, separation, TCPA and DCPA at real decision boundaries |
| `.commands.csv` | Heading/speed commands actually accepted for the simultaneous live decisions |
| `.settings.json` | Model checksum, settings, aircraft/time filters, completion and coverage flags |

The most-visited continuation is a diagnostic summary of the tree; it is not
the entire distribution of rollouts used in a candidate's mean return. An
unvisited candidate has blank reward/value fields and no forecast, rather than
a fabricated zero value. Forecasts assume other aircraft hold their root
actor commands, exactly as in the current planner.

`explored_depth` is the deepest visited branch below a root candidate;
`principal_path_depth` describes the retained most-visited continuation. A
maximum depth of six permits 90 seconds but does not ensure that much coverage.
The actual flight can choose new commands every five seconds, while searched
macro steps currently hold targets for 15 seconds. Longer-horizon prediction
errors therefore also include subsequent replanning by the own aircraft.

## Compare predictions and actual outcomes

```powershell
.\.venv312\Scripts\python.exe -m scripts.analyze_search_trace models/ma_navigation_v5_responsive/scenario9_debug
```

This writes `.analysis.csv`, `.analysis.json` and `.analysis.md`. It matches the
same episode, directed aircraft pair and timestamp. Comparisons include:

- Predicted minimum separation versus the sampled actual minimum.
- The next-real-decision prediction error, separated from longer-horizon error.
- Own and neighbour position errors.
- The neighbour command assumed at the root versus the one actually selected.
- Forecast depth, visits and average return of the selected candidate.

The comparisons are overlapping windows, not counts of independent conflicts.
Actual diagnostics sample decision boundaries; the official intrusion metrics
continue sampling every second. Geometry predictions use local planar NM and
actual CPA uses the existing geodesic calculation. Inspect coverage flags when
a replay ends before all forecasts can be observed.

The completed [scenario-9 findings](SCENARIO9_SEARCH_FINDINGS.md) include the
recorded comparison, plot and interpretation. Diagnostics can add logging
overhead; leave the trace flags off for ordinary performance measurements.
