# Cheaper optional lookahead

The new `--lookahead-preset fast` option reduces the MCTS budget without changing
the saved model. It works in both evaluation and replay. Existing commands retain
the checkpoint's search configuration; no retraining is required.

| Setting | Current saved model | Fast preset |
| --- | ---: | ---: |
| Simulations per aircraft decision | 24 | 12 |
| Maximum tree depth | 6 | 3 |
| Coordination passes | 1 | 1 |
| Candidate manoeuvres | 7 | 7 |
| Forecast segment | 15 s | 15 s |
| Command interval | 5 s | 5 s |
| Forecast hazard sampling | 1 s | 1 s |

The fast preset preserves the learned priors/critic, reward coefficients, candidate
count, forecast dynamics and optional safety filter settings. Explicit simulation,
depth or pass overrides take precedence. With the default 15-second segments, the
maximum tree horizon decreases from 90 to 45 seconds; actual explored paths vary.
Fewer visits can change actions and worsen arrivals or safety. This is an optional
inference experiment, not a claim of equal performance.

## Commands

Replay with the latest fresh checkpoint and cheaper planning:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_navigation_v5_fresh_v6/model.zip --mdp navigation-v5 --seed 42 --episodes 1 --lookahead-preset fast
```

Compare the same first 20 seed-42 scenarios, using a separate output file:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5_fresh_v6/model.zip --episodes 20 --lookahead-preset fast --out models/ma_navigation_v5_fresh_v6/fast.evaluation.csv
```

For an intermediate budget, try `--lookahead-simulations 18 --lookahead-depth 4`
in either command instead of the preset. These are configurable experiments,
not independently validated recommendations. Omit the overrides or use
`--lookahead-preset saved` to retain saved settings. Evaluation records the requested
preset and full resolved configuration in its settings JSON. Replay prints the
resolved override; optional search traces also retain the resolved configuration.

These settings speed up evaluation/replay. PPO training already collects actor-only
actions, so they do not accelerate its policy updates. To skip the expensive automatic
post-training evaluation, use the training launcher's `--eval-episodes 0`, then run
the separate evaluation command above. Training weights and rewards are unchanged.

## Measured cost and correctness checks

Using the fresh-v6 checkpoint, compare copied snapshots from seed-42 scenarios 1
and 9 at 0 and 5 simulated seconds, with ten live aircraft. All modes see the same
state; only the saved planner's actions advance the environment between snapshots.
Order alternates, and timings exclude loading and profiler overhead.

| Mode | Mean planning wall time per joint decision |
| --- | ---: |
| Saved budget | 0.845 s |
| Fast budget | 0.449 s |

The fast mode was about 1.88 times faster (47% less planning time) on these four
snapshots. This does not establish a full-episode speedup or unchanged safety.
A separate full scenario 9 replay checks behavior at the cheaper budget.

A profile at the saved budget recorded about 3.3 million calls for a ten-aircraft
decision. Forecast transitions, CPA calculations and copied-state observation
construction account for much of the work. Halving searches avoids some of this
work; changing GPU settings would not eliminate these Python forecast loops.

There is also a default-path optimization: forecasts no longer calculate paths or
repeat arrival tests for traffic already marked arrived in a copied forecast scene.
This keeps arrival-sample ordering and frozen positions unchanged. Its benefit
mainly applies to deeper nodes and scenes near arrival. No general speedup is claimed
for that small shortcut alone. Default-budget actions and visit diagnostics exactly
matched the pre-change implementation on all four snapshots; bounded arrival cases
also matched rewards, copied states and observer samples. All 167 tests passed.

Evidence:

- [Timings and resolved configurations](../../.cache/fast_lookahead/benchmark.json)
- [Profile](../../.cache/fast_lookahead/profile.txt)
- [Benchmark script](../../.cache/fast_lookahead/benchmark.py)

## Competition scope

This remains learned-policy/value-guided search. Changing its budget does not
change live dynamics, separation/capture thresholds, metric sampling, termination,
scenario generation or scoring. The original 1000-scenario seed-42 protocol with
10 aircraft is still required for reportable multi-agent results. The
[published rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
allow RL algorithm design and heading/speed mapping; this retains the existing
hybrid interpretation rather than claiming explicit organizer approval. Do not
compare results across budgets without recording which inference configuration ran.

## Full scenario 9 check and intermediate budget

The fast preset completed the full seed-42 scenario 9 with all ten aircraft,
using the same fresh-v6 checkpoint. It reached all 10 goals, but clean arrivals
fell from 10 to 8. One encounter generated two per-aircraft intrusion events
and 152 aircraft-seconds of intrusion occupancy. Restricted-area and sector-exit
metrics remained zero. Total flight time increased from 15,649 to 16,113
aircraft-seconds. The fast replay took 145.4 wall seconds including loading and
trajectory logging; no comparable isolated full-scenario saved-budget timing
was collected, so do not interpret that as a measured episode speedup.

The cheaper configuration should remain optional. It is not established as a
replacement for the saved model's 90.5% clean-arrival result across 20 scenarios.
No 20-scenario or official-protocol fast-budget evaluation was run here.

An intermediate 18-simulation/depth-4 configuration was also timed on the same
four copied snapshots in a separate paired run: 0.927 seconds at the saved budget
versus 0.723 seconds at 18/4, about 1.28 times faster (22% less planning time).
This preserves more search work than the fast preset. A separate scenario 9
replay checks that intermediate configuration's behavior.

- [Fast-budget scenario comparison and resolved settings](../../models/ma_navigation_v5_fresh_v6/scenario9_fast.comparison.json)
- [Fast-budget trajectory](../../models/ma_navigation_v5_fresh_v6/scenario9_fast.csv)
- [Intermediate-budget paired timings](../../.cache/fast_lookahead/budget18_benchmark.json)

The completed 18/4 scenario 9 replay reached all 10 goals with all 10 flights clean,
matching the saved budget's clean-arrival count on this one case. All intrusion,
restricted-area and sector-exit event/time metrics were zero. This is a bounded
check, not evidence that the 20-scenario clean-arrival rate will be preserved.
See [18/4 full-scenario results and timing](../../models/ma_navigation_v5_fresh_v6/scenario9_budget18.comparison.json)
and [trajectory](../../models/ma_navigation_v5_fresh_v6/scenario9_budget18.csv).
