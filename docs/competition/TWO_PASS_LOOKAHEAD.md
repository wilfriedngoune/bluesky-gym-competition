# Optional two-pass coordination experiment

The default remains one pass. Existing checkpoints without `coordination_passes`
load as one pass; no retraining or checkpoint modification is required.

With two passes, each aircraft first searches against the simultaneous PPO actor
commands. The complete first-pass proposal is frozen, then every aircraft searches
again against those proposals. Only the second-pass actions are executed together.
Both passes use the same copied current state. Neither pass sees another aircraft's
partially updated choices from its own pass. This reduces one source of prediction
mismatch; it does not guarantee mutually compatible final choices, collision freedom,
or accurate long-term predictions.

The simulation budget is total across passes: the saved 24 simulations become
12 + 12. Odd budgets put the extra simulation in the first pass. Splitting a budget
reduces the search depth/coverage available to each pass, and can worsen outcomes.
A two-pass decision can also cost more wall time despite equal simulation counts,
because each tree has root expansion overhead. Do not equate equal visit budgets
with equal runtime.

## Run with the existing model

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_navigation_v5_responsive/model.zip --mdp navigation-v5 --seed 42 --scenario-start 9 --episodes 1 --headless --lookahead-passes 2 --trajectory-csv models/ma_navigation_v5_responsive/scenario9_two_pass.csv
```

For a separate 20-scenario check (not official evaluation):

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5_responsive/model.zip --episodes 20 --lookahead-passes 2 --out models/ma_navigation_v5_responsive/two_pass.evaluation.csv
```

Evaluation settings record the resolved pass count and total budget. Search traces
record only the final pass, with pass index, per-pass and total simulation counts.
Use unique output names to preserve prior experiments.

## Competition scope

[Published competition rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
allow custom observations, rewards and heading/speed action mapping, while requiring
an RL-based solution and preserving the world, metrics and evaluation protocol.
This hybrid retains learned actor priors and learned critic values in both passes.
Forecasts are independent approximations operating on copied current observations;
they do not advance or reset the live simulator or use future scenario information.

This is an interpretation consistent with those published allowances, not explicit
organizer approval of this particular hybrid. The implementation changes no A320
dynamics, separation threshold, goal capture radius, termination, scoring, scenario
generation, or evaluation seed protocol. Scenario 9 is a development diagnostic;
results on it are not evidence of official performance. Keep the prescribed 1000
seed-42 scenarios and 10 aircraft for reportable multi-agent evaluation.

## Completed scenario 9 result

Same responsive checkpoint, seed 42, scenario 9, all 10 aircraft, full 3000-second
budget, 24 total simulations per aircraft decision. This is a development check.

| Metric | One pass | Two passes (12 + 12) |
| --- | ---: | ---: |
| Arrivals | 8 | 8 |
| Clean arrivals | 6 | 4 |
| Clean flights (including timeouts) | 7 | 4 |
| Intrusion events (per aircraft) | 4 | 10 |
| Intrusion time (aircraft-seconds) | 580 | 1,668 |
| Restricted-area events | 0 | 1 |
| Restricted-area time (seconds) | 0 | 342 |
| Sector exits | 0 | 0 |
| Flight time (aircraft-seconds) | 17,703 | 17,615 |

Do not promote this configuration. The experiment does not establish why it became
worse: reduced per-tree coverage and residual disagreement between final actions
are possible contributors, not proven explanations. No broader evaluation or
retraining was started, since the proposed first diagnostic failed.

At the initial 10-aircraft snapshot, two repeated timings averaged about 0.95 seconds
for one pass and 1.08 seconds for two passes (about 13% overhead). These include
cold per-decision forecast caches and exclude model loading; a concurrent replay
was running. This is a small snapshot timing, not a full-episode runtime comparison.
The default one-pass actions exactly matched the archived baseline for all 10
aircraft at this initial snapshot. The full test suite passed 145 tests.

Artifacts:

- [Completed trajectory](../../models/ma_navigation_v5_responsive/scenario9_two_pass.csv)
- [Comparison, final aircraft records and checkpoint checksum](../../models/ma_navigation_v5_responsive/scenario9_two_pass.comparison.json)
- [Timing measurements and resolved settings](../../models/ma_navigation_v5_responsive/two_pass_root_timing.json)
