# Competition pipeline: changes and tradeoffs

Updated: 27 October 2026. This covers the changes made during our SA-to-MA work,
including the new reward experiment. Benefits describe design intent; improved
competition performance still needs to be established by training and evaluation.

| Implemented change | Benefit | Downside / limitation |
| --- | --- | --- |
| Extracted `NavigationMDP` from `SingleAgentEnv` | SA and MA reuse one observation/action implementation. | Shared navigation changes affect both; the original reward remains imperfect. |
| Added `MultiAgentEnv` with independent aircraft histories | Shared PPO can control all aircraft without mixing goals, actions or obstacle hysteresis. | Five visible traffic slots can omit four other aircraft. |
| Added `SharedPolicyVecEnv` for one BlueSky simulator | Preserves individual arrivals, terminal observations, timeouts and final metrics. | Inactive aircraft still occupy absorbing slots in memory and the timestep counter; PPO now excludes their padding by default. |
| Excluded inactive aircraft from PPO updates | Removes padding from advantage normalization, policy/value/entropy losses and explained variance, while retaining real final transitions and timeout bootstrapping. | Fewer optimizer samples per rollout; improved competition performance is unproven. Use `--include-inactive-samples` to compare with the previous behavior. |
| Added active-sample and padding diagnostics | Logs `rollout/active_samples`, `rollout/padding_fraction` and `time/active_timesteps`; run JSON separates active and absorbing timesteps. | Requested budgets still count all vector slots, so equal budgets can contain different numbers of useful samples. |
| Added optional smooth sector-buffer reward penalty | Supplies feedback before a sector exit; bounded, nonpositive cost stays at its maximum outside, so moving farther outside cannot remove it. Saved with the checkpoint and logged as `reward_sector_buffer`. | Disabled by default; may encourage excessive detours or hinder goals near the boundary. Endpoint-sampled shaping is not exact one-second occupancy. Performance needs a controlled comparison. |
| Added optional independent heading/speed command deadbands | Holds the last accepted target for negligible changes, including small navigation-reference shifts. Handles angular wraparound, aircraft-local history and resets. Settings are saved and restored in replay/evaluation/GIFs. | Defaults to zero; can suppress useful small corrections and does not prevent large reversals. Raw policy-action observations and the existing smoothness reward remain unchanged. Test thresholds with safety/arrival metrics. |
| Added opt-in `navigation-v4` passage experiment | Expands the heading offset to +/-180 degrees, allowing tangent directions even in avoidance mode; buffer defaults to 0.5 NM with 1 NM release, configurable. | New action semantics and observations require fresh v4 weights; smaller margins may increase violations. No guarantee of finding a flyable passage. |
| Added 24 body-relative polygon-clearance rays | Shows gaps and nearby corners across multiple obstacles using actual polygon edges, with a default 20 NM range and cached/vectorized geometry. | More computation and inputs; rays ignore turn radius and moving aircraft (existing traffic features remain). A geometrically open ray is not a safe trajectory. |
| Added optional mixed narrow-passage training scenarios | Reproducible rotated 3..8 NM gaps, 2..10 aircraft, with five obstacles. Explicit training-only probability can be combined with traffic curriculum. | Hand-selected examples may introduce distribution bias; final evaluation always uses original scenarios. No forced routing or planner replaces PPO. |
| Added replay trajectory CSVs | Records per-aircraft clearances, reference/target headings, active obstacle, actions and metrics, including early arrivals. | Extra I/O and diagnostic geometry work when enabled; requires a navigation MDP. Existing failed episodes must still be inspected to confirm their mechanism. |
| Added fresh training and optional checkpoint initialization | Train MA independently or reuse learned weights. | Transfer can inherit bad behavior; initialization uses a fresh optimizer. |
| Added CLI settings, backups and JSON run metadata | Experiments are configurable and easier to reproduce. | PPO finishes rollout buffers and can exceed the requested timestep count. |
| Added per-aircraft metrics CSV | Includes early arrivals and timeouts exactly once. | Training returns include exploration; use separate deterministic evaluation. |
| Connected PPO loading to evaluation, replay and GIF recording | Use the matching environment and trained model throughout the workflow. | MDP selection must match the checkpoint; small checks are not official results. |
| Batched live-agent inference with one CPU thread | Reduces overhead for the small policy network. | Best thread count can differ on other hardware. |
| Extended recordings to the full simulated-time budget | Navigation episodes can run all 600 decisions; final frames are retained. | Full reels require more time and memory than capped previews. |
| Added opt-in `navigation-v2` reward | Fixes asymmetric progress incentives without changing policy inputs/actions or the baseline. | New reward magnitudes invalidate comparisons of total reward with old runs. |
| Added potential-based distance shaping | Closed loops have no extra discounted shaping return compared with the same endpoint and horizon. | Shaping discount must equal PPO gamma; terminal/timeout handling matters. |
| Removed restricted-area escape/exit bonuses and outside-buffer costs | No separate payout for re-exiting an obstacle; avoids the old penalty discontinuity. | Less dense obstacle guidance; inherited navigation still provides an avoidance reference. |
| Added configurable metric-based safety costs and bounded conflict warnings | Costs reflect actual events and seconds, including violations between decisions. | Weights and predictive warning strength are initial hypotheses requiring validation. |
| Added per-component reward columns and saved reward config inside checkpoints | Shows why returns change and reproduces shaping settings during evaluation. | A transferred value network initially predicts the old reward scale. |
| Added regression tests and workspace-cache ignore | Checks lifecycle, transfer, scoring isolation and reward incentives; keeps smoke artifacts out of Git. | Tests establish correctness, not competitive policy quality. |
| Added automatic end-of-run totals reports (`.md` / `.json`) | Prints the sharing template, completion rates and clean-flight counts; can combine SA/MA CSVs. | Training totals describe changing, exploratory policies; they are labeled separately from evaluation. |
| Added separate post-training evaluation (1000 scenarios by default) | Reports the saved deterministic policy under the fixed seed-42 protocol, with 10000 MA aircraft records. | Adds evaluation runtime after training; `--eval-episodes 20` is a quick check and `--eval-episodes 0` skips it. |
| Added opt-in `navigation-v3` with nine traffic slots, presence masks and remaining time | Every other controlled aircraft is visible; absent slots and the deadline are explicit. Current intrusions sort first. | Changed inputs require fresh weights or a matching v3 checkpoint; SA/v2 checkpoints cannot load directly. |
| Added v3 discount 0.998 and configurable earlier warnings | Longer planning horizon; predictive warnings start within 180 s and cost up to 0.05/s by default. | Initial hypotheses; broader warnings can cause unnecessary detours. Actual safety costs and goal bonus are unchanged. |
| Added optional fixed-budget 2-to-4-to-6-to-10 curriculum | Learns on lighter traffic before full density; fixed observation shapes allow weight transfer. Each stage uses its own process. | Stages advance by timestep budget, not demonstrated competence; each starts a fresh optimizer. |
| Saved curriculum manifest and observation shapes | Records stage seeds, inputs/outputs, budgets and actual counts; evaluation/replay reconstruct the saved reward profile. | Final PPO timestep counter covers the final stage; use the manifest for the complete training budget. |
| Fixed near-zero heading command formatting and rejected nonfinite navigation values | BlueSky accepts headings that Python would otherwise write in scientific notation; invalid values fail with the aircraft ID. Exact finite target values are preserved. | An already running Python process retains the old implementation; rerun evaluation from the saved checkpoint to use the fix. |

## Narrow-passage diagnosis and v4 verification

Replayed the original v3 model on the first four seed-42 scenarios with the
new trace logger; final goals, intrusion events, restricted entries, sector
exits and flight time match the previously saved baseline for those scenarios.
In scenario 3, KL005 loops near polygons 1 and 2, whose minimum gap is 3.893 NM.
Their 3 NM avoidance buffers overlap. During the final 600 seconds the aircraft
is in avoidance mode for 72.5% of decisions and ends 2.55 NM farther from its
goal. This supports investigating a buffered bottleneck but does not prove
turn feasibility or a unique cause.

See the [trajectory plot](../../.cache/passage_audit/stalled_v3.png) and
[audit notes](../../.cache/passage_audit/README.md). V4's controlled 4 NM corridor
test completes a straight-ahead flight without restricted-area entries.
The full suite passes 76 tests; a four-stage smoke curriculum, saved-profile
evaluation and GIF recording also completed. No trained-v4 performance gain
is claimed, and old v3 checkpoints are not directly compatible with v4 inputs.

## Short boundary experiments

All four policies were checked on the same 20 seed-42 scenarios (200 aircraft).
The three continuations start from the original v3 checkpoint with training
seed 4, active-only PPO samples, and 100,000 requested / 102,400 actual steps.
Their declared settings match apart from the optional boundary cost.

| Policy | Goals | Clean arrivals | Sector exits | Intrusion time [s] |
| --- | ---: | ---: | ---: | ---: |
| Original v3 | 172 | 55 | 17 | 43,940 |
| Continuation, cost 0 | 170 | 56 | 54 | 38,910 |
| Continuation, cost 0.01 | 169 | 50 | 7 | 36,394 |
| Continuation, cost 0.005 | 170 | 53 | 23 | 41,294 |

The boundary cost reduced exits in these checks, with mixed consequences for
clean arrivals and other safety metrics. The weaker cost recovered three clean
arrivals relative to 0.01 but remained below the zero-cost continuation and
original checkpoint. These exploratory results do not establish an overall
winner or justify replacing the original baseline.

For cost 0.01 versus zero, scenarios 2 and 10 account for seven fewer clean
arrivals. More aircraft experienced intrusions (134 versus 129), despite lower
total intrusion time; the weaker cost exposed 131 aircraft. Thus both conflict
duration and the number of affected aircraft matter. Aggregate records do not
identify the geometric cause. See the
[full comparison](../../models/ma_navigation_v3_boundary_005_100k/comparison.md)
for all metrics, per-scenario clean arrivals, provenance and reproduction.

## Reward v2: initial weights

Select explicitly with `--mdp navigation-v2`; existing defaults still select
`navigation`. The original model and training CSV are not rewritten by the
command below. The v2 reward is MA-only; single-agent rewards remain unchanged.

| Term | Value |
| --- | --- |
| Goal reached | +20 once |
| Progress | `10 * (gamma * Phi(next) - Phi(current))` |
| Intrusion | -5 per new event, -0.1 per aircraft-pair second |
| Restricted area | -5 per entry, -0.05 per occupied second |
| Sector exit | -5 per exit, -0.05 per outside second |
| Predicted conflict | Up to -0.02 per second; approaching within 120 s / 8 NM, outside actual intrusion |
| Flight time | -0.002 per second |
| Action change | -0.005 times squared change in the normalized policy action |

`Phi` is negative remaining distance outside the 5 km capture radius, normalized
by 100 km. Goal termination has zero potential. Time-limit truncation retains
its potential and terminal observation for value bootstrapping; no timeout
bonus or penalty is added. One-step shaping can be positive while stationary:
its protection against extra loop reward concerns discounted trajectories.
See [potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

Safety and time rates use actual simulated seconds. Action smoothness is per
decision. The warning cost uses the most threatening live aircraft and does not
double-count an existing intrusion. Diagnostics add `reward_*` columns to the
training CSV; their per-flight sum equals the official `total_reward` field.

## Latest experiment: navigation v3

V3 includes the proposed observation, horizon and warning changes. It uses nine
traffic slots at every training density, explicit masks, remaining episode time,
and prioritizes current intrusions before approaching traffic. The competition
scenario generator, metrics, 3000-second budget and heading/speed actions are
inherited unchanged. A larger input requires a new model; omit `--init-model`.

```powershell
.\.venv312\Scripts\python.exe train_zoo.py train 2000000 --mdp navigation-v3 --curriculum 2 4 6 10 --model models/ma_navigation_v3/model --metrics models/ma_navigation_v3/training_metrics.csv --checkpoint-dir models/ma_navigation_v3/checkpoints
```

This splits the requested budget equally across four stages and runs the normal
1000-scenario evaluation only after the final stage. Add `--eval-episodes 20` for
a shorter nonofficial check. For a direct ten-aircraft experiment, omit
`--curriculum`. V3 defaults to gamma 0.998, a 180 s warning horizon, 8 NM predicted
closest-approach buffer, and warning cost 0.05/s. Override via `--gamma`,
`--conflict-horizon`, `--conflict-buffer-nm`, and `--conflict-risk-cost`.
The warning horizon is limited to 600 s because of the shared CPA encoding.

Earlier stage models, metrics, reports and backups are stored under
`models/ma_navigation_v3/model_curriculum/`; the final stage uses the requested
output paths. `curriculum.json` records progress, seeds and actual stage budgets.
The final model JSON also references the manifest and total stage timesteps.

These changes combine several hypotheses; performance must be measured against
the saved baseline on identical validation scenarios. For ablations, use v3
with `--gamma 0.99 --conflict-horizon 120 --conflict-risk-cost 0.02`, or disable
the curriculum, to isolate subsets of the changes. No full v3 training has been
performed as part of implementation checks.

## Reproduce the reward-only v2 experiment

Keep gamma at 0.99 to test the reward change first. This command initializes
from the saved MA policy, starts a fresh optimizer, and writes separate outputs:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py train 1000000 --mdp navigation-v2 --init-model multi_agent_model.zip --model models/ma_reward_v2/model --metrics models/ma_reward_v2/training_metrics.csv --checkpoint-dir models/ma_reward_v2/checkpoints
```

Omit `--init-model` to train from random weights. Test `--gamma 0.998` as a
separate follow-up experiment, with a separate output directory. Reward shaping
automatically uses the same gamma and saves its full configuration in the model.

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_reward_v2/model --mdp navigation-v2
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-v2 --model models/ma_reward_v2/model --episodes 20 --out models/ma_reward_v2/quick_eval.csv
```

The 20-scenario check is nonofficial. Compare completion, safety and flight time
against the original model on identical scenarios. Remove `--episodes 20` for
the official 1000-scenario seed-42 protocol; avoid selecting configurations on
the official scenarios repeatedly. Use separate validation scenarios for tuning.

## Preserved and deferred

The official scenario generator, scoring, termination, A320 dynamics, 1-second
sampling, separation/capture thresholds and simulated-time budget are unchanged.
V2 keeps the original five traffic slots, speed range, navigation reference,
architecture and PPO defaults to isolate the reward experiment.

V3 implements the nine-aircraft observations/masks, remaining-time input and
fixed-budget traffic curriculum. Exclusion of padding from PPO losses is now
implemented as the trainer default. For the previous
training behavior, add `--include-inactive-samples`. Forced encounter curricula
and centralized critics remain deferred.

Checks: `python -m unittest discover -s tests -v`, plus short real PPO training,
checkpoint loading and recording checks. No full v2 training or claim of improved
competition performance is included in these implementation checks.
