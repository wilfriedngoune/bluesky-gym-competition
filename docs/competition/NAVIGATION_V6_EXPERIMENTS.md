# Navigation v6 controlled experiments

These tools implement additional ten-aircraft training, timeout diagnosis,
training-only obstacle practice, and search-to-actor imitation with subsequent PPO
refinement. Existing models and default reward/world settings are preserved.
No full new training run or improvement claim is implied by the implementation.

## 1. More practice at competition traffic density

The fresh-v6 curriculum used about two million agent-slot timesteps across four
stages; only 501,760 were at ten aircraft, including 336,224 active aircraft-steps.
`train_navigation_v6` continues its weights at ten aircraft for the entire new
budget, retaining saved observations, rewards, timing and command mapping.
Like the existing trainer, this transfers weights into a fresh optimizer; it does
not resume optimizer or rollout-buffer state. Absorbing slots remain masked.

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_navigation_v6
```

Defaults: current fresh-v6 checkpoint, two million requested additional agent-slot
steps, training seed 20, checkpoints every 100,000 slot steps, new output
`models/ma_navigation_v6_continued/model`. Rollout budgets can overshoot; metadata
records active and absorbing steps separately. Automatic seed-42 evaluation is
skipped by default so training does not unexpectedly launch an expensive benchmark.
Use `--eval-episodes 20` for a separate final development check. Add `--dry-run` to
inspect the exact command without creating outputs or training. Existing experiment
artifacts are rejected; choose a new output directory for every run.

## 2. Diagnose clean timeouts

The current 20-scenario CSV contains 12 missed arrivals, including 10 clean misses,
in scenarios 1, 5, 10, 12, 13, 14, 15 and 19. Its rows do not contain aircraft IDs.
The inventory groups the known ordered ten-record scenario blocks; it never guesses
which aircraft failed from row order. Replays identify actual aircraft IDs.

Inventory only (fast):

```powershell
.\.venv312\Scripts\python.exe -m scripts.diagnose_navigation_timeouts --list-only --out-dir models/ma_navigation_v5_fresh_v6/timeouts_inventory
```

Replay and analyze one failed scenario with the saved inference settings:

```powershell
.\.venv312\Scripts\python.exe -m scripts.diagnose_navigation_timeouts --scenarios 1 --out-dir models/ma_navigation_v5_fresh_v6/timeouts_scenario1
```

Omit `--scenarios` to replay all affected scenarios. This may take substantial time.
Use distinct output directories; an existing directory is never overwritten.
`--max-steps 4` is a cheap pipeline smoke check, and explicitly marks unfinished
flights capped/incomplete rather than timing them out. The tool checks available
CSV settings against the requested seed/scenario count and records model hashes,
source files, search settings and overrides. Reduced lookahead changes behavior,
so use the saved configuration when diagnosing the original failures.

Reports contain net goal-distance gain, observed path length, speed and turning,
safety events/occupancy, and low-progress/possible-circling windows. These flags are
hypotheses: avoidance can legitimately look like a detour or a loop. Trace positions
and goal distances are observations before each step; the last recorded distance
is explicitly not an exact terminal position. Actual timeout/arrival classification
uses the final metric time and goal flag.

## 3. Train an obstacle-practice comparison arm

Keep the initialization, seed, hyperparameters and requested budget the same as
the ordinary continuation; change only the training scenario mixture:

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_navigation_v6 --targeted-probability 0.25 --model models/ma_navigation_v6_obstacles/model
```

This uses 75% ordinary generator calls and 25% custom practice, split between
randomized narrow passages and blocked-direct-route detours. The custom examples
vary geometry, rotation, traffic placement and speeds; tests cover safe starts,
goals, initial separation and feasible detour geometry. They are a training
curriculum, not a replacement competition distribution. The original generator
and RNG stream are preserved when custom probability is zero. Evaluation and
held-out validation always construct the original scenario generator.

The low-level equivalent is `train_zoo.py train ... --passage-training-probability
0.25 --passage-training-profile mixed`. The old passage-only profile remains the
default. Reward coefficients, A320 physics, 5 NM separation and goal capture
criteria are unchanged. A custom mixture is not guaranteed to improve outcomes;
compare checkpoints on held-out seeds before selecting one.

## 4. Learn useful search choices, then refine with PPO

The teacher is a fixed PPO actor/critic-guided search. The student is a separate
copy of its PPO weights. On training scenarios, the tool records only live-aircraft
observations and teacher target actions, rolls out a student/teacher mixture, then
fits actor weights in a separate supervised phase. Labels accumulate across
iterations in a bounded reservoir. This is an imitation-learning experiment
inspired by [DAgger](https://proceedings.mlr.press/v15/ross11a.html), not a claim to
reproduce that paper's guarantees with an imperfect search teacher.

```powershell
.\.venv312\Scripts\python.exe -m scripts.distill_navigation_policy --model models/ma_navigation_v5_fresh_v6/model.zip --out models/ma_navigation_v6_distilled/model --seed 20
```

Defaults are a small initial experiment: two iterations, 64 joint decisions per
iteration, five fitting epochs, batch size 64 and teacher-mixture probability 0.5.
That is roughly 640 simulated seconds at five-second decisions, often less than a
whole scenario. Increase `--decisions-per-iteration` and `--iterations` to cover
full flights and multiple scenarios; runtime depends heavily on teacher search.
Use `--dry-run` to inspect outputs. Teacher search presets/budgets can be overridden
and are recorded. An inferior teacher can supply inferior labels; there is no
arrival/safety guarantee from reducing imitation loss.

Only the actor MLP/head are fitted. The fixed teacher, student critic, Gaussian
variance and PPO optimizer state are not updated by imitation. Trainable shared
feature extractors are rejected to prevent silently changing the critic. Heading
loss accounts for the periodic heading action, and the retained model uses the
best aggregate training-label loss including the initial fit. Outputs include a
new checkpoint, bounded non-pickle NPZ dataset and provenance JSON.

Refine the student using ordinary on-policy PPO:

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_navigation_v6 --init-model models/ma_navigation_v6_distilled/model.zip --model models/ma_navigation_v6_distilled_refined/model --seed 30
```

Search-selected actions are never recorded as if PPO sampled them. The supervised
phase and PPO phase remain separate. The source model stays untouched. Distilled
checkpoints default to actor-only inference, and that default survives PPO refinement,
evaluation, replay and GIF recording. `--with-lookahead` explicitly enables search;
`--no-lookahead` explicitly disables it. An explicit search-budget override also
opts into search in evaluation/replay. Existing checkpoints retain their former
search-enabled default. The imitation phase alone is not a replacement for
validation and PPO refinement.

## Held-out checkpoint comparison

Run the same validation seed streams for each candidate, without learning:

```powershell
.\.venv312\Scripts\python.exe -m scripts.validate_navigation_checkpoints --models models/ma_navigation_v5_fresh_v6/model.zip models/ma_navigation_v6_continued/model.zip models/ma_navigation_v6_obstacles/model.zip --seeds 101 102 --episodes 3 --out-dir models/ma_navigation_v6_validation
```

Alternatively use `--checkpoint-dir <directory>` to compare all matching ZIPs.
Choose seeds not used for training or repeated development tuning; the tool cannot
infer every previous training run. Defaults use 101 and 102. Each seed is supplied
on the first reset only, followed by unseeded resets, for every model. Reports save
raw final metrics, model hashes, resolved search mode and settings, per-seed results
and totals. Clean arrival requires both goal completion and zero safety event/time
metrics. No scalar winner or automatic promotion is selected.

By default, each model uses its saved deployment mode. Actor-only distilled models
and search-enabled original models are therefore end-to-end approach comparisons,
not equal-compute policy comparisons. Use `--no-lookahead` on the whole comparison
to compare actor strength directly, or `--with-lookahead` for search-enabled runs.
The modes appear in the table and provenance. These results are explicitly marked
validation/nonofficial; they do not replace the required first 1000 seed-42 scenarios.
Separate test environments and multiple training seeds follow
[SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

## Rules and verification

The [published rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
allow RL training regimes and custom training scenarios, while requiring the original
world/scoring and evaluation distribution. No live dynamics, official metric hooks,
thresholds, one-second sampling, termination or built-in resolution settings were
changed. The canonical evaluator still fixes seed 42 for its official protocol;
other seeds are exposed only through the development validation tool. Organizer
acceptance of the search teacher/hybrid is still not explicitly confirmed; offline
imitation does not itself establish organizer approval.

Verification includes bounded PPO training, supervised actor fitting, PPO refinement
of an actor-only checkpoint, saved inference-mode checks, synthetic timeout analyses,
seed-prefix checks and actual held-out validation smoke runs. Tiny smoke artifacts
are under `.cache`, clearly separate from production models. No full two-million-step
experiment, new production performance result or official evaluation has been run.

## Completed implementation checks

The final suite passed **201 tests**. End-to-end smoke runs trained a ten-aircraft
mixed-scenario PPO for 20 slot steps, refined a distilled student for 20 slot steps,
and validated the student/refined checkpoints on one seed-101 two-aircraft scenario.
Both validation candidates used actor-only inference. These tiny runs establish
pipeline behavior only, not improvement. The original fresh-v6 checkpoint SHA-256
still matches the pre-change benchmark.

A full saved-budget scenario-1 replay completed with nine arrivals and one clean
timeout, KL005. It flagged two 300-second low-progress windows (0..300 and
1200..1500 seconds): observed travel was about 17.1 and 14.1 NM with goal-distance
gains of -0.11 and 0.40 NM, and heading changes exceeding 700 degrees. This is
consistent with circling, but does not identify whether the avoidance was necessary.
The last observed distance was 5.51 NM at 2995 seconds, not the terminal position.

- [Verification manifest](../../.cache/navigation_v6_verification.json)
- [Timeout inventory](../../.cache/navigation_v6/timeout_inventory/diagnostics.md)
- [Full scenario-1 diagnosis](../../.cache/navigation_v6/timeout_scenario1/diagnostics.md)
- [Held-out smoke validation](../../.cache/navigation_v6_validation_smoke/summary.md)
- [Distillation smoke provenance](../../.cache/distillation_smoke_actual/student.distillation.json)

The remaining seven affected scenarios have not been replayed by this implementation
check. No production continuation, targeted training, distilled-policy performance
claim, or official 1000-scenario score has been produced.
