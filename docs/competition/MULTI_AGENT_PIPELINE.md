# Multi-agent navigation pipeline

For the opt-in `--mdp navigation-v5` experiment with policy/value-guided tree
search and soft shrinking waypoint zones, see
[Lookahead and shrinking zones](LOOKAHEAD_AND_ZONE.md). V5 requires fresh or
matching v5 weights; its search uses a separate predictor during inference and
preserves the authoritative competition simulator and scoring.

The multi-agent pipeline uses one shared PPO `MultiInputPolicy` for all aircraft,
starting from the observation, reward and action design in `SingleAgentEnv`.
`bluesky_zoo.multi_agent_env.MultiAgentEnv` inherits the official multi-agent
environment's simulation, scenario generation, termination and scoring.

The opt-in `--mdp navigation-v2` uses a corrected MA-only reward with the same
observations and actions. See the [change overview](CHANGE_OVERVIEW.md) for its
weights, benefits/tradeoffs, and a command that preserves the existing baseline.
Select the same MDP for training, replay, evaluation and recording. The default
`navigation` MDP remains available for reproducing earlier runs.

## Navigation v4: narrow-passage experiment

Select `--mdp navigation-v4` for the passage experiment. It retains v3's
traffic features, reward defaults, five-second decisions and original
simulation/scoring. PPO may choose any heading: its normalized output maps to
an offset of up to 180 degrees around the inherited navigation reference,
including directions parallel to an obstacle. The reference still uses goal,
obstacle and sector modes; this change removes its previous +/-70-degree
restriction, rather than adding a route planner.

The configurable avoidance activation/release clearances default to 0.5/1 NM,
via `--obstacle-buffer-nm` / `--obstacle-release-nm`. These are action-mapping
buffers, not changes to the restricted polygons. Use
`--max-heading-offset-deg` to experiment with the heading range. For a legacy
mapping ablation within v4, use `--max-heading-offset-deg 70
--obstacle-buffer-nm 3 --obstacle-release-nm 4`.

New `clearance_rays` inputs contain 24 normalized distances at body-relative
angles -180,-165,...,165 degrees (0 is forward). They measure the first actual
sector/obstacle edge within `--lookahead-nm` (default 20 NM), without inflating
the obstacles. Rays are zero inside an obstacle/outside the sector. They are
static geometric observations, not turn-feasibility or traffic-safety checks.
The existing traffic inputs still describe the other aircraft.

**Start a fresh model:** v3 checkpoints lack these inputs and have different
heading semantics, so direct v3-to-v4 initialization is rejected. All v4
densities share spaces. A short first training experiment is:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py train 100000 --mdp navigation-v4 --curriculum 2 4 6 10 --passage-training-probability 0.25 --heading-deadband-deg 1 --speed-deadband-kt 1 --model models/ma_navigation_v4/model --metrics models/ma_navigation_v4/training_metrics.csv --checkpoint-dir models/ma_navigation_v4/checkpoints --eval-episodes 20
```

This divides the total budget over four traffic stages; it is a short pilot
for a new policy, not equivalent to another 100k continuation of trained v3.
More training may be needed. This example explicitly enables the experimental
1-degree/1-knot deadband; omit those flags to keep fresh-run deadbands disabled.

`--passage-training-probability` defaults to zero. When enabled for v4 training
only, that fraction of resets draws a rotated custom 3..8 NM corridor example;
the remainder uses the original generator. The examples have five obstacles
and 2..10 aircraft with separated starts. They teach approach, passage entry
and multi-aircraft interaction, but need not all admit safe simultaneous
passage. Evaluation/replay/GIFs never install this custom generator. They use
the original scenario distribution, ten aircraft and five obstacles by default.

Saved checkpoints/JSON include the passage heading/buffer/range profile;
training inherits it from matching v4 weights unless explicitly overridden.
The training mixture probability is recorded in run metadata and remains an
explicit training setting. Evaluation and recording restore the saved profile.

To inspect a failure with either an old v3 or new v4 checkpoint, log a replay:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_navigation_v3/model.zip --mdp navigation-v3 --episodes 4 --seed 42 --trajectory-csv models/ma_navigation_v3/trace_seed42.csv
```

Seed 42 is set only on the first reset; the fourth episode is scenario 4 in
that stream, not a reset with seed 4. CSV episode indices are zero-based.
Each row captures geometry/position before the action, the selected command,
navigation mode and active obstacle at command time, and metric totals after
the step. Finished aircraft retain their final row. Compare repeated mode or
obstacle switches with stalled goal progress. To replay trained v4, change the
model path and `--mdp navigation-v4`; add the same trace flag.

Checks include a physical straight-ahead traversal of a 4 NM corridor without
restricted-area entries, ray intersections/edge cases, configuration round-trip,
fixed spaces at each density and identical official metrics for identical
issued commands. These establish implementation behavior, not learned-policy
improvement. Smaller buffers/full headings can also worsen safety, and existing
failed trajectories still require inspection to establish their actual cause.

## Navigation v3: all traffic and curriculum

`--mdp navigation-v3` implements the next configuration: all nine other aircraft,
an explicit `intruder_mask`, normalized `remaining_time`, and current intrusions
sorted before predicted encounters. Observation shapes are fixed at every
2-to-10-aircraft training density. The goal, obstacles, navigation reference,
heading/speed actions, world dynamics, scoring and termination are unchanged.

V3 defaults to `gamma=0.998`, a 180-second predictive warning horizon, warning
cost up to 0.05 per second and an 8 NM predicted closest-approach buffer. These
are tunable training signals; actual separation scoring remains 5 NM. Override
the profile with `--gamma`, `--conflict-horizon`, `--conflict-risk-cost`, and
`--conflict-buffer-nm`. Defaults and overrides are saved inside every checkpoint
and restored by evaluation, replay and GIF recording.

Start fresh because the larger observation is incompatible with SA/v2 models:

```powershell
python train_zoo.py train 2000000 --mdp navigation-v3 --curriculum 2 4 6 10 --model models/ma_navigation_v3/model --metrics models/ma_navigation_v3/training_metrics.csv --checkpoint-dir models/ma_navigation_v3/checkpoints
python train_zoo.py replay models/ma_navigation_v3/model --mdp navigation-v3
```

The total requested timestep budget is split equally across the listed stages.
Each stage starts in a separate process, transfers the preceding policy/value
weights with a fresh optimizer, and writes its own metrics and training report.
Its seed is the initial seed plus its stage index. Earlier stages skip evaluation;
only the final ten-aircraft stage runs the requested `--eval-episodes` (default
1000). Add `--eval-episodes 20` for a quick nonofficial check. Stages advance after
their assigned budget, with no automatic success threshold or performance promise.

Intermediate files and `curriculum.json` are under `model_curriculum/` next to
the final model. PPO's final timestep counter is stage-local; the manifest and
final model JSON include the summed actual count. Full rollout buffers can
overshoot each requested stage budget. A matching v3 checkpoint can initialize
a new run via `--init-model`. Omit `--curriculum` to train directly with ten aircraft.

Full v3 performance remains to be measured; compare clean arrivals, completion,
intrusion events and duration on matched validation scenarios against the saved
v2 baseline. See the [change overview](CHANGE_OVERVIEW.md) for benefits/tradeoffs.

## Rules checked

Checked against the upstream
[competition rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
and [getting-started guide](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/GETTING_STARTED.md)
on 23 October 2026.

- Observation, reward, heading/speed action mapping and decision frequency may change.
- The approach must use reinforcement learning; horizontal information/actions only.
- Preserve `_update_metrics`, `_get_info`, A320 dynamics and `RESO OFF`.
- Keep 5 NM separation, 5 km goal capture, 3000 simulated seconds and one-second metric sampling.
- Official multi-agent evaluation uses 10 agents, 5 obstacles and the original scenario distribution.
- Report the first 1000 episodes of seed 42, seeding only the first reset.
- Count each aircraft's final metrics, including early arrivals. A conflict contributes to both participants.
- Submission includes a 3–4 page report, a five-scenario video and reproducible code/model on request.

The inherited action mapping uses a heuristic navigation reference for goals,
restricted-area buffers and sector recovery, plus PPO-controlled heading offsets
and speed. Document this residual-control design in the report. The rules allow
custom action mapping and require RL; they do not explicitly approve this
particular hybrid. A successful zero-action rollout alone is not evidence of
learned conflict resolution.

## What transfers from single-agent

The navigation MDP keeps the same Dict keys, shapes and normalization as
`SingleAgentEnv`, including five tracked traffic slots, polygon geometry, goal
direction/distance, previous action and navigation mode. Traffic features now
describe other live controlled aircraft. Each aircraft has its own action
history, goal-progress history and obstacle-buffer state.

Actions keep the existing meaning: a heading offset up to 70 degrees around the
navigation reference and an absolute speed target from 120 to 220 knots.
Decisions occur every five simulated seconds. All aircraft use the same policy
weights, while their observations and histories remain separate.

The trainer adapts the PettingZoo parallel environment to SB3 without creating
multiple BlueSky simulators in one process. Evaluation and replay use the raw
parallel environment and apply the policy only to live aircraft. PPO checkpoints
consume Dict observations directly; they do not require flattening.

## Train or warm-start

Run commands from the repository root with the project's Python environment
(`uv run python` can replace `python`). Training defaults to the navigation MDP.

```powershell
# Train the shared policy from scratch.
python train_zoo.py train 1000000 --model multi_agent_model --metrics multi_agent_training_metrics.csv

# Transfer a compatible existing single-agent policy, then learn MA behavior.
python train_zoo.py train 1000000 --init-model single_agent_model.zip --model multi_agent_model --metrics multi_agent_training_metrics.csv

# Shorter rollout buffers for a quick training smoke check.
python train_zoo.py train 320 --n-steps 32 --batch-size 64 --model models/ma_smoke --metrics models/ma_smoke_metrics.csv --eval-episodes 0
```

`--init-model` copies compatible policy weights and starts a fresh PPO optimizer;
it does not resume single-agent optimizer state. It fails on incompatible
observation or action spaces. Training writes a model, competition metrics CSV,
periodic checkpoints and adjacent JSON run metadata. The CSV records per-aircraft
results, so ten arrivals/timeouts produce ten rows per scenario.

Training timesteps count agent slots, including absorbing slots after arrival:
with 10 aircraft, one joint decision adds 10 timesteps. PPO collects complete
rollout buffers, so a short request can
run past its requested timestep count. BlueSky is process-global: avoid callbacks
that run a second environment in the same training process. Run evaluation as a
separate command after training.

Use `--mdp base` to train on the original observation/reward/action mapping, or
`--n-agents` for a training curriculum. Default evaluation still uses 10 agents.
The legacy `python train_zoo.py 250000` invocation remains supported.

## Active-aircraft training samples

New training runs exclude departed aircraft's zero-filled padding from PPO
updates. Each aircraft's actual arrival/timeout transition remains included.
SB3 first computes returns on the full time/aircraft grid, including its normal
timeout bootstrapping; then the buffer filters inactive transitions before
advantage normalization, policy/value/entropy losses and explained variance.
This changes the learner only, not simulation, reward, scoring or evaluation.

The trainer logs `rollout/active_samples`, `rollout/padding_fraction` and
`time/active_timesteps`. Model JSON includes `mask_inactive`, `active_timesteps`
and `absorbing_timesteps`. Budgets and PPO's `num_timesteps` still count all
vector slots. Fewer samples reach the optimizer per rollout; a better policy
must be demonstrated through validation, not inferred from this correction.

Existing v3 weights remain compatible. For example, continue at full density
with the current reward, using separate outputs:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py train 1000000 --mdp navigation-v3 --n-agents 10 --seed 4 --init-model models/ma_navigation_v3/model.zip --model models/ma_navigation_v3_active/model --metrics models/ma_navigation_v3_active/training_metrics.csv --checkpoint-dir models/ma_navigation_v3_active/checkpoints
```

This starts a fresh optimizer and runs the usual post-training evaluation.
For an ablation of filtering, add `--include-inactive-samples` with otherwise
identical settings and different output paths. That flag also passes through
all curriculum stages. It restores the previous unfiltered sampling behavior;
the active/padding counters remain available in either mode. Other earlier
fixes, including heading formatting, still apply.

## Optional command deadband

`--heading-deadband-deg` and `--speed-deadband-kt` hold the previous accepted
command when the proposed change is strictly smaller than its threshold.
Heading differences use the shortest circular angle. Heading and speed are
filtered independently; the first command of each aircraft/scenario always
passes. Comparisons use the last accepted target, so small requested changes
can accumulate past the threshold. Larger corrections pass immediately.
This is command hysteresis, not a turn-rate limit, and cannot eliminate large
left/right reversals or jumps in the navigation reference.

Try a visual preview with the original model and experimental 1-degree/1-knot
thresholds (these are tuning values, not certified aircraft limits):

```powershell
.\.venv312\Scripts\python.exe train_zoo.py replay models/ma_navigation_v3/model.zip --mdp navigation-v3 --episodes 3 --seed 42 --heading-deadband-deg 1 --speed-deadband-kt 1
```

This replay-only override does not edit the saved checkpoint. To train with
the same filtering, use separate outputs:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py train 100000 --mdp navigation-v3 --seed 4 --init-model models/ma_navigation_v3/model.zip --heading-deadband-deg 1 --speed-deadband-kt 1 --model models/ma_navigation_v3_deadband_100k/model --metrics models/ma_navigation_v3_deadband_100k/training_metrics.csv --checkpoint-dir models/ma_navigation_v3_deadband_100k/checkpoints --eval-episodes 20
```

New checkpoints save `competition_action_config`; adjacent JSON records
`action_config`. Training inherits a compatible checkpoint's deadband unless
explicitly overridden; `--heading-deadband-deg 0 --speed-deadband-kt 0` disables
it. Old checkpoints and fresh runs default to zero. Replay, evaluation and GIF
recording automatically restore saved settings. Curriculum stages forward
explicit settings and inherit them with weights.

This changes the permitted action mapping only. The simulation/scoring and
policy input shapes stay unchanged. Each aircraft keeps internal accepted-
command history; `previous_action` still contains the raw clipped policy
output, and the existing smoothness reward still measures that output.
No accepted-command observation features or new reward terms are added.
Filtering may suppress useful small corrections, so compare arrival/safety
metrics as well as visual smoothness. No improved performance is established.

## Optional boundary reward experiment

The 20-scenario continuation comparison showed 37 more sector exits, with 33
of that increase concentrated in scenarios 4, 7 and 10 of the seed-42 stream.
Repeated crossings are a hypothesis to investigate; their cause has not been
confirmed by replay. The next isolated reward experiment provides feedback
before leaving the sector, without changing observations, action mapping or
the official sector polygon/scoring.

Enable it with `--sector-buffer-cost 0.01 --sector-buffer-nm 5`. Inside the
sector, `risk = (1 - clip(clearance_nm / buffer_nm, 0, 1)) ** 2`; outside,
`risk = 1`. The added component is `-cost * risk * elapsed_seconds`.
It is zero at least 5 NM inside, rises smoothly toward the boundary and stays
at maximum outside. No return/escape bonus is paid. Existing sector-event and
outside-duration penalties, all other reward terms and goal bonus stay as-is.
Risk uses the current geometry at each reward decision, scaled by actual
elapsed seconds; it does not add an exact one-second occupancy measurement.

Cost defaults to zero for v2/v3 and old checkpoints. The 5 NM warning buffer is
a training setting, distinct from the fixed 5 NM aircraft separation minimum.
Configuration is saved inside the policy and reproduced by evaluation/replay.
The training CSV adds `reward_sector_buffer` (zero when disabled). A strong
cost can encourage unnecessary detours or hurt arrivals near the boundary;
0.01/s is an initial test value, not a demonstrated optimum.

For a controlled 100k-step comparison against `ma_navigation_v3_active_100k`,
initialize the original v3 checkpoint again with the same seed/settings:

```powershell
.\.venv312\Scripts\python.exe train_zoo.py train 100000 --mdp navigation-v3 --n-agents 10 --seed 4 --init-model models/ma_navigation_v3/model.zip --sector-buffer-cost 0.01 --sector-buffer-nm 5 --model models/ma_navigation_v3_boundary_100k/model --metrics models/ma_navigation_v3_boundary_100k/training_metrics.csv --checkpoint-dir models/ma_navigation_v3_boundary_100k/checkpoints --eval-episodes 20
```

Both continuations use active-only PPO updates; the new term is the intended
reward difference. Compare goal completion, clean arrivals, boundary events
and occupancy, intrusion metrics and flight time against both saved baselines.

## Inspect and evaluate

Every newly started `train_zoo.py train` run now prints a totals table in the
competition sharing format and saves Markdown/JSON reports. After saving and
closing training, it starts deterministic evaluation in a fresh Python process
to isolate BlueSky state. The default is 1000 scenarios, seed 42, 10 aircraft
and 5 obstacles, even if training used fewer aircraft. Evaluation reports all
10000 final aircraft records. This adds runtime after training has finished.

For `--model models/ma_reward_v2/model`, the output files are:

- `model.training_report.md` / `.json`: completed training aircraft, labeled non-evaluation.
- `model.evaluation.csv`: final-policy evaluation records.
- `model.evaluation.md` / `.json`: totals, completion rate and clean-flight counts.

Add `--eval-episodes 20` for a shorter nonofficial check, or `--eval-episodes 0`
to save only the training summary. Reduced-traffic base-MDP models can have
incompatible observation shapes at 10 aircraft; skip evaluation during those
curriculum stages and evaluate after training with the full configuration.

`violation-free` means no intrusion/restricted-area/sector events AND no time
in those violations; a clean timeout is included. `violation-free_success`
additionally requires reaching the goal. Both definitions are printed in each
report; they are derived statistics, not changes to the official metric layer.
Missing track columns are `NA`, never filled with guessed single-agent scores.

Already-running Python processes do not pick up this change automatically.
Create a training snapshot or combine existing SA/MA evaluation CSVs with:

```powershell
python -m scripts.competition_report --ma models/ma_reward_v2/training_metrics.csv --training --out models/ma_reward_v2/model.training_snapshot
python -m scripts.competition_report --sa sa_eval.csv --ma ma_eval.csv --episodes 1000 --seed 42 --out results/comparison
```

Imported CSV reports cannot verify their provenance and are labeled accordingly.
The evaluation script also saves the sharing template next to any `--out` CSV;
`--env both` additionally produces a combined table. Per-aircraft conflict totals
are not divided by two. Total reward remains separate from the comparable metrics.

```powershell
# Visual replay of three scenarios.
python train_zoo.py replay multi_agent_model --mdp navigation --episodes 3 --seed 0

# Quick metric check; explicitly marked NONOFFICIAL because it uses 5 scenarios.
python -m scripts.evaluate_competition --env ma --mdp navigation --model multi_agent_model --episodes 5 --out models/ma_quick.csv

# Reportable MA protocol: 1000 scenarios, 10000 final aircraft records, seed 42.
python -m scripts.evaluate_competition --env ma --mdp navigation --model multi_agent_model --out competition_eval_ma_navigation.csv

# Compare the original single-agent checkpoint using the same evaluation protocol.
python -m scripts.evaluate_competition --env sa --mdp navigation --model single_agent_model --out competition_eval_sa_navigation.csv
```

Evaluation and GIF recording default to `--mdp base` to preserve the original
demo behavior. Always pass `--mdp navigation` for the custom pipeline. Checkpoint
space mismatches fail with an explanatory error. Old flattened `MlpPolicy`
checkpoints need their original flattening wrapper; navigation uses
`MultiInputPolicy` checkpoints. Omitting `--model` runs a zero-action baseline.

The evaluation harness retains the official metric aggregation and continuous
seed-42 scenario stream. Nondefault episode or MA agent counts are explicitly
marked `NONOFFICIAL CHECK`. Training reward is a diagnostic; judge performance
by goal completion, safety events/occupancy and flight time.

Evaluation and recording batch all live aircraft into one shared-policy
prediction per decision and use one Torch CPU thread for these small networks.

If evaluation prints `txt2hdg needs either a float ...`, BlueSky has rejected a
heading command and continued the simulation without applying it. The shared
navigation action mapping now writes headings in decimal notation, preserving
their exact float value; BlueSky cannot parse scientific notation near zero.
It also rejects nonfinite actions/targets with the aircraft ID before sending
commands. A running process retains the old code. To evaluate a saved v3 model
with the fix, start a new evaluation process; retraining is unnecessary. Use a
separate output path to preserve the earlier evaluation:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-v3 --model models/ma_navigation_v3/model --episodes 1000 --out models/ma_navigation_v3/model.evaluation_fixed.csv
```

## Record the submission reel

```powershell
python -m scripts.record_competition_gifs --env ma --mdp navigation --model multi_agent_model --seeds 1 2 3 4 5 --out multi_agent_reel.gif
```

The recorder loads the same MDP and deterministic policy as evaluation. It
defaults to the full simulated-time budget, which is up to 600 navigation
decisions. `--max-steps 20` provides an explicit short preview. GIF frames are
subsampled across each scenario, including its final frame.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The evaluation/reel tests check seed-stream continuity, early arrivals and
timeouts, checkpoint mismatch reporting, deterministic Dict inference, cleanup
on failure, nonofficial labeling and the complete navigation recording budget.
Smoke training establishes that the pipeline runs; it does not establish
competitive performance. Train sufficiently and run the full evaluation before
reporting competition results.
