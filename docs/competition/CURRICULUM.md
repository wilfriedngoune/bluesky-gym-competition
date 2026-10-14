# Single-agent training curriculum

The curriculum command trains one PPO model through increasingly difficult
versions of the task. It keeps the student's reward, action mapping, network,
physics, scoring and observation implementation unchanged. The competition
permits custom training scenarios; official evaluation must use the original
distribution. See [the competition rules](COMPETITION.md).

## Start a training experiment

From the repository root, using this checkout's Python environment:

```powershell
.\.venv312\Scripts\python.exe train_single_agent.py curriculum --seed 0
```

This starts a new model, preserving the student's archived baseline. It can
request up to 1.5 million training steps across six stages with the defaults,
plus evaluation time. It stops earlier if a stage exhausts its budget without
meeting the validation targets. This command starts a substantive training run;
the implementation was verified with a small smoke run, not this full budget.

| Stage | Retained obstacles | Retained scripted intruders |
| --- | ---: | ---: |
| navigation | 0 | 0 |
| obstacles_sparse | up to 2 | 0 |
| obstacles_full | up to 5 | 0 |
| traffic_sparse | up to 5 | 3 |
| traffic_medium | up to 5 | 6 |
| full | original distribution, up to 5 | 10 |

Each easier stage first generates an original full scenario, then removes a
suffix of its obstacles and intruder routes. Sector geometry, initial aircraft
state, destination, hazard geometry, and RNG consumption remain unchanged.
The final stage restores the original generator without a scenario filter.
The original generator may produce fewer than five obstacles when placement
attempts are exhausted; the curriculum does not change that behavior.

Observation capacities stay fixed at ten aircraft and five obstacles. Existing
observation components zero-pad absent hazards; no `_get_obs()` override was
added. The policy and optimizer continue across stages. Each stage transition
starts a new episode, discarding any unfinished training episode from the previous
stage. Training CSVs contain only completed episodes.

## Advancement and evaluation

By default, the trainer:

1. Trains in 50,000-step chunks, with a maximum requested 250,000 steps per stage.
2. Saves a checkpoint and checks it on 20 stage-validation scenarios after each chunk.
3. Requires at least 100,000 stage steps and two consecutive checkpoint checks
   with at least 95% goal completion and 80% safe goal completion before advancing.
4. Evaluates each stage's last checkpoint on the **full original competition
   distribution**, using the development seed, before advancing or stopping.

Thresholds are experimental targets, not official scoring thresholds or a
guarantee of performance. The two checks use the same development scenarios,
not independent samples. Twenty episodes give a coarse initial signal; use
larger samples and additional development seeds for serious model selection.
PPO rounds requested chunks to complete rollout batches, so actual steps can
exceed a stage's requested budget. Actual counts are recorded.

Change the budget or validation sample size explicitly:

```powershell
.\.venv312\Scripts\python.exe train_single_agent.py curriculum --seed 0 --stage-budget 500000 --eval-episodes 50
```

Other controls are `--chunk-steps`, `--min-stage-steps`, `--goal-threshold`,
`--safe-threshold`, `--passes-required`, and `--eval-seed`. Validation seed 1234
is the default; it must differ from the training seed and official seed 42.
The regular PPO settings and checkpoint interval remain available.

Validation runs in child processes using the same Python executable. This
preserves the live training environment because BlueSky has global state
within a process. Child-process output is saved in adjacent `.log` files.

## Inspect the run

New runs appear under `models/single_agent/curriculum_seed0_TIMESTAMP/`.

- `manifest.json` records source/package versions, configuration, stage checks,
  full-distribution results, actual steps and final status.
- `latest_model.zip` is the most recent trained checkpoint.
- Stage directories hold episode logs, PPO logs, regular checkpoints,
  checkpoint-validation snapshots and a stage-final `model.zip`.
- `validation_STEPS/` contains **curriculum validation** results, saved scenarios
  and the exact model. Its protocol is `curriculum_validation`, not official reporting.
- `full_evaluation/` contains ordinary development evaluation on the original
  distribution. Its scores are the relevant comparison with the archived baseline.
- The run-root `model.zip` is saved only when every stage meets its gate.

`stage_not_mastered` means the trainer stopped normally at an unmet target.
Do not treat `latest_model.zip` as a completed curriculum merely because it
exists. Inspect failure replays before increasing the training budget or changing
other parts of the learning setup. If necessary, resume research with another
new run; restoring a stopped curriculum's full trainer state is not implemented.

Replay a stage-validation failure using its saved scenario:

```powershell
.\.venv312\Scripts\python.exe -m scripts.replay_single_agent models/single_agent/RUN/00_navigation/validation_STEPS --episode 0
```

## Compare and report

Use the existing full-distribution evaluator to compare a curriculum checkpoint
with the archived baseline on identical development scenarios:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_single_agent --models single_agent_model models/single_agent/RUN/model.zip --episodes 100
```

Evaluate the completed curriculum on additional development seeds and repeat
training seeds before making performance claims.

To measure the curriculum's effect, also train a new non-curriculum control
with the same seed, PPO settings and total actual training steps (read from
the curriculum manifest). The archived student's model is a useful reference,
but its different training seed/history and budget do not isolate the effect
of the curriculum.

For official reporting:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_single_agent --protocol official --models models/single_agent/RUN/model.zip
```

The official environment builder is unchanged by this curriculum feature.
It uses the original scenario generator, fixed counts/settings, and the existing
1000-episode seed-42 protocol. The evaluation guard rejects the easier training
generator; it is available only through the explicitly labeled stage validator.
The baseline trainer remains available for matched experiments without a curriculum.

## Implementation verification

Sixteen regression tests cover scenario/RNG preservation, fixed observation
shapes, rejection of easier scenarios by the normal evaluation guard, stage
gates, consecutive-pass resets, checkpoint continuity and the existing
evaluation protocol. A real 384-step smoke run exercised all six stages, each
with stage validation and full-distribution development evaluation. Advancement
targets were deliberately set to zero in that smoke run to exercise transitions;
its completion is not evidence of learned competence. The full-budget training
experiment has not been run yet.

A separate real 128-step run with normal pass requirements verified that an
unsuccessful navigation stage stops with `stage_not_mastered`, preserves its
checkpoints, and does not save a completed curriculum model at the run root.
