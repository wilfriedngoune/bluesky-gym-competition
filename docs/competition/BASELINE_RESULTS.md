# Initial single-agent baseline — 14 October 2026

These are **development results**, not the official seed-42 reporting run.
Each policy was evaluated on the first 100 scenarios of seed 1234, seeding
only the first reset. Scenario hashes confirmed identical scenarios for both
policies. Actions were deterministic and the original competition world and
scoring methods were retained.

Command used with this checkout's `.venv312\Scripts\python.exe`:

```powershell
python -m scripts.evaluate_single_agent --models neutral single_agent_model --episodes 100 --out-dir results/single_agent/baseline_dev_1234
```

The archived PPO model records 250,880 training steps, `n_steps=1024`,
`batch_size=256`, learning rate 0.0003, gamma 0.99, and GAE lambda 0.95.
It records no training seed. These are read from the model archive; the original
training source and full history cannot be inferred from that archive alone.
Both evaluated policies used the current `SingleAgentEnv` implementation.

## Results

| Metric | Neutral reference | Saved PPO model |
| --- | ---: | ---: |
| Goal completion | 100% | 12% |
| Goal reached with no safety violations | 15% | 2% |
| Flight time, all episodes [s] | 860.82 | 2885.55 |
| Flight time, successful episodes only [s] | 860.82 | 2046.25 |
| Mean intrusion events | 1.23 | 1.76 |
| Mean intrusion time [s] | 101.23 | 133.28 |
| Mean restricted-area entries | 0.72 | 2.43 |
| Mean restricted-area time [s] | 114.00 | 474.45 |
| Mean sector exits | 0.00 | 0.41 |
| Mean time outside sector [s] | 0.00 | 169.32 |

The saved PPO policy missed the goal in 88 scenarios, incurred loss of separation
in 72, entered restricted areas in 66, and left the sector in 23. Categories
overlap. Safe goal completion is a supplementary diagnostic, not an official
competition score. The straight-flight reference is a diagnostic baseline,
not an eligible RL solution.

The current model performs worse on these scenarios across the task and safety
metrics. This result does not establish the cause. It is one development seed
and one saved training run; it does not establish general performance or show
which reward/observation component is responsible.

## Next experiment

Prioritize learning reliable goal completion. Replay several failures first to
distinguish persistent turning, ineffective steering, obstacle deadlocks, and
speed behavior. The saved per-episode metrics alone cannot identify those causes.

The next controlled comparison should test the existing training setup against
a navigation-first training curriculum, introducing obstacles and scripted
traffic progressively, then finishing on the original competition distribution.
The rules permit custom training scenarios; final evaluation must retain the
original distribution and fixed settings. Curriculum support is not implemented
in this first tooling change: its current training command uses the original
environment defaults.

Keep the initial baseline archived. Compare new policies on the same development
scenarios, validate promising improvements on additional development seeds,
and repeat training with several training seeds before selecting a submission.
Use the unchanged official 1000-episode seed-42 harness for final reporting.

## Artifacts and validation

The complete local run is in `results/single_agent/baseline_dev_1234/`, including
source/model snapshots, hashes, package versions, raw metrics, summaries, saved
scenarios, and failure indices. See [the experiment guide](EXPERIMENTS.md) for
commands to compare checkpoints and replay failures.

Nine regression tests passed for seed streams, safe-completion classification,
protocol enforcement, fixed-world checks, failure cleanup, and model-loader
scope. A 640-step training smoke run saved its final model, five checkpoints,
PPO logs, and a completed-episode metric record. A headless replay of saved PPO
episode 0 reproduced all nine final metrics exactly.

An AST comparison against the repository version verified that edits to the
official harness are confined to its two designated edit points. The base
competition environment, scenario generator, actions, and geographic scoring
helpers have no code changes. No full official 1000-episode run was performed
as part of this development baseline.
