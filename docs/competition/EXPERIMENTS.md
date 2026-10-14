# Reproducible single-agent experiments

Use the project Python environment from the repository root. On this Windows
checkout the executable is `.venv312\Scripts\python.exe`; the commands below
use `python` for readability.

The [initial baseline report](BASELINE_RESULTS.md) records the first measured
comparison of the student's saved model against straight flight.

## Evaluate the existing baseline

```powershell
python -m scripts.evaluate_single_agent --models neutral single_agent_model --episodes 100
```

This is **development evaluation**, using seed 1234 on the first reset and
continuing the random scenario stream thereafter. Both policies see identical
scenarios, verified by scenario hashes. The neutral policy uses the same custom
environment and decision interval as the learned policy. It keeps heading and
speed unchanged; it is a reference, not an RL submission.

Each new directory under `results/single_agent/` contains:

- `manifest.json`: protocol, seeds, package versions, Git state, source hashes,
  model hashes, elapsed time, and completion/failure status.
- `source/`: copies of the relevant code and local competition rules.
- `comparison.md` / `comparison.json`: policy summaries.
- One directory per policy: exact `model.zip` when applicable, `episodes.csv`,
  `summary.json`, `scenarios.jsonl`, and `failures.jsonl`.

Episode metrics are flushed after each episode so interrupted development runs
retain their completed results. Only completed runs should be compared as full
batches. Summary JSON includes all original metrics, their mean and population
standard deviation, failure counts, successful-flight time, and safe goal
completion. Safe completion means reaching the goal with every safety event and
time counter equal to zero. It is an additional diagnostic, not an official score.
Failure categories overlap. Training reward is never used to rank policies.

Keep development seeds consistent when comparing designs, then validate
promising designs on other development seeds and repeat training with several
training seeds. Keeping seed 42 out of development is our experiment policy to
reduce tuning to the reporting scenarios; it is not an additional competition rule.

## Replay failures

Find an `episode_index` in a policy directory's `failures.jsonl`, then run:

```powershell
python -m scripts.replay_single_agent results/single_agent/RUN/01_single_agent_model --episode 0
```

Use `--headless` to check a replay without a window. Replay loads the saved
scenario and saved model; it does not generate official reporting metrics.
Replay uses the current environment implementation. To reproduce older runs
after changing observations, rewards, or actions, restore their source snapshot
and recorded package versions first.

## Train and compare checkpoints

```powershell
python train_single_agent.py train 250000 --seed 0
python train_single_agent.py train 250000 --seed 1
python -m scripts.evaluate_single_agent --models models/single_agent/RUN/model.zip --episodes 100
python -m scripts.evaluate_single_agent --models models/single_agent/RUN/checkpoints/ppo_50000_steps.zip models/single_agent/RUN/model.zip --episodes 100
```

Training writes a new directory under `models/single_agent/`; it never overwrites
the student's existing `single_agent_model.zip`. Each run contains a configuration
and source manifest, PPO `progress.csv`, Monitor log, objective episode metrics,
checkpoints (default every 50,000 steps), and a final `model.zip`. Failed or
interrupted learning saves an `interrupted_model.zip` when a model exists.
PPO completes full rollout batches, so actual timesteps can exceed the requested
number; both are recorded. CPU is the default device. Seeds help reproducibility
but do not guarantee identical results across package versions or hardware.

Allowed algorithm settings are exposed via `--learning-rate`, `--n-steps`,
`--batch-size`, `--gamma`, `--gae-lambda`, and `--device`. This implementation
retains the student's observation, reward, and action design.

BlueSky is global within a process: do not create an evaluation environment
while a training environment is active. Run evaluation separately after training.

## Official reporting

```powershell
python -m scripts.evaluate_competition --env sa --model single_agent_model --out results/official_sa.csv
python -m scripts.evaluate_single_agent --protocol official --models single_agent_model
```

The first command runs the official harness directly. The second wraps that
same harness with model/source snapshots and summaries. Both use **1000 episodes,
seed 42 on the first reset only**, and the original scenario distribution.
The tracked command rejects altered official episode counts or seeds.
Official CSV format and metrics remain unchanged. Official runs do not save
development-style failure scenarios; use development runs for failure analysis.

Only the official harness's two designated edit points have changed: the
single-agent builder now loads `SingleAgentEnv`, and the policy loader loads PPO.
The scoring loop, seed stream, table and CSV writer are unchanged. The PPO loader
deliberately rejects multi-agent models until a suitable loader is implemented.

Evaluation checks the fixed constructor settings (10 intruders, 5 obstacles,
5 NM separation, 5 km goal capture, 3000 simulated seconds, 1-second sampling,
and the original aircraft setup) and rejects overridden scoring/step methods or
a custom scenario generator. These checks detect common accidental changes;
source snapshots support review of dynamics and simulator configuration.
The base environment continues to use A320 dynamics and `RESO OFF`.

The existing single-agent subclass changes permitted reward/action hooks and
inherits observations. Its heading/speed scales and decision interval remain
permitted tuning choices, including at evaluation. No scoring, scenario,
simulation, or termination code has been changed. See
[the competition rules](COMPETITION.md) for the authoritative requirements.

## Verify the experiment tooling

```powershell
python -m unittest discover -s tests -v
```
