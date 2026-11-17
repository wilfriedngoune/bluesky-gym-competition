# bluesky_gym/scripts
 
This folder contains example training loops for the different scenarios in bluesky_gym/envs

The current v5 response experiment runs with 5-second decisions and command
deadbands of 1 degree and 1 knot:

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_responsive_v5
```

It continues the saved 5-second baseline and writes a new experiment directory.
Use `--fresh` for fresh weights or `--dry-run` to inspect its exact command.
See [the v5 guide](../docs/competition/LOOKAHEAD_AND_ZONE.md) for configuration,
comparison and targeted trajectory replay.

The optional v5 separation filter can be evaluated with the current checkpoint:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5_responsive/model.zip --episodes 20 --safety-filter --out models/ma_navigation_v5_responsive/safety_filter.evaluation.csv
```

It screens forecast root commands against an 8 NM target and remains off by default.
See [the separation filter experiment](../docs/competition/SEPARATION_FILTER.md) for
configuration, limitations, test results and an optional parallel development runner.

For cheaper v5 replay/evaluation, add `--lookahead-preset fast` (12 simulations,
maximum depth 3). Defaults retain the saved budget. Evaluation also accepts
`--lookahead-simulations` and `--lookahead-depth` overrides.
See [fast lookahead](../docs/competition/FAST_LOOKAHEAD.md) for timings, commands and
limits; a smaller search budget can change safety and arrival results.

The navigation v6 experiment tools are now available:

- `python -m scripts.train_navigation_v6`: ten-aircraft continuation, optional targeted obstacle mixture.
- `python -m scripts.diagnose_navigation_timeouts`: inventory/replay missed arrivals.
- `python -m scripts.distill_navigation_policy`: separate search-labelled actor imitation.
- `python -m scripts.validate_navigation_checkpoints`: matched held-out validation seed streams.

See [the v6 experiment guide](../docs/competition/NAVIGATION_V6_EXPERIMENTS.md) for
separate output paths, PPO refinement, commands and experimental limitations.
