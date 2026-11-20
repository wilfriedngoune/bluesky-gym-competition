# Shared single-/multi-agent navigation experiment

The opt-in MDP `navigation-transfer` combines the useful features identified in
both implementations. The existing `navigation`, v2-v5 environments and trained
checkpoints keep their original contracts. The current legacy single-agent file
received only the missing NumPy, BlueSky and Gymnasium-space imports; its controller
and reward formulas were not silently replaced.

## Implemented transfers

Both new hosts use one navigation-state implementation with identical observations,
normalization, reward defaults and action meanings:

- Ten detailed traffic slots with an explicit mask. The original SA world has ten
  scripted intruders; the MA world has up to nine live neighbours and pads the rest.
  Actual intrusions remain prioritized before predicted-conflict urgency.
- Twenty-four polygon-clearance rays, using the existing MA polygon implementation.
- Normalized time-to-go and deadline slack. The nominal estimate uses straight-line
  distance to the capture radius and the configured maximum target speed. It ignores
  turns, detours and speed/groundspeed differences; it is not an arrival guarantee.
- Observable recent progress and stalled duration, plus obstacle-side commitment.
  These history values are aircraft-local and reset every scenario. Refreshing an
  observation twice at the same simulated time does not increment the counters.
- Optional side commitment chooses an initial tangent relative to the obstacle's
  outward normal. After 150 seconds with insufficient progress on that obstacle,
  the reference may switch side once. A physically inside-obstacle escape retains
  the existing nearest-boundary reference. The actor keeps its full heading freedom.
- Configurable metric-based reward components and per-component training CSV logs.
  They are the shared MA potential-based progress, goal, intrusion, restricted-area,
  sector, warning, flight-time, smoothness, aircraft-buffer and soft-zone components.

The `navigation_memory` feature is `[side, stalled_fraction, progress_rate, switched]`.
Side is -1/0/+1; stalled duration is divided by the episode budget; progress rate is
scaled by the configured maximum speed; the last field indicates the current side
switch. Together with the geometric features this makes the controller memory
visible to the policy instead of adding hidden avoidance state.

Both hosts use a reference-relative heading offset (default +/-180 degrees) and
absolute target CAS speed (default 120..220 kt), at five-second command intervals.
The same action vector has the same meaning and feature normalization in both hosts.
This intentionally differs from the independently evolved legacy SA action mapping.
No hard-deadline direct-flight rescue, deadline safety-penalty reduction or additional
traffic-heading safety filter was copied into the new MDP.

## Explicit checkpoint compatibility

These observations have different keys/shapes from existing SA and v5 models.
Start a fresh `navigation-transfer` model; loading an older model is rejected.
After a new shared-contract checkpoint exists, its weights can initialize the other
host with the same saved configurations. The policy/value weights transfer into a
fresh PPO optimizer; rollout and optimizer state are not resumed. Shared dimensions
are necessary for transfer, not proof of equal performance: scripted intruders and
reactive learning aircraft remain different traffic distributions.

New checkpoints default to actor-only inference. Single-agent evaluation and replay
can now explicitly enable an adapted learned-policy/value lookahead with
`--with-lookahead`. It models the shared ten-slot/deadline/memory inputs and scripted
traffic; the old v5 predictor is not used with an incompatible observation layout.
Multi-agent `navigation-transfer` search remains unsupported. Old v5 search remains
available and unchanged. Models store contract version 1, reward/action/zone/passage/
transfer configurations, environment of training origin and actor-only inference intent.

## Train single-agent, then transfer into multi-agent

Fresh SA example (select a suitable training budget):

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_transfer_navigation --env sa --model models/navigation_transfer_sa/model --timesteps 1000000 --seed 20
```

Then initialize MA training from that compatible checkpoint:

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_transfer_navigation --env ma --init-model models/navigation_transfer_sa/model.zip --model models/navigation_transfer_ma/model --timesteps 2000000 --seed 21
```

The reverse direction is supported by swapping `--env` and initializing from the MA
shared-contract checkpoint, with another unused output directory. Each run refuses
existing model/metrics/checkpoint artifacts. Add `--dry-run` to validate inputs and
inspect the plan without creating an environment or training. The original v5 model
cannot initialize this profile; use it separately as an evaluated baseline.

SA timesteps are single-aircraft decisions. MA timesteps are agent slots, including
absorbing slots after arrivals; those are masked from policy updates, and active
versus absorbing totals are recorded. Equal numeric budgets do not imply equal
simulation exposure or compute. Checkpoints default to every 10,000 slot/decision
steps. No automatic long post-training evaluation is launched.

## Configure features, mappings and reward coefficients

`--transfer-config config.json` accepts the fields of `TransferConfig`, for example:

```json
{"side_commitment": false}
```

Fields include `min_target_speed_kt`, `max_target_speed_kt`,
`stagnant_rate_nm_s` and `side_switch_after_s`. Changing the saved action/memory
contract on an initialization checkpoint is rejected; start fresh for that ablation.
With no explicit override, saved settings are restored across hosts.

`--reward-config rewards.json` accepts explicit reward coefficient overrides, e.g.:

```json
{"goal_bonus": 30.0, "restricted_event_cost": 6.0}
```

Unspecified coefficients retain the saved/default values. Unknown names, invalid
values or mismatched checkpoint spaces fail clearly. Reward changes are recorded
and require separate validation. They never change the official scoring penalties.
Training CSVs log the fixed objective metrics and separate `reward_*` contributions;
SA captures those diagnostics before SB3's automatic episode reset.

## Replay and evaluate

Interactive replay of a new shared-contract model:

```powershell
.\.venv312\Scripts\python.exe -m scripts.replay_transfer_navigation --env ma --model models/navigation_transfer_ma/model.zip --seed 42
```

Use `--headless --max-steps 2` for a bounded smoke check; capped flights are explicitly
unfinished and not final evaluation records. Full replay follows the original
termination/time budget. The GIF recorder also accepts `--mdp navigation-transfer`.

Separate development evaluations using the original worlds:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env sa --mdp navigation-transfer --model models/navigation_transfer_sa/model.zip --episodes 20 --out models/navigation_transfer_sa/model.evaluation.csv
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env ma --mdp navigation-transfer --model models/navigation_transfer_ma/model.zip --episodes 20 --out models/navigation_transfer_ma/model.evaluation.csv
```

The evaluator restores the saved feature/action/reward configurations. Use the
correct environment kind for the desired traffic setting, and preserve separate
output files. SA and MA rates are different tasks, not directly interchangeable
scores. No existing v5 result carries over to these new models.

Held-out comparisons within the same world are supported too:

```powershell
.\.venv312\Scripts\python.exe -m scripts.validate_navigation_checkpoints --env sa --mdp navigation-transfer --models models/navigation_transfer_sa/model.zip models/navigation_transfer_ma/model.zip --seeds 101 102 --episodes 3 --out-dir models/navigation_transfer_sa_validation
```

Use another output directory with `--env ma` to assess the same candidates in the
reactive-aircraft world. Choose training/validation seeds separately and keep
checkpoints unchanged during comparisons. The validation tool retains raw metrics,
model hashes, exact configurations and explicit nonofficial labels. The canonical
scoring evaluator still seeds 42 on its first reset only, followed by unseeded resets.
Official reporting still requires the prescribed first 1000 scenarios.

## Single-agent lookahead

The same trained SA checkpoint can be evaluated with the adapted PPO-guided search.
No retraining is required. Keep the search report separate from the actor-only report:

```powershell
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env sa --mdp navigation-transfer --model models/navigation_transfer_sa/model.zip --episodes 20 --with-lookahead --out models/navigation_transfer_sa/model.lookahead.evaluation.csv
.\.venv312\Scripts\python.exe -m scripts.evaluate_competition --env sa --mdp navigation-transfer --model models/navigation_transfer_sa/model.zip --episodes 20 --no-lookahead --out models/navigation_transfer_sa/model.actor_only.evaluation.csv
```

Both commands use the first 20 scenarios from seed 42 in the original SA world,
with ten scripted intruders and five restricted areas. Search defaults to the MA
budget of 24 simulations, depth six, seven policy-derived candidates and one pass.
It uses PPO action priors and critic leaf values; its separate forecast never steps
or commands the live simulator. The forecast is approximate and does not guarantee
safe flight. Model weights, original dynamics, metrics and termination stay unchanged.

The adapter copies progress anchors, stalled duration and obstacle-side commitment
into independent search branches. It samples predicted safety events every second,
and accumulates rewards and memory at the saved decision cadence within each held
command forecast. Scripted intruders are extrapolated from current horizontal
position, heading and speed; future route waypoints and autopilot targets are not
read. Their waypoint turns can therefore differ from the forecast. The approximate
turn/speed model and local planar geometry also differ from live A320 dynamics.

Add `--lookahead-preset fast` for 12 simulations/depth three, or explicit
`--lookahead-simulations 18 --lookahead-depth 4`. These change planning behavior and
need separate comparisons. Explicit budget overrides opt into search unless
`--no-lookahead` was specified, in which case conflicting options are rejected.
Resolved budgets, inference mode and environment configurations are saved in the
evaluation `.settings.json`. The optional MA separation filter and coordination
passes are not supported by this single-aircraft adapter.

The expanded suite passes 227 tests. New checks cover exact current policy inputs,
forecast observations, deterministic search without live simulator or RNG changes,
scripted-traffic forecasts without route access, branch-local obstacle memory,
saved speed/deadband settings, goal/timeout handling and discounted macro rewards.
A regression across the first 20 seed-42 scenarios checks that the copied progress
anchor uses the same planar distance metric as subsequent forecasts, preventing
projection offsets from creating fictitious progress or stalls. The live progress
history and exact current policy input are preserved.

Matched development evaluation of `models/navigation_transfer_sa/model.zip` on
20 original seed-42 scenarios (same weights and environment settings):

| Metric | Actor only | PPO-guided lookahead |
| --- | ---: | ---: |
| Arrivals | 16/20 (80%) | 15/20 (75%) |
| Clean arrivals | 0/20 (0%) | 9/20 (45%) |
| Clean flights, including timeouts | 0/20 | 13/20 |
| Intrusion events / seconds | 42 / 3,729 | 12 / 741 |
| Restricted-area events / seconds | 30 / 6,093 | 3 / 411 |
| Sector exits / outside seconds | 0 / 0 | 0 / 0 |
| Total flight seconds | 37,467 | 42,377 |
| Evaluation wall time | 84.8s | 866.3s |

This is a substantial safety improvement with slightly fewer arrivals, longer
flights and about ten times the evaluation compute. It is not a finished safe
single-agent solution or an official 1000-scenario result. Four of the five
lookahead timeouts were clean. No training or model-weight updates were performed.
The source checkpoint SHA-256 remained
`51c3935848d8237ef5000c4605201049d2a15549f60b335f700406436423cb6a`.

- [Full comparison](../../models/navigation_transfer_sa/model.lookahead.comparison.md)
- [Settings, metrics and provenance](../../models/navigation_transfer_sa/model.lookahead.comparison.json)
- [Implementation/evaluation verification](../../.cache/sa_lookahead_verification.json)

Replay can use the same search and budget flags:

```powershell
.\.venv312\Scripts\python.exe -m scripts.replay_transfer_navigation --env sa --model models/navigation_transfer_sa/model.zip --seed 42 --with-lookahead
```

The GIF recorder and held-out checkpoint validator also accept `--with-lookahead`
with `--env sa --mdp navigation-transfer`. All default commands retain actor-only
inference. PPO training still collects its own sampled actions; this change adds
inference search rather than silently substituting planned actions into PPO rollouts.

## Verification and limits

The unit/integration checks cover matching spaces, command meanings and feature
values; all ten scripted intruders; masks; per-aircraft memory reset; idempotent
progress updates; one-time side switching; unchanged inside-obstacle escape;
deadline arithmetic; reward-component capture; rejection of incompatible models;
and actual SA->MA->SA PPO weight transfer/save/load. The original base `step`,
`_update_metrics` and `_get_info` methods are inherited unchanged in each world.
Defaults retain five obstacles, ten scripted SA intruders or ten MA aircraft,
5 NM separation, 5 km capture, 3000 seconds, one-second metric sampling, A320
physics and built-in resolution off. No vertical actions or observations were added.

The legacy test previously assuming the independently evolved SA class was still
identical to the old MA mixin was already broken (missing imports). It now freezes
the legacy MA contract against that mixin; a new test checks cross-host parity of
this explicitly versioned MDP. No claim of old SA/MA weight compatibility remains.

Tiny training/replay/evaluation smoke artifacts live under `.cache`. They validate
the interfaces, not improved arrival or safety rates. Full new training, controlled
feature ablations and held-out evaluation are still needed. No production model was
retrained or overwritten as part of this implementation.

The [published competition rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
allow custom observations, shaped rewards and heading/speed mappings while fixing
the world and objective metrics. This implementation preserves those fixed layers;
these tests are not organizer certification or proof of improved performance.

## Completed implementation verification

The final suite passed **211 tests**. Bounded integration tests actually trained,
saved and loaded SA->MA->SA weights under the shared contract. Public replay,
evaluation and SA held-out-validation commands also completed with the tiny smoke
models. Evaluation settings correctly recorded one controlled SA aircraft and ten
controlled MA aircraft, with lookahead disabled. The original best v5 checkpoint
SHA-256 remained unchanged.

- [Verification manifest](../../.cache/transfer_verification.json)
- [SA evaluation smoke settings](../../.cache/transfer_evaluation_smoke/sa.settings.json)
- [MA evaluation smoke settings](../../.cache/transfer_evaluation_smoke/ma.settings.json)
- [Held-out validation smoke](../../.cache/transfer_validation_sa/summary.md)

These artifacts use minimally trained smoke models and one-scenario checks. They
are implementation evidence only, not replacement performance results or new
production models. Train and evaluate new shared-contract models before comparing
with the existing 90.5% lookahead-assisted or 37.5% actor-only v5 clean-arrival results.
