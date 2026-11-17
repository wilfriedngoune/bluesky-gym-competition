# Navigation v5: learned lookahead and shrinking waypoint zones

`--mdp navigation-v5` adds a soft shrinking zone for each aircraft and uses
policy/value-guided Monte Carlo tree search during inference. It builds on v4's
polygon-clearance observations and full heading range. Existing navigation
modes and checkpoints retain their original behavior.

## Current experiment: responsive decisions with fewer small corrections

The initial matched 20-scenario results favored the saved 5-second model over
fresh 15-second training: clean arrivals were 90.5% versus 75%, with 0 versus 9
sector exits. These separately trained policies do not isolate decision timing
as the sole cause. Keep both checkpoints as references.

The next controlled experiment uses **5-second decisions** and **1-degree /
1-knot command deadbands**. This allows timely corrections while suppressing
small changes to accepted targets. It retains the 8 NM aircraft warning bubble,
shrinking waypoint zones, and learned policy/value lookahead. No new penalty or
automatic recovery controller is added. Improved clean arrivals remain to be
measured.

The [matched evaluation and boundary audit](V5_RESPONSE_AUDIT.md) documents
the completed actor-only comparison and the reproduced scenario-10 exit.

From the repository root in PowerShell:

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_responsive_v5
```

This continues the saved 5-second baseline with ten aircraft, using seed 4 and
one million requested timesteps, then evaluates 20 scenarios. Outputs go to
`models/ma_navigation_v5_responsive/`. Add `--fresh` for fresh weights with the
2/4/6/10-aircraft curriculum, or `--dry-run` to inspect the command first. The
launcher refuses to overwrite existing model/metrics files; use `--model` with
a new directory for another run. General training defaults and the settings of
existing checkpoints remain unchanged.

The equivalent continuation command is:

```powershell
python train_zoo.py train 1000000 --mdp navigation-v5 --init-model models/ma_navigation_v5_separation/model.zip --decision-seconds 5 --heading-deadband-deg 1 --speed-deadband-kt 1 --aircraft-buffer-nm 8 --aircraft-buffer-cost 0.05 --seed 4 --model models/ma_navigation_v5_responsive/model --metrics models/ma_navigation_v5_responsive/training_metrics.csv --checkpoint-dir models/ma_navigation_v5_responsive/checkpoints --eval-episodes 20
```

## Targeted diagnostic replay

For candidate scores, predicted/actual separation and neighbour-command
comparisons, see [Search diagnostics](SEARCH_DIAGNOSTICS.md) and the completed
[scenario-9 findings](SCENARIO9_SEARCH_FINDINGS.md).

Replay can select a 1-based scenario from the original seeded reset stream
without simulating all preceding flights. The reset stream is advanced with
the first reset seeded once; these options do not change official evaluation.
An explicit replay cap prints that unfinished flights have no final evaluation
record. The trajectory CSV includes safety occupancy durations and accepted
heading/speed target changes.

For the early, long sector excursion in scenario 10 of the 15-second model:

```powershell
python train_zoo.py replay models/ma_navigation_v5_fresh_hold15/model.zip --mdp navigation-v5 --episodes 1 --seed 42 --scenario-start 10 --max-steps 40 --headless --trajectory-csv models/ma_navigation_v5_fresh_hold15/scenario10_search.csv
```

Add `--no-lookahead` and choose a different CSV path to inspect the same
scenario with the actor alone. This is a diagnostic subset, not a replacement
for the evaluation harness.

## Optional longer-decision experiment

Use `--decision-seconds 15` during training to hold each aircraft's heading and
speed targets for 15 simulated seconds. This reduces the maximum number of
heading/speed decisions in a 3000-second flight from **600 to 200**. Aircraft
continue turning/accelerating toward the targets; the flight dynamics are not
frozen. All aircraft use the same configured decision interval.

The simulator and official metrics still run every **1 second**, and an aircraft
still terminates on the exact sampled second it reaches its goal. There is no
automatic safety override or extra controller between decisions. Longer holds
can make paths more predictable but also delay useful corrections, so compare
5, 10 and 15 seconds using clean arrivals and intrusion metrics. No improvement
is assumed merely from reducing the action rate.

The interval is saved inside each model and restored during continuation,
evaluation, replay and GIF recording. Existing checkpoints retain the original
5-second navigation interval (10 seconds for the base MDP). Inputs keep the same
shape, so compatible v5 weights can initialize the experiment. Train further
with the new interval to adapt the actor and critic to its different dynamics.

When `--gamma` is omitted, training rescales the discount as
`old_gamma ** (new_interval / old_interval)` to retain its discount over
simulated time; an explicit `--gamma` is used per decision as supplied.
Lookahead segments are aligned to complete decisions and cannot choose another
manoeuvre earlier than the real next decision. The simulator's inherited loop
requires intervals to divide 3000 seconds; supported values are integer
divisors from 1 through 120 seconds, including 5, 10, 15, 20 and 30.

For a continuation experiment with fewer decisions and the aircraft buffer:

```powershell
python train_zoo.py train 1000000 --mdp navigation-v5 --init-model models/ma_navigation_v5_separation/model.zip --decision-seconds 15 --aircraft-buffer-nm 8 --aircraft-buffer-cost 0.05 --model models/ma_navigation_v5_hold15/model --metrics models/ma_navigation_v5_hold15/training_metrics.csv --checkpoint-dir models/ma_navigation_v5_hold15/checkpoints --eval-episodes 20
```

This continues directly with ten aircraft. Omitting `--init-model` starts fresh;
add `--curriculum 2 4 6 10` for a fresh traffic curriculum. Optional
`--heading-deadband-deg 1 --speed-deadband-kt 1` additionally suppresses small
changes between decisions; it does not bypass the decision interval.

## Shrinking zone

Each aircraft's virtual circle is centred on its own waypoint. Its schedule is
fixed at episode reset; flying in a loop or leaving the circle cannot restart
the clock or enlarge the circle. Defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| Grace period | 120 s | Wait before shrinking |
| Detour allowance | 10 NM | Added to the initial waypoint distance |
| Nominal closing speed | 150 kt | Used only to set a distance-dependent deadline |
| Arrival-time factor | 1.5 | Allows extra time beyond nominal direct travel |
| Minimum shrinking duration | 300 s | Avoids an immediate deadline for short flights |
| Maximum outside cost | 0.01/s | Bounded training penalty |
| Excess-distance scale | 10 NM | Distance outside at which the cost saturates |

The shrinking deadline is the grace period plus the larger of 300 seconds and
1.5 times the nominal remaining travel time, capped by the original episode
budget. The radius shrinks linearly to the existing **5 km goal capture radius**
and stays there. The world time limit is never extended.

The `zone` observation contains normalized radius, signed distance to the circle
boundary and remaining time until its deadline. The additional reward is:

```text
-outside_second_cost * actual_elapsed_seconds
    * clip(distance_outside / excess_scale_nm, 0, 1)^2
```

There is no crossing bonus or additional termination. The cost uses geometry at
the end of each policy decision and the aircraft's actual flown seconds; it is a
training approximation, not exact one-second circle occupancy. Official safety
occupancy remains measured every second. The training CSV records `reward_zone`
alongside the other reward components.

## Aircraft separation bubbles

Fresh v5 training also uses an **8 NM warning radius around each aircraft**.
Distances are measured between aircraft centres: entering another aircraft's
8 NM circle starts a proximity cost; overlapping two 8 NM drawings at 16 NM
separation does not. Both aircraft receive the cost, including when they are
moving apart. Each nearby neighbour contributes separately, so avoiding one
aircraft does not hide another conflict.

The cost rises smoothly from zero at 8 NM to at most **0.05 per second per
neighbour** at 5 NM, remaining saturated inside 5 NM. Existing actual-intrusion
event and duration penalties remain in addition. The new reward component is
`reward_aircraft_buffer` in training CSVs:

```text
-aircraft_buffer_second_cost * actual_elapsed_seconds
    * sum(clip((aircraft_buffer_nm - neighbour_distance_nm)
               / (aircraft_buffer_nm - 5 NM), 0, 1)^2)
```

The real reward uses current distances at the end of each decision, multiplied
by actual flown seconds. It does not count exact one-second occupancy of the
warning bubble. Official intrusions are still detected every second at the
unchanged **5 NM** threshold. Lookahead includes the same proximity cost in its
one-second forecast samples. The bubble adds no physical constraint, automatic
controller or termination, and cannot guarantee that a learned policy avoids
conflicts.

Use `--aircraft-buffer-nm 8 --aircraft-buffer-cost 0.05` to set the radius and
penalty explicitly; the radius must exceed 5 NM. Use a zero cost for an ablation.
Existing v5 observations already include all other aircraft distances, so this
reward change does not change the checkpoint input shape. V2/v3/v4 retain a zero
default cost; their trainers also accept the explicit options.

Existing saved reward profiles remain unchanged during evaluation and
continuation, including old checkpoints without bubble fields (disabled).
To enable the new cost when continuing an existing v5 model, explicitly pass
both bubble options and save to a new experiment directory:

```powershell
python train_zoo.py train 1000000 --mdp navigation-v5 --init-model models/ma_navigation_v5/model.zip --aircraft-buffer-nm 8 --aircraft-buffer-cost 0.05 --model models/ma_navigation_v5_separation/model --metrics models/ma_navigation_v5_separation/training_metrics.csv --checkpoint-dir models/ma_navigation_v5_separation/checkpoints --eval-episodes 20
```

This transfers learned weights with a fresh optimizer. Retraining is needed to
teach the actor and critic the new reward; reward settings alone do not update
the behavior of a saved model. Model improvement requires comparison of both
conflict metrics and clean waypoint arrivals on matched scenarios.

## Policy/value-guided lookahead

For profiling results, implementation optimizations and the unchanged scoring
constraints, see [Evaluation performance](EVALUATION_PERFORMANCE.md). Existing
checkpoints use these optimizations when evaluated from a new Python process.

PPO training collects the learned actor's own actions. Search does **not** replace
actions in PPO's rollout buffer, so the actions and their recorded probabilities
remain consistent. The shrinking-zone reward trains both the policy and critic.

During evaluation, replay and recording, MCTS uses the learned actor to propose
heading/speed candidates and assign their prior probabilities. The learned
critic evaluates predicted leaf states. Search compares continuations and
executes only the first selected action before observing the real simulator
again. There is no search-only fallback without an RL checkpoint.

Defaults are 24 simulations per aircraft decision, up to 6 tree levels, 7
candidate actions and 15 simulated seconds per forecast segment. This permits a
maximum 90-second forecast horizon; a finite search budget need not explore
every branch to that depth. Discounting accounts for forecast duration and the
PPO decision interval.

Each other aircraft follows its deterministic learned actor command from the
shared root snapshot, held throughout the forecast. This is not reactive traffic
prediction or a joint search over all aircraft's decisions.

Forecasts run in independent horizontal state copies. The approximate predictor
uses limited turns and speed changes with one-second integration. It does not
step, rewind, reset or replace BlueSky. Its default limits are 3 degrees/s and
1 kt/s; these are forecasting assumptions, not changes to A320 dynamics.
Model error, limited search depth and other aircraft's future decisions can all
affect the result. Neither search nor the shrinking circle guarantees safety
or successful arrival.

## Competition rules

Checked against the [published competition rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md)
on 4 November 2026. The allowed design surface includes observations, rewards,
heading/speed action mapping, decision frequency and RL algorithms. V5 uses
these extension points and inherits the competition's authoritative world and
scoring:

| Fixed requirement | V5 behavior |
| --- | --- |
| Horizontal control only | Heading and speed; no altitude actions or search inputs |
| A320 dynamics | Original BlueSky simulator |
| Built-in conflict resolution | `RESO OFF` |
| Separation minimum | 5 NM |
| Waypoint capture radius | 5 km |
| Episode budget | 3000 simulated seconds |
| Metric sampling | Every 1 simulated second |
| Scoring and lifecycle | Inherited `_update_metrics`, `_get_info` and `step` |
| Evaluation scenarios | Original scenario generator; 10 aircraft and 5 obstacles |
| Reporting protocol | First 1000 scenarios of seed 42; seed the first reset only |

The RL requirement is addressed by learning the policy and value estimates with
PPO and making both central to the inference search. A pure handcrafted search
controller is prohibited. The rules do not explicitly name or pre-approve this
particular hybrid, so this is an implementation-based reading of the published
rules, not a claim of organizer approval.

Custom passage scenarios and lighter traffic remain training-only options.
Evaluation always restores the original scenario distribution. Checkpoints and
their adjacent JSON files save the reward, command, passage, zone and lookahead
profiles so replay and evaluation reconstruct the same behavior.
Evaluation also writes a `.settings.json` file beside its CSV, recording whether
lookahead was enabled and the reward, command, passage, zone and search profiles.

## Train, compare and record

V5 adds observation fields, so start fresh or initialize from a matching v5
checkpoint. Older v3/v4 policies cannot be loaded directly. The current lookahead
adapter requires the standard decentralized PPO critic; `--centralized-critic`
is rejected for v5.

```powershell
python train_zoo.py train 100000 --mdp navigation-v5 --curriculum 2 4 6 10 --model models/ma_navigation_v5/model --metrics models/ma_navigation_v5/training_metrics.csv --checkpoint-dir models/ma_navigation_v5/checkpoints --eval-episodes 20
```

The budget is illustrative, not a claim that this amount of training is enough.
Use `--zone-outside-second-cost 0` for a zone-penalty ablation, or adjust
`--zone-grace-seconds`, `--zone-detour-allowance-nm` and the other `--zone-*`
training options. `--lookahead-simulations` and `--lookahead-depth` set the saved
inference search budget. A continuation restores saved profiles and applies
only explicit overrides.

```powershell
python train_zoo.py replay models/ma_navigation_v5/model --mdp navigation-v5 --episodes 3 --seed 42
python train_zoo.py replay models/ma_navigation_v5/model --mdp navigation-v5 --episodes 3 --seed 42 --no-lookahead
python -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5/model --episodes 20 --out models/ma_navigation_v5/quick_search.csv
python -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5/model --episodes 20 --no-lookahead --out models/ma_navigation_v5/quick_actor.csv
```

Compare completion, clean arrivals, each safety metric, flight time and inference
cost on identical scenarios. Short checks are labelled nonofficial; training
reward and smoke tests do not establish competition performance.

For the prescribed final evaluation and five-scenario video:

```powershell
python -m scripts.evaluate_competition --env ma --mdp navigation-v5 --model models/ma_navigation_v5/model --out models/ma_navigation_v5/official_evaluation.csv
python -m scripts.record_competition_gifs --env ma --mdp navigation-v5 --model models/ma_navigation_v5/model --seeds 1 2 3 4 5 --out models/ma_navigation_v5/reel.gif
```
