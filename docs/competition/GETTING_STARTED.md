# Getting Started

A practical walk-through of the competition code: install, run the demo
scripts, understand the example training script, and customize the MDP. For the
rules and scoring see **[COMPETITION.md](COMPETITION.md)**.

---

## 1. Setup

First, clone the competition branch and navigate to the repository:

```bash
git clone --branch AI4REAL-NET-Competition --single-branch https://github.com/TUDelft-CNS-ATM/bluesky-gym.git
cd bluesky-gym
```

Requires Python ≥ 3.10. The project is managed with [uv](https://docs.astral.sh/uv/)
(there is a `uv.lock`), but any virtualenv works.

```bash
# with uv (creates .venv from the lockfile)
uv sync

# then either prefix commands with `uv run`, or call the venv python directly:
.venv/bin/python test.py
```

Two ways to run code, depending on where the file lives:

- **repo-root scripts** (`test.py`, `test_zoo.py`, `train_zoo.py`): `python test.py`
- **`scripts/` modules** (`evaluate_competition`, `record_competition_gifs`):
  run as a module from the repo root, e.g. `python -m scripts.evaluate_competition` or `uv run python -m scripts.evaluate_competition`.

---

## 2. See it run (the demo scripts)

### Single-agent — [`test.py`](../../test.py)

A quick visual check of `CompetitionEnv-v0`. It opens a pygame window, flies a
neutral (`[0, 0]` = keep heading and speed) policy for a few episodes, and prints
the objective scoring `info` dict at the end of each.

```bash
python test.py
```

### Multi-agent — [`test_zoo.py`](../../test_zoo.py)

The same idea for `CompetitionZooEnv` with `n_agents=10` (the competition
config): all agents fly straight, and each agent's metrics print as it leaves
the episode.

```bash
python test_zoo.py
```

> Running headless (no display)? Prefix with `SDL_VIDEODRIVER=dummy` to skip the
> window, or use the `render_mode=None` / `"rgb_array"` modes in your own code.

---

## 3. The example training script — [`train_zoo.py`](../../train_zoo.py)

`train_zoo.py` trains one shared PPO policy across the multi-agent environment.
It defaults to the **navigation MDP**, which reuses the observation, reward and
heading/speed action mapping from `SingleAgentEnv`. See the
**[multi-agent pipeline guide](MULTI_AGENT_PIPELINE.md)** for the checked rules,
warm-start details and complete training/evaluation/reel workflow.

```bash
python train_zoo.py train 250000      # save multi_agent_model.zip and run metadata
python train_zoo.py replay            # render the shared navigation policy

# Start from a compatible existing single-agent checkpoint.
python train_zoo.py train 1000000 --init-model single_agent_model.zip

# Explicitly select the original competition MDP instead.
python train_zoo.py train 250000 --mdp base --model ppo_competition_base
```

**Algorithm.** Stable-Baselines3 `PPO` with a `MultiInputPolicy`, using
**parameter sharing**: one shared policy drives all 10 homogeneous agents.
The navigation MDP keeps the same Dict keys and feature shapes as the
single-agent policy, so compatible SA weights can initialize MA training.

**SB3 adapter.** `SharedPolicyVecEnv` in `bluesky_zoo/sb3_vec_env.py` exposes
the agents as vector slots backed by one PettingZoo `ParallelEnv`. It keeps
Dict observations, final metrics, terminal observations and time-limit flags.
Aircraft that arrive early use absorbing slots until the shared scenario ends;
the training CSV records each aircraft's real final metrics once.

**Why a single underlying env matters:** BlueSky is a **process-global
singleton** — only one BlueSky-backed environment can be active per process.
Run evaluation in a separate command after training. Avoid callbacks that
interleave another BlueSky environment in the training process.

**Default hyperparameters:** `n_steps=1024`, `batch_size=256`,
`gamma=0.99`, `gae_lambda=0.95`, `learning_rate=3e-4`, `ent_coef=0.0`,
separate policy/value MLPs `[256, 256]`, and initial log standard deviation `-0.5`.
CLI flags control rollout size, seed, checkpoint frequency and device.

After training, the script now saves a totals report and runs a separate
1000-scenario evaluation by default. Use `--eval-episodes 20` for a short
nonofficial check or `--eval-episodes 0` to skip it. Markdown/JSON reports contain
the sharing-template totals and goal/clean-flight rates; see the pipeline guide
for output filenames and metric definitions.

> Training timesteps count agent slots, including absorbing slots after arrival.
> `--n-agents` can lower traffic for training, but report/evaluate at 10.

---

## 4. Customize the MDP

You design your MDP by overriding three hooks — `_get_obs`, `_get_reward`,
`_get_action` — or by wrapping the env. Minimal reward-shaping example:

```python
import gymnasium as gym
import bluesky_gym
from bluesky_gym.envs.competition_env import CompetitionEnv

class MyEnv(CompetitionEnv):
    def _get_reward(self, ac_id):
        # your reward shaping (does NOT affect the scored metrics)
        reward = super()._get_reward(ac_id)
        reward += self._my_custom_reward_component(ac_id)
        return reward
```

Or via a standard wrapper (no subclass needed):

```python
bluesky_gym.register_envs()
env = gym.wrappers.TransformReward(gym.make("CompetitionEnv-v0"), lambda r: r)
```

If you change the **observation**, also rebuild the observation space in your
subclass `__init__` (or use an observation wrapper that declares the new space).
Both environment source files have detailed docstrings describing the contract:
[`competition_env.py`](../../bluesky_gym/envs/competition_env.py) and
[`competition.py`](../../bluesky_zoo/competition/competition.py).

See the [windfield wrapper](../../bluesky_gym/wrappers/wind.py) for an example wrapper
that also modifies the observation function. 

### Train on the single-agent track

There is no bundled single-agent trainer, because SB3 handles it directly. The
observation is a `Dict`, so either use `MultiInputPolicy`, or flatten it and use
`MlpPolicy`:

```python
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import PPO

bluesky_gym.register_envs()
env = gym.wrappers.FlattenObservation(gym.make("CompetitionEnv-v0"))  # Dict -> Box(131,)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_competition_sa")
```

---

## 5. Reference

### Observation (base class)

A `Dict` of these components (flattened: **131** dims single-agent, **124**
multi-agent with `n_agents=10`). All bearings are cos/sin pairs; distances are
normalized.

| Component | Fields | Size |
| --- | --- | --- |
| Waypoint | `cos_drift`, `sin_drift`, `waypoint_distance` | 3 |
| Own airspeed | `airspeed` | 1 |
| Intruders (per neighbour) | `x_r`, `y_r`, `vx_r`, `vy_r`, `cos_track`, `sin_track`, `intruder_distance` | 7 × N |
| Obstacles (×5) | `obstacle_distance`, `obstacle_cos_bearing`, `obstacle_sin_bearing`, `obstacle_radius` | 4 × 5 |
| Sector (×12 + flag) | `sector_point_distance`, `sector_point_cos_bearing`, `sector_point_sin_bearing`, `inside_sector` | 37 |

N = number of intruders shown (single-agent 10; multi-agent `n_agents − 1` = 9).

### Action

`Box(-1, 1, shape=(2,))`: `action[0]` scales a heading change (× `d_heading`,
default 45°), `action[1]` scales a speed change (× `d_speed`, default ≈6.67 kt),
both relative to the current state. `[0, 0]` keeps heading and speed.

Current action space is continuous, [discrete action spaces](../../core/actions.py) are available. 

Note that the action space is deliberately **2D**. Allowing altitude differences 
trivializes the separation management task for the traffic densities simulated
and is not within the scope of this competition

### Metrics

See the [metrics table in COMPETITION.md](COMPETITION.md#metrics-the-objective-score).

---

## 6. Evaluate & record

```bash
# official metrics over the first 1000 episodes of seed 42
python -m scripts.evaluate_competition --env sa                
python -m scripts.evaluate_competition --env ma                

# quick check + your trained policy (see load_policy in the script)
python -m scripts.evaluate_competition --env ma --mdp navigation --model multi_agent_model --episodes 20

# official report protocol for the trained navigation policy
python -m scripts.evaluate_competition --env ma --mdp navigation --model multi_agent_model

# record GIFs (docs images, or your 5-scenario submission reel)
python -m scripts.record_competition_gifs                        # both doc GIFs
python -m scripts.record_competition_gifs --env ma --mdp navigation --model multi_agent_model --seeds 1 2 3 4 5 --out reel.gif
```

Evaluation and GIF scripts default to the original `--mdp base` demo. Select
`--mdp navigation` to match training's default MDP, and pass the PPO checkpoint
with `--model`. Mismatched checkpoint spaces fail clearly. The recording budget
covers a complete episode unless `--max-steps` sets an explicit shorter cap.
