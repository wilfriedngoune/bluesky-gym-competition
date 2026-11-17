# V5 response experiment and boundary audit

Checked on 5 November 2026. The next experiment uses five-second decisions
with one-degree heading and one-knot speed command deadbands. It retains
policy/value-guided search. Its benefit has not yet been established by a full
training run; the launcher and a short continuation have been checked.

## Matched evaluation

Each column uses the first 20 scenarios of seed 42, ten aircraft and the
original evaluation distribution. These are preliminary checks, not the
official 1000-scenario protocol. The actor-only column was generated from the
same saved 15-second weights and settings as the search column, with only
lookahead disabled. The five-second model was trained separately, so it does
not isolate decision frequency as the only cause of differences.

| Metric | 5-second baseline + search | 15-second model + search | Same 15-second model, actor only |
| --- | ---: | ---: | ---: |
| Goal arrivals | 192 / 200 | 169 / 200 | 187 / 200 |
| Clean arrivals | 181 / 200 (90.5%) | 150 / 200 (75%) | 71 / 200 (35.5%) |
| Intrusion events | 4 | 2 | 206 |
| Intrusion time [s] | 392 | 190 | 28,392 |
| Restricted-area events | 10 | 30 | 22 |
| Restricted-area time [s] | 2,076 | 7,870 | 1,995 |
| Sector-exit events | 0 | 9 | 7 |
| Time outside sector [s] | 0 | 13,695 | 1,942 |
| Flight time [s] | 353,060 | 424,698 | 356,228 |

Intrusions are per-aircraft counts, so both participants can contribute to one
encounter. Search substantially improved separation and clean arrivals for the
15-second checkpoint, while worsening its restricted-area occupancy, boundary
occupancy and completion. Removing search is therefore not the chosen change.

Sources:

- [Five-second baseline](../../models/ma_navigation_v5_separation/model.evaluation.md)
- [15-second model with search](../../models/ma_navigation_v5_fresh_hold15/model.evaluation.md)
- [New actor-only evaluation](../../models/ma_navigation_v5_fresh_hold15/model.actor_only.evaluation.md)
- [Actor-only inference settings](../../models/ma_navigation_v5_fresh_hold15/model.actor_only.evaluation.settings.json)

## Targeted boundary reproduction

Replayed scenario 10 of the seed-42 reset stream for its first 40 decisions
(600 simulated seconds), using the saved 15-second checkpoint and unchanged
search settings. The reset RNG was seeded once and advanced through nine
resets without flying the preceding scenarios. This is a diagnostic subset,
not another complete evaluation.

The trajectory reproduces the early exit associated with the longest saved
boundary failure:

- Aircraft `KL003` first exits at simulation second 231, as inferred from its
  cumulative official outside time at second 600: 369 seconds.
- Its first outside observation occurs at the decision at second 240.
- It remains outside through second 600. The sample at second 585 is about
  18.42 NM from the sector boundary.
- All 24 observed outside decisions use navigation mode 2 (sector recovery).
  On 16 of them, the selected target heading differs by more than 90 degrees
  from the inward recovery reference.
- In the matched actor-only replay, no aircraft exits during those first
  600 seconds.

This establishes that search can select outward commands despite receiving a
sector-recovery reference. It does not establish a unique cause or prove that
command deadbands will repair it: the deadbands suppress small changes, not
large outward turns. Limited search horizon, the learned value estimates and
the competing reward terms need a separate investigation if the response
experiment still exhibits boundary failures. No forced recovery controller
or extra penalty was introduced on the basis of this one trace.

Diagnostic files:

- [Search trace](../../.cache/v5_response_audit/scenario10_search.csv)
- [Actor-only trace](../../.cache/v5_response_audit/scenario10_actor.csv)

## Run the next experiment

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_responsive_v5
```

The launcher continues the five-second baseline at ten-aircraft density, uses
five-second decisions and one-degree/one-knot deadbands, and saves to the new
`models/ma_navigation_v5_responsive/` directory. Saved source checkpoints are
not rewritten. Use `--dry-run` to inspect the command, or `--fresh` to train
fresh weights with a traffic curriculum. Existing output files are protected
from overwriting. See the [v5 guide](LOOKAHEAD_AND_ZONE.md) for full commands.
