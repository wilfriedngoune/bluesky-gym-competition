# Evaluation performance: unchanged search and scoring

Profiled and checked on 9 November 2026 using the saved responsive v5 model.
Tree search is much more expensive than the earlier actor-only inference:
each live aircraft searches up to 24 simulations with multi-second forecasts,
learned policy/value evaluations and traffic/polygon checks. Returning from
15-second to 5-second decisions also triples the maximum number of planning
calls in a full-length flight. Neither effect is a change to scoring speed.

The first ten-aircraft planning call made about 7.47 million function calls
under the profiler. Avoidable work included calculating each CPA three times
per predicted second, rebuilding identical opponent forecasts across branches,
and projecting points onto every polygon boundary just to test occupancy.

## Changes

- Reuse each computed CPA for intrusion, predictive warning and proximity cost.
- Cache identical immutable predicted paths within a single planning decision.
- Cache copied-world polygon queries, with explicit memory bounds and resets
  between live snapshots and scenarios.
- Test polygon occupancy without nearest-boundary projection. Point/edge/vertex
  inclusion matches the former `covers(Point)` predicate; nearest points still
  use the original calculation wherever distances or bearings are required.
- Skip clearance calculations whose reward coefficient is zero.
- Show elapsed time and an estimated remaining time during evaluation; save
  evaluation wall time separately in the settings JSON.

These changes affect the independent predictor's implementation, not the
search configuration or objective. Existing checkpoints need no retraining.
A running Python process retains its imported code; a newly started evaluation
or replay uses the optimized implementation.

## Measurements and limits

Paired runs from the same four live snapshots of seed 42, responsive model,
ten aircraft and unchanged 24-simulation/depth-6 search settings:

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Mean planning time per ten-aircraft decision | 2.994 s | 1.408 s |
| Relative planning speed | 1x | 2.13x |

This is about 53% less planning wall time in that local microbenchmark, not a
measurement of an entire 20- or 1000-scenario evaluation. Hardware load, scene
geometry, active aircraft count and simulation cost affect end-to-end speed.

Selected actions and search visit counts were exactly equal on all four
snapshots. Three additional synthetic conflict/boundary cases also reproduced
actions, priors, root action values, visit counts, expansion counts and depth
exactly, including an enabled sector-buffer reward. These cases are regression
checks, not competition performance evaluations. The full suite passed 137
tests, including one-second scoring, read-only planning, geometry boundaries,
cache invalidation, early arrivals and the official episode lifecycle.

Local evidence:

- [Paired timings](../../.cache/evaluation_performance/paired_benchmark.json)
- [Conflict/boundary parity](../../.cache/evaluation_performance/stress_parity.json)
- [Paired benchmark](../../.cache/evaluation_performance/paired_benchmark.py)
- [Original predictor copy](../../.cache/evaluation_performance/lookahead_policy_reference.py)

## Competition constraints

Checked against the [published rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md).
No scoring hooks or authoritative BlueSky dynamics were changed. Metric
sampling stays at one simulated second; the episode budget remains 3000 s,
separation 5 NM, goal capture 5 km and built-in resolution off. Evaluation
retains ten aircraft, five obstacles, the original scenario distribution and
the seed-42 reset stream. The official report still requires 1000 scenarios.
The learned policy and critic remain central to search.

No speedup is obtained by dropping aircraft, reducing search simulations,
skipping safety samples or silently switching to actor-only inference.

The vectorized point predicate follows [Shapely's documented intersection
semantics](https://shapely.readthedocs.io/en/stable/reference/shapely.intersects_xy.html),
including polygon boundaries, without constructing a Point for every query.
