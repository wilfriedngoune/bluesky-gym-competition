# Scenario 9: what the search expected before the conflicts

Inspected on 9 November 2026, using the unchanged responsive v5 checkpoint,
seed 42, scenario 9, ten aircraft, 24 search simulations and maximum depth six.
Root decisions from seconds 40 through 400 were recorded for KL001, KL006 and
KL009, with actual replay continued to second 500. No controller or reward was
changed and no model was retrained.

## Verification

The diagnostic replay exactly reproduced all 1000 aircraft/time records of the
original replay through second 500 for position, heading, airspeed, policy
actions, accepted heading/speed targets and intrusion counters. The logger
therefore did not change the encounter it was inspecting. All 438 selected
root/pair forecast windows have actual samples through their forecast end.

The generated [plot](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.png)
shows actual separations, example predicted continuations, assumed/actual
neighbour command differences and explored forecast coverage.

## Finding 1: the other aircraft do not execute the commands search assumes

Search forecasts other aircraft using their deterministic actor commands at
the root. In the real simulation each aircraft instead chooses its own search
action. Across the logged directed root/pair windows, the assumed and actual
neighbour target headings differ by an average 55.5 degrees; target speed
differs by an average 12.1 knots. Projection differences are included in these
comparisons, but do not explain the approximately 90/180-degree differences.

For KL001/KL006:

| Root time | Current separation | Forecast coverage | Predicted minimum | Actual sampled minimum | Neighbour heading difference at root |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 240 s | 5.76 NM | 45 s | 5.13 NM | 4.05 NM | 90.2 deg |
| 245 s | 5.50 NM | 45 s | 5.34 NM | 3.84 NM | 179.8 deg |
| 250 s | 5.29 NM | 60 s | 5.14 NM | 2.96 NM | 90.2 deg |

The average error at the next real decision is about 0.029 NM; error grows
over longer horizons. These longer comparisons include changes to the own
aircraft's subsequent plan, not just opponent prediction or dynamics error.
The data establish a command-assumption mismatch, but do not isolate its full
causal contribution to the conflict.

## Finding 2: some dangerous choices are predicted to be dangerous

At second 255, KL001 is about 5.14 NM from KL006. Its selected root candidate
predicts a minimum of approximately 4.77 NM across the logged pairs in its
first 15 seconds. That candidate receives 11 visits and average backed-up
return -14.70. Its first transition reward is -6.59, including the modeled
safety costs, and its first learned leaf value is about -9.28.

Another candidate predicts at least 5.12 NM across the same logged pairs over
the same first 15 seconds, with first transition reward -0.63. It receives only
three visits and average return -16.13. The search therefore prefers the
predicted-dangerous option according to its estimated longer-term return.
This does not prove the alternative is globally safe: only the selected
neighbours are logged, and later obstacles/traffic can affect return.

For the later KL001/KL009 encounter, the selected continuation at second 350
already predicts separation below 5 NM before the actual intrusion around
second 378. Detection is present; value/reward trade-offs and exploration also
need attention. The diagnostic observer does not alter these choices; the
official one-second safety metrics continue recording the resulting failures.

## Finding 3: configured depth is not typical explored coverage

There are 219 logged root decisions. Their retained continuations cover an
average 38.1 seconds; 203 of 219 cover 30 or 45 seconds. Only one retained path
reaches the configured 90-second maximum. Increasing depth alone would not
necessarily change this under the existing simulation budget, and exploring
farther with inconsistent other-aircraft assumptions can amplify error.

## Next controlled experiment

First investigate making predicted neighbour commands consistent with the
simultaneous joint action selection, keeping the learned policy and value
central. Compare the same encounter and unchanged checkpoint against the
current planner before retraining. Treat changing the safety/value trade-off
or search budget as separate experiments. The diagnostics identify multiple
contributors, not a single proven fix.

## Evidence and reproduction

- [Analysis report](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.analysis.md)
- [Selected-window comparisons](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.analysis.csv)
- [All candidate scores](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.candidates.csv)
- [Retained forecasts](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.forecasts.csv)
- [Actual encounters](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.actual.csv)
- [Actual commands](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.commands.csv)
- [Checkpoint checksum and trace settings](../../models/ma_navigation_v5_responsive/scenario9_diagnostics_40_400.settings.json)
- [Reproduction commands and field definitions](SEARCH_DIAGNOSTICS.md)
