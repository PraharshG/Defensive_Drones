# Publication Talking Points

Dataset: `outputs/per_run_results.csv`.
Sample: 1000 scenarios and 9000 strategy runs.

## Main Claims

- **Best overall strategy:** `optimized_global` achieved 25.0% mission success.
- **Best attrition strategy:** `optimized_global` achieved a mean kill ratio of 0.588.
- **Removing corridors helped optimization:** `optimized_global` improved success from 13.1% to 25.0% (+11.9 percentage points).
- **Optimized global intercepted more attackers:** mean kills rose from 5.67 to 6.38 per scenario.
- **Collaborative corridors are the middle ground:** `optimized_collab` achieved 21.8% success and 6.06 mean kills while preserving corridor-first behavior.
- **Scenario-level improvement:** `optimized_global` killed more attackers than corridor `optimized` in 48.9% of paired scenarios, tied in 31.1%, and killed fewer in 20.0%.
- **Success conversion:** `optimized_global` turned 140 scenarios from corridor-optimized failures into successes.
- **Collaboration conversion:** `optimized_collab` turned 88 corridor-optimized failures into successes and improved kill count in 20.4% of paired scenarios.
- **Deadline-first policies are not enough:** earliest-deadline variants are useful as urgency baselines, but they trail the optimized global policy in both success and mean kill ratio.
- **The result is still capacity constrained:** even the best strategy leaves substantial breach risk, so future gains likely require softer handoff rules, more defenders, faster defenders, or a shorter dwell requirement.

## Figure Guide

- Figure 1: `01_success_rate_by_strategy.png` - Shows which strategy most often kills all attackers before breach.
- Figure 2: `02_mean_kill_ratio_by_strategy.png` - Normalizes kills by scenario size so mixed attacker counts can be compared.
- Figure 3: `03_mean_kills_by_strategy.png` - Reports the absolute number of intercepted attackers before breach.
- Figure 4: `04_success_rate_by_defender_count.png` - Shows how added defensive assets change mission outcomes.
- Figure 5: `05_success_rate_by_attacker_load.png` - Compares strategy robustness as attacker counts grow.
- Figure 6: `06_success_rate_heatmap.png` - Identifies operating regimes where each strategy is viable.
- Figure 7: `07_kill_ratio_distribution.png` - Shows consistency and tail behavior beyond mean performance.
- Figure 8: `08_terminal_time_distribution.png` - Shows whether strategies delay breach or finish quickly.
- Figure 9: `09_global_vs_corridor_success_uplift.png` - Compares hard corridors, collaborative corridors, and global assignment.
- Figure 10: `10_optimized_global_kill_ratio_delta.png` - Shows how often global optimization improves or hurts individual scenarios.

## Overall Metrics

| Strategy | Success rate | Mean kills | Mean kill ratio | Median terminal time |
| --- | ---: | ---: | ---: | ---: |
| Optimized Global | 25.0% | 6.38 | 0.588 | 45.88s |
| Optimized Collab | 21.8% | 6.06 | 0.560 | 42.75s |
| Optimized Corridor | 13.1% | 5.67 | 0.521 | 42.50s |
| Nearest Global | 12.1% | 5.56 | 0.510 | 41.00s |
| Nearest Collab | 19.2% | 5.94 | 0.549 | 42.00s |
| Nearest Corridor | 12.6% | 5.60 | 0.514 | 41.88s |
| Deadline Global | 14.9% | 5.66 | 0.529 | 50.25s |
| Deadline Collab | 21.1% | 6.29 | 0.580 | 49.62s |
| Deadline Corridor | 13.6% | 5.92 | 0.543 | 49.75s |

## Best Operating Regimes

- Optimized Global with 5 defenders vs 5-10 attackers: 92.0% success across 87 runs.
- Deadline Collab with 5 defenders vs 5-10 attackers: 81.6% success across 87 runs.
- Optimized Collab with 5 defenders vs 5-10 attackers: 75.9% success across 87 runs.
- Optimized Global with 4 defenders vs 5-10 attackers: 70.4% success across 108 runs.
- Nearest Collab with 5 defenders vs 5-10 attackers: 70.1% success across 87 runs.
- Deadline Global with 5 defenders vs 5-10 attackers: 67.8% success across 87 runs.
- Deadline Collab with 4 defenders vs 5-10 attackers: 65.7% success across 108 runs.
- Deadline Corridor with 5 defenders vs 5-10 attackers: 62.1% success across 87 runs.
