# Defensive_Drones: Research-Style Repository Overview

## Abstract

This repository implements a 3-D Monte Carlo simulator for evaluating target-allocation policies for defensive drones under large swarm attack scenarios. It is an abstract research simulator, not a flight-control, weapons-guidance, or operational deployment system. Attackers are modeled as point masses that spawn around a protected target, maneuver toward sampled target points inside a breach radius, and are resolved either by being killed after sustained close contact with a defender or by crossing the target breach sphere. Defenders are also modeled as point masses with bounded speed and acceleration. Each defender selects targets independently from noisy observations, so the simulator intentionally exposes target contention and duplicate assignments.

The core scientific question is: given many attackers and fewer defenders, how do local corridor-restricted, global, and collaborative assignment rules affect mission success, attrition, breach rates, target contention, and scaling with defender resources?

The result set used in this document is `outputs_smoke_v4`. It contains 25 Monte Carlo scenario seeds, 10 initial defender counts, 9 strategies, and therefore 2,250 strategy runs. This smoke-v4 setting is a high-load stress test: attacker counts range from 534 to 1,005 per scenario, while initial defender counts range from 10 to 100 and later waves add reinforcements equal to 50% of the initial defender count. Under this result set, every full mission failed under the strict zero-breach success criterion. The most informative results are therefore attrition and pass-through metrics rather than binary success rate. The best average attrition policy in `outputs_smoke_v4` is `optimized_collab`, with mean kill ratio 0.7655 and mean breach rate 0.2345. The strongest high-resource endpoint is `earliest_deadline_collab` at 100 initial defenders, with mean kill ratio 0.962 and mean breach rate 0.038, but still no full zero-breach successes.

## Keywords

Monte Carlo simulation, defensive swarms, target allocation, autonomous agents, point-mass dynamics, receding-horizon assignment, dwell-time kill condition, breach-rate analysis, target contention, corridor allocation.

## 1. Problem Statement

The repository studies a stylized defense problem. A protected target is fixed at the origin. Multiple waves of attacking drones spawn several miles away and travel toward the target. Defensive drones begin near the target and attempt to intercept attackers before they reach a small target breach radius.

The problem is hard in this simulator for several reasons:

- The attacker load is much larger than the defender count.
- Attackers arrive in waves rather than all at once.
- Attackers approach from the full 3-D sphere around the target in the default Python simulator.
- Attackers do not fly perfectly straight to the center; each aims at a sampled point inside the target breach sphere and adds bounded lateral maneuver velocity.
- Defender observations are noisy.
- A kill requires continuous proximity for a fixed dwell time, not merely a momentary intercept.
- Defenders do not communicate or maintain a shared assignment table.
- Independent defender decisions can converge on the same attacker, wasting defensive capacity.

The goal is not to prove that a real defensive swarm would work. The goal is to compare assignment policies inside a controlled abstract model and determine which mechanisms improve or harm performance.

## 2. Repository Map

| Path | Role |
| --- | --- |
| `README.md` | User-facing overview of the simulator, scenario design, strategy labels, and output files. |
| `defensive_drones/model.py` | Core dataclasses and configuration: attackers, defenders, scenarios, observations, run results, wave results, and default physical constants. |
| `defensive_drones/geometry.py` | Geometric helper functions: norms, unit vectors, corridor indexing, time-to-target estimates, constant-speed intercept helper, and segment-sphere breach detection. |
| `defensive_drones/scenario.py` | Random scenario generation: attacker waves, attacker spawn positions, target points, defender rings, reinforcements, and density-based corridor allocation. |
| `defensive_drones/strategies.py` | Assignment logic for nine strategy variants. Each active defender independently chooses a target from noisy observations. |
| `defensive_drones/engine.py` | Time-stepped simulation loop: activation, corridor rebalancing, sensing, assignment, defender motion, attacker motion, dwell-time kills, breach marking, and run summaries. |
| `defensive_drones/reporting.py` | Writes CSV outputs and baseline plots from completed `RunResult` objects. |
| `defensive_drones/simulate.py` | Command-line Monte Carlo runner with optional process parallelism and status logging. |
| `scripts/generate_publication_figures.py` | Generates publication-style PNG/PDF figures and markdown figure summaries from `per_run_results.csv`. |
| `scripts/run_server_simulation.sh` | Convenience script for creating a virtual environment, installing requirements, running the simulation, and generating publication figures. |
| `tests/test_simulation.py` | Unit and smoke tests for scenario generation, dynamics, assignment behavior, output generation, and figure generation. |
| `viz.html` | Browser-based D3 demonstration of the strategy families on smaller single-wave scenarios. It is useful for intuition, but it is not the same experiment as `outputs_smoke_v4`. |
| `outputs_smoke_v4/` | Requested result set for this document. Contains CSV tables, plots, publication figures, and a run status log. |

## 3. Scientific Goal

The simulator evaluates how target-allocation rules behave when many independent defensive agents respond to a large, noisy, dynamic attack field.

The main experimental variables are:

- Initial defender count: 10, 20, ..., 100.
- Attacker count: randomly generated by wave, producing 534 to 1,005 total attackers in `outputs_smoke_v4`.
- Wave count: 2 or 3 waves in `outputs_smoke_v4`.
- Strategy family: optimized, nearest, or earliest-deadline.
- Assignment mode: corridor, global, or collaboration.

The main response variables are:

- Mission success: whether zero attackers breach.
- Kills: attackers killed by sustained defender proximity.
- Breaches: attackers crossing the target breach sphere.
- Kill ratio: kills divided by total attackers.
- Breach rate: breaches divided by total attackers.
- Completion time: simulated time until every attacker is killed or breached.
- First breach time: simulated time of the first pass-through.
- Target contention: duplicate target assignments and maximum simultaneous defenders assigned to one attacker.

## 4. Model Assumptions

This repository intentionally simplifies real-world physics and command-and-control. The assumptions are important because all conclusions are conditional on them.

### 4.1 Point-Mass Agents

Attackers and defenders are point masses in 3-D Euclidean space. There is no attitude model, aerodynamics, fuel budget, communication latency, terrain, radar cross-section, payload model, or collision-avoidance model.

### 4.2 Fixed Protected Target

The protected target is the origin:

```text
x_T = (0, 0, 0)
```

A breach occurs when an attacker intersects the breach sphere centered at the origin.

### 4.3 High-Load Attack Waves

Default Python configuration:

| Parameter | Value |
| --- | ---: |
| Initial defenders | 10 to 100 in steps of 10 |
| Attackers per wave | 250 to 350 |
| Waves | 2 to 3 |
| Wave spacing | 120 s |
| Spawn distance | 4.5 to 5.5 miles |
| Attacker speed | 18 to 25 m/s |
| Defender max speed | 45 m/s |
| Defender max acceleration | 15 m/s^2 |
| Time step | 0.25 s |
| Kill radius | 5 m |
| Kill dwell time | 3 s |
| Target breach radius | 5 m |
| Position observation noise | sigma = 1.0 m |
| Velocity observation noise | sigma = 0.5 m/s |

For later waves, reinforcements equal to `ceil(0.5 * initial_defender_count)` are activated at the wave start. Thus, with 100 initial defenders:

- A 2-wave scenario has 150 total defenders.
- A 3-wave scenario has 200 total defenders.

## 5. Scenario Generation

Scenario generation is implemented in `defensive_drones/scenario.py`.

### 5.1 Random Seeds and Scenario Cells

The Monte Carlo runner samples scenario seeds from a master random seed. For each sampled scenario seed, the simulator evaluates every defender count. The same seed is reused across strategies and defender counts. The attacker geometry is therefore comparable across resource levels; changing the defender count changes the corridor discretization and defensive resources, not the sampled attacker positions.

In `outputs_smoke_v4`:

| Quantity | Value |
| --- | ---: |
| Master seed | 42 |
| Monte Carlo scenario seeds | 25 |
| Defender counts | 10 |
| Scenario cells | 250 |
| Strategies | 9 |
| Strategy runs | 2,250 |
| Full mission successes | 0 |

The run log begins with:

```text
START runs=25 seed=42 ... defender_counts=10,20,30,40,50,60,70,80,90,100 scenario_cells=250 jobs=8
```

and ends with:

```text
FINISH strategy_runs=2250 successes=0 output_dir=outputs_smoke_v4
```

### 5.2 Attacker Spawn Geometry

The default approach cone is 180 degrees. In the implementation, this samples:

```text
cos(theta) ~ Uniform(cos(pi), 1) = Uniform(-1, 1)
phi        ~ Uniform(0, 2*pi)
```

The initial direction from the target to the attacker is:

```text
u = (cos(theta), sin(theta) cos(phi), sin(theta) sin(phi))
```

Because `cos(theta)` is uniform over `[-1, 1]` and `phi` is uniform over `[0, 2*pi)`, default attackers are spread across the full sphere. Distance is then sampled uniformly from 4.5 to 5.5 miles:

```text
p_a(0) = d * u
```

### 5.3 Target Point Sampling

Each attacker has its own target point inside the breach sphere, not merely the exact origin. The code samples a random direction and a radius:

```text
r = R_b * U^(1/3)
```

where `U ~ Uniform(0, 1)` and `R_b` is the breach radius. The `U^(1/3)` factor makes the sampled target point uniform in volume rather than biased toward the center.

### 5.4 Defender Initial Ring

Active defenders spawn on a ring around the target in the lateral `y-z` plane:

```text
p_d(0) = (0, R_d cos(alpha), R_d sin(alpha))
```

where `R_d = 120 m` and `alpha` is the midpoint angle of the defender's assigned corridor.

## 6. Corridors and Density-Based Rebalancing

Corridors are angular sectors in the `y-z` plane. The lateral angle for a position `p = (x, y, z)` is:

```text
angle(p) = atan2(z, y) mod 2*pi
```

If the initial defender count is `N_D`, the corridor index is:

```text
c(p) = floor(angle(p) / (2*pi / N_D))
```

with clipping to the range `0` through `N_D - 1`.

Corridor assignment matters because corridor strategies only allow a defender to select attackers inside its current corridor. Collaboration strategies use the defender's own corridor first, then fall back to off-corridor attackers only if the defender has no local live targets. Global strategies ignore corridors entirely.

At wave starts and when reinforcements activate, defender corridors are rebalanced according to live attacker density:

1. Count live active attackers per corridor.
2. If no corridor has attackers, distribute defenders as evenly as possible.
3. If there are at least as many nonempty corridors as defenders, assign one defender to the densest corridors first.
4. Otherwise, give each nonempty corridor one defender.
5. Allocate remaining defenders proportionally to attacker counts using a largest-remainder rule.

This gives dense sectors more defenders while preserving deterministic, repeatable allocation.

## 7. Dynamics and Mathematics

The engine is a fixed-step discrete simulation with time step:

```text
Delta t = 0.25 s
```

### 7.1 Attacker Kinematics

Each attacker has position `p_a`, velocity `v_a`, scalar speed `s_a`, target point `q_a`, maneuver amplitude `A_a`, maneuver frequency `omega_a`, and two random maneuver phases.

At each step, the target direction is:

```text
u_a(t) = (q_a - p_a(t)) / ||q_a - p_a(t)||
```

The simulator constructs two orthonormal lateral basis vectors `e_1(t)` and `e_2(t)` perpendicular to `u_a(t)`. The attacker velocity is then:

```text
v_a(t) = s_a u_a(t)
         + A_a [sin(phi_a + omega_a tau) e_1(t)
         + cos(psi_a + 0.73 omega_a tau) e_2(t)]
```

where:

```text
tau = max(0, t - spawn_time_a)
```

Position is updated by explicit Euler integration:

```text
p_a(t + Delta t) = p_a(t) + v_a(t) Delta t
```

This model preserves radial progress toward the sampled target point while adding bounded 3-D lateral maneuver.

### 7.2 Defender Controller

Each active defender receives a target assignment, if any, then computes a desired velocity.

If the assigned attacker is close, within 20 m, the defender attempts to match the observed attacker velocity while closing the remaining offset:

```text
v_des = v_obs + 0.8 (p_obs - p_d)
```

If the assigned attacker is farther away, the defender leads the target. The lead time is bounded by both the travel time at defender maximum speed and the observed attacker's time-to-target:

```text
lead = min(||p_obs - p_d|| / v_D_max, T_target_obs)
aim  = p_obs + v_obs * lead
v_des = v_D_max * unit(aim - p_d)
```

The desired command is then acceleration-limited:

```text
Delta v_cmd = v_des - v_d
||Delta v_cmd|| <= a_D_max Delta t
```

and speed-limited:

```text
||v_d(t + Delta t)|| <= v_D_max
```

The defender position update is:

```text
p_d(t + Delta t) = p_d(t) + v_d(t + Delta t) Delta t
```

### 7.3 Sensing Model

Strategies do not observe true attacker state exactly. At each time step:

```text
p_obs = p_true + epsilon_p,  epsilon_p ~ Normal(0, sigma_p^2 I)
v_obs = v_true + epsilon_v,  epsilon_v ~ Normal(0, sigma_v^2 I)
```

where:

```text
sigma_p = 1.0 m
sigma_v = 0.5 m/s
```

The physics engine continues to update true state. Noise only affects assignment and pursuit decisions.

### 7.4 Kill Condition

A defender does not kill an attacker on first contact. The kill condition requires sustained proximity:

```text
||p_d(t) - p_a(t)|| <= R_k
```

for a continuous dwell time:

```text
T_k = 3.0 s
```

with:

```text
R_k = 5.0 m
```

For every defender-attacker pair, the engine stores dwell time:

```text
m_ij(t + Delta t) =
    m_ij(t) + Delta t,  if ||p_i - p_j|| <= R_k
    0,                 otherwise
```

An attacker is killed when any active defender satisfies:

```text
m_ij >= T_k
```

The dwell timer is reset when the defender leaves the kill radius, when an attacker is inactive or resolved, or when a defender is inactive.

### 7.5 Breach Condition

An attacker breaches if its path segment over the current time step intersects the target breach sphere:

```text
S = {x : ||x - x_T|| <= R_b}
```

where:

```text
R_b = 5.0 m
```

The segment check is important because an attacker could otherwise numerically step across the sphere between discrete time samples. The engine checks whether the closest point on the segment from `p_a(t)` to `p_a(t + Delta t)` lies inside the breach sphere.

### 7.6 Terminal Condition

There is no hard time deadline. A run ends when every attacker has either been killed or breached. A run succeeds only if:

```text
breaches = 0
```

This zero-breach criterion is strict. In high-load experiments it can remain zero even when kill ratios are high.

## 8. Strategy Families

Strategies are implemented in `defensive_drones/strategies.py`. All strategies are autonomous: there is no shared assignment table, no centralized matching, and no target deconfliction. Each defender independently scores candidates and chooses the best target under its policy.

### 8.1 Assignment Modes

There are three assignment modes:

| Mode | Behavior |
| --- | --- |
| Corridor | A defender only considers attackers in its assigned corridor. |
| Global | A defender can consider all active observed attackers. |
| Collaboration | A defender first considers own-corridor attackers. If none exist, it may assist another corridor. |

### 8.2 Base Scoring Rules

There are three base scoring rules:

#### Nearest

The defender chooses the candidate with minimum observed distance:

```text
score_i = ||p_obs,i - p_d||
```

This is a strong local baseline because the kill condition rewards sustained proximity.

#### Earliest Deadline

The defender chooses the candidate with the shortest estimated time remaining to that attacker's sampled target point:

```text
T_i = ||p_obs,i - q_i|| / ||v_obs,i||
score_i = T_i
```

This prioritizes imminent threats, but it may send defenders on long pursuit paths.

#### Optimized

The optimized policy is a receding-horizon heuristic, not a proof of global optimality. It uses a conservative rendezvous estimate:

```text
tau_i = ||p_obs,i - p_d|| / v_D_max
```

Then it estimates slack:

```text
slack_i = T_i - (tau_i + T_k)
```

where `T_i` is the attacker's estimated deadline and `T_k` is the kill dwell time.

If slack is nonnegative, the target is feasible under the conservative estimate:

```text
score_i = tau_i + 0.01 T_i
```

If slack is negative, the target is penalized heavily:

```text
score_i = 10000 + 500 |slack_i| + tau_i
```

For collaboration mode, off-corridor assistance receives a small additive penalty:

```text
score_i = score_i + 2.0
```

### 8.3 Near-Capture Lock

If a defender was already assigned to a target and is dwelling or close to capture, the strategy can hold that target instead of replanning away. This protects partially accumulated dwell time. For corridor mode, the held target must remain in the defender's corridor. For global and collaboration modes, assisted or cross-corridor near-capture locks can be retained.

### 8.4 Nine Strategy Labels

| Strategy | Base rule | Assignment mode |
| --- | --- | --- |
| `optimized` | Optimized score | Corridor |
| `nearest` | Nearest target | Corridor |
| `earliest_deadline` | Earliest deadline | Corridor |
| `optimized_global` | Optimized score | Global |
| `nearest_global` | Nearest target | Global |
| `earliest_deadline_global` | Earliest deadline | Global |
| `optimized_collab` | Optimized score | Collaboration |
| `nearest_collab` | Nearest target | Collaboration |
| `earliest_deadline_collab` | Earliest deadline | Collaboration |

The word "global" in this repository means global visibility, not centralized global optimization. In the current implementation, global variants often herd many defenders onto the same target because each defender makes the same independent decision from a similar global candidate set.

## 9. Metrics

Let `A` be total attackers in a run, `K` kills, and `B` breaches. Every attacker resolves as either killed or breached, so:

```text
A = K + B
```

### 9.1 Mission Success

```text
success = 1 if B = 0 else 0
```

### 9.2 Kill Ratio

```text
kill_ratio = K / A
```

### 9.3 Breach Rate

```text
breach_rate = B / A
```

Since every attacker resolves:

```text
breach_rate = 1 - kill_ratio
```

up to rounding.

### 9.4 Target Contention

At each time step, the simulator counts how many defenders are assigned to each attacker. If an attacker has `n` assigned defenders, then `n - 1` of those assignments are duplicates. The duplicate target assignment count accumulates across time steps:

```text
duplicate_assignments += sum(max(0, n_j - 1))
```

The contention rate is:

```text
contention_rate = duplicate_assignments / total_assignments
```

The maximum simultaneous defenders on target is:

```text
max_j,t n_j(t)
```

High contention means the strategy wastes defender capacity by sending many defenders after the same attacker.

## 10. Output Files

The standard output directory contains:

| File | Meaning |
| --- | --- |
| `per_run_results.csv` | One row per strategy run. This is the primary table for aggregate results. |
| `wave_summary.csv` | One row per attack wave per strategy run. Useful for wave-level attrition and breach analysis. |
| `summary.csv` | Aggregates by strategy, defender count, and attacker bucket. |
| `success_rate_matrix.csv` | Success-rate matrix by strategy, defender count, and attacker-count bucket. |
| `breach_rate_matrix.csv` | Pass-through-rate matrix by strategy, defender count, and attacker-count bucket. |
| `run_status.log` | Timestamped simulation progress and final run status. |
| `*.png` | Baseline result plots generated by `reporting.py`. |
| `publication_figures/*.png` and `*.pdf` | Publication-style figures generated by `scripts/generate_publication_figures.py`. |

The requested `outputs_smoke_v4` result set includes all of these artifacts.

## 11. Experimental Dataset: `outputs_smoke_v4`

The smoke-v4 result set is smaller than the README default. The README describes a default 1,000-scenario run, which would produce 90,000 strategy runs across 10 defender counts and 9 strategies. The smoke-v4 run used 25 scenario seeds and produced 2,250 strategy runs.

### 11.1 Dataset Shape

| Item | Value |
| --- | ---: |
| Strategy runs | 2,250 |
| Unique scenario seeds | 25 |
| Scenario cells | 250 |
| Rows per strategy | 250 |
| Initial defender counts | 10, 20, ..., 100 |
| Attackers per scenario | 534 to 1,005 |
| Mean attackers per scenario | 746.56 |
| 2-wave scenario seeds | 13 |
| 3-wave scenario seeds | 12 |
| Full mission successes | 0 |

### 11.2 Attacker Count Distribution

| Attacker bucket | Scenario seeds | Scenario cells | Strategy-run rows |
| --- | ---: | ---: | ---: |
| 500-599 | 7 | 70 | 630 |
| 600-700 | 6 | 60 | 540 |
| 750-849 | 3 | 30 | 270 |
| 850-949 | 6 | 60 | 540 |
| 950-1050 | 3 | 30 | 270 |

## 12. Smoke-v4 Results

### 12.1 Main Interpretation

No strategy achieved full mission success in `outputs_smoke_v4`. This does not mean all strategies behaved equivalently. The zero-breach criterion is extremely strict under the tested force ratios. Kill ratio, breach rate, and contention reveal large differences.

The best overall attrition strategy was `optimized_collab`:

```text
mean kill ratio  = 0.7655
mean breach rate = 0.2345
mean kills       = 567.86 attackers per run
```

The next strongest policies were `optimized`, `nearest_collab`, and `nearest`. The global variants performed poorly because global visibility without central deconfliction created severe target herding.

### 12.2 Overall Strategy Performance

Sorted by mean kill ratio:

| Strategy | Success rate | Mean kills | Mean kill ratio | Mean breaches | Mean breach rate | Median completion | Median first breach | Mean contention | Peak target contention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_collab` | 0.0% | 567.86 | 0.7655 | 178.70 | 0.2345 | 597.50 s | 303.75 s | 0.3452 | 200 |
| `optimized` | 0.0% | 553.79 | 0.7460 | 192.77 | 0.2540 | 600.25 s | 303.75 s | 0.2716 | 4 |
| `nearest_collab` | 0.0% | 547.78 | 0.7398 | 198.78 | 0.2602 | 600.50 s | 303.00 s | 0.3466 | 200 |
| `nearest` | 0.0% | 534.79 | 0.7219 | 211.77 | 0.2781 | 601.25 s | 303.00 s | 0.2724 | 4 |
| `earliest_deadline_collab` | 0.0% | 483.36 | 0.6568 | 263.20 | 0.3432 | 601.25 s | 323.00 s | 0.4221 | 200 |
| `earliest_deadline` | 0.0% | 482.94 | 0.6562 | 263.62 | 0.3438 | 601.25 s | 323.00 s | 0.3535 | 3 |
| `optimized_global` | 0.0% | 96.84 | 0.1357 | 649.72 | 0.8643 | 609.25 s | 303.50 s | 0.8909 | 200 |
| `nearest_global` | 0.0% | 84.86 | 0.1195 | 661.70 | 0.8805 | 609.25 s | 301.25 s | 0.9118 | 200 |
| `earliest_deadline_global` | 0.0% | 2.40 | 0.0034 | 744.16 | 0.9966 | 609.25 s | 300.00 s | 0.9810 | 200 |

Key points:

- `optimized_collab` has the highest mean kill ratio and lowest mean breach rate overall.
- `optimized` is close behind and has much lower peak contention than collaboration/global modes.
- `nearest` is a strong baseline because the dwell kill condition rewards proximity.
- `earliest_deadline` delays first breach relative to nearest and optimized, but its overall kill ratio is lower at low and moderate defender counts.
- Global variants are not centralized assignment algorithms. They show what happens when every defender can see every attacker but still chooses independently. This causes severe target herding.

### 12.3 Assignment Contention

The contrast between corridor and global policies is stark:

| Strategy family | Corridor mean contention | Collaboration mean contention | Global mean contention |
| --- | ---: | ---: | ---: |
| Optimized | 0.2716 | 0.3452 | 0.8909 |
| Nearest | 0.2724 | 0.3466 | 0.9118 |
| Earliest deadline | 0.3535 | 0.4221 | 0.9810 |

The global modes also reach peak simultaneous target contention of 200 defenders on one attacker in 3-wave, 100-initial-defender cases. This explains the poor global attrition result: more visibility without deconfliction lets many defenders independently select the same target, leaving many attackers uncontested.

### 12.4 Defender Resource Scaling

Mean kill ratio by initial defender count:

| Initial defenders | `optimized_collab` | `optimized` | `nearest_collab` | `nearest` | `earliest_deadline_collab` | `earliest_deadline` | `optimized_global` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.247 | 0.246 | 0.193 | 0.193 | 0.077 | 0.077 | 0.120 |
| 20 | 0.497 | 0.494 | 0.422 | 0.420 | 0.231 | 0.231 | 0.130 |
| 30 | 0.671 | 0.662 | 0.621 | 0.611 | 0.418 | 0.417 | 0.133 |
| 40 | 0.785 | 0.767 | 0.757 | 0.740 | 0.594 | 0.594 | 0.138 |
| 50 | 0.841 | 0.821 | 0.826 | 0.807 | 0.728 | 0.728 | 0.139 |
| 60 | 0.878 | 0.853 | 0.870 | 0.848 | 0.816 | 0.816 | 0.138 |
| 70 | 0.911 | 0.884 | 0.903 | 0.877 | 0.878 | 0.877 | 0.140 |
| 80 | 0.929 | 0.899 | 0.923 | 0.895 | 0.918 | 0.918 | 0.138 |
| 90 | 0.940 | 0.911 | 0.936 | 0.908 | 0.946 | 0.945 | 0.139 |
| 100 | 0.956 | 0.924 | 0.949 | 0.920 | 0.962 | 0.959 | 0.142 |

The most important pattern is that corridor and collaboration policies scale strongly with defender count, while global policies barely improve. For example:

- `optimized_collab` rises from kill ratio 0.247 at 10 defenders to 0.956 at 100 defenders.
- `optimized` rises from 0.246 to 0.924.
- `earliest_deadline_collab` starts very weak at 10 defenders but reaches 0.962 at 100 defenders.
- `optimized_global` stays near 0.12 to 0.14 across the entire defender range because extra defenders mostly increase duplicate assignments.

### 12.5 Attacker Load Scaling

Mean kill ratio by attacker-count bucket:

| Attacker bucket | Scenario cells | `optimized_collab` | `optimized` | `nearest_collab` | `earliest_deadline_collab` | `optimized_global` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500-599 | 70 | 0.793 | 0.770 | 0.773 | 0.709 | 0.169 |
| 600-700 | 60 | 0.771 | 0.751 | 0.749 | 0.670 | 0.149 |
| 750-849 | 30 | 0.769 | 0.749 | 0.740 | 0.659 | 0.122 |
| 850-949 | 60 | 0.745 | 0.730 | 0.714 | 0.615 | 0.105 |
| 950-1050 | 30 | 0.727 | 0.711 | 0.693 | 0.591 | 0.106 |

As attacker load rises, all useful strategies degrade, but not equally. `optimized_collab` preserves the highest attrition across every attacker bucket. The global policy remains poor across all loads because contention, not only attacker volume, dominates its behavior.

### 12.6 Wave-Level Results

Mean kill ratio by wave:

| Strategy | Wave 0 | Wave 1 | Wave 2 |
| --- | ---: | ---: | ---: |
| `optimized_collab` | 0.780 | 0.756 | 0.739 |
| `optimized` | 0.773 | 0.734 | 0.700 |
| `nearest_collab` | 0.763 | 0.726 | 0.698 |
| `earliest_deadline_collab` | 0.741 | 0.595 | 0.574 |
| `earliest_deadline` | 0.741 | 0.594 | 0.573 |
| `optimized_global` | 0.281 | 0.031 | 0.021 |

Wave-level interpretation:

- `optimized_collab` remains relatively stable across waves.
- Corridor-only `optimized` degrades more by wave 2.
- Global optimized collapses after wave 0. Once many defenders are active, global independent choice creates extreme assignment duplication.
- Some later individual waves are resolved without breaches under collaboration policies, but full runs still fail because the mission success metric requires zero breaches over all waves.

### 12.7 Paired Policy Comparisons

Because strategies are evaluated on matched scenario cells, pairwise comparisons isolate algorithmic differences from scenario randomness.

| New policy vs baseline | Mean kill-ratio delta | Mean kill delta | Improved kills | Tied | Worse | Success conversions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_collab` vs `optimized` | +0.0195 | +14.07 | 93.2% | 6.4% | 0.4% | 0 |
| `optimized_global` vs `optimized` | -0.6102 | -456.94 | 0.0% | 0.0% | 100.0% | 0 |
| `nearest_collab` vs `nearest` | +0.0179 | +12.99 | 87.2% | 12.4% | 0.4% | 0 |
| `nearest_global` vs `nearest` | -0.6024 | -449.93 | 0.0% | 0.0% | 100.0% | 0 |
| `earliest_deadline_collab` vs `earliest_deadline` | +0.0006 | +0.41 | 27.2% | 70.4% | 2.4% | 0 |
| `earliest_deadline_global` vs `earliest_deadline` | -0.6528 | -480.54 | 0.0% | 0.0% | 100.0% | 0 |

This table is one of the clearest results in the repository:

- Collaboration helps optimized and nearest policies modestly but consistently.
- Global visibility without deconfliction is harmful in every matched comparison.
- Deadline collaboration mostly ties deadline corridor, because deadline prioritization already tends to concentrate defenders on urgent threats.

### 12.8 Best-Per-Scenario Attrition Winners

Counting the highest-kill strategy within each matched scenario cell:

| Strategy | Strict best cells |
| --- | ---: |
| `optimized_collab` | 168 |
| `earliest_deadline_collab` | 31 |
| `nearest_collab` | 12 |
| `optimized` | 1 |
| `earliest_deadline` | 1 |

There are 250 scenario cells total. Ties account for the remaining cells when no single strict winner exists. `optimized_collab` is the dominant attrition winner over the full smoke-v4 design, while `earliest_deadline_collab` becomes competitive in the highest defender-count regime.

## 13. Figures in `outputs_smoke_v4`

The publication figure index is in:

```text
outputs_smoke_v4/publication_figures/figure_index.md
```

Important figures for this smoke-v4 interpretation:

![Mean kill ratio by strategy](outputs_smoke_v4/publication_figures/02_mean_kill_ratio_by_strategy.png)

![Mean kills by strategy](outputs_smoke_v4/publication_figures/03_mean_kills_by_strategy.png)

![Success rate by defender count](outputs_smoke_v4/publication_figures/04_success_rate_by_defender_count.png)

![Kill ratio distribution](outputs_smoke_v4/publication_figures/07_kill_ratio_distribution.png)

![Terminal event timing](outputs_smoke_v4/publication_figures/08_terminal_time_distribution.png)

Note that success-rate figures are flat at zero for this result set because every full mission has at least one breach. They are still useful as a stress-test signal, but attrition, breach-rate, and contention figures are more scientifically informative here.

## 14. Implementation Details Worth Knowing

### 14.1 Simulation Loop

The main loop in `engine.py` performs:

1. Activate attackers whose wave spawn time has arrived.
2. Activate defender reinforcements whose wave spawn time has arrived.
3. Rebalance defender corridors when waves or reinforcements activate.
4. If all attackers are inactive/resolved, return a `RunResult`.
5. If no attackers are currently active but future waves remain, jump simulation time to the next spawn.
6. Generate noisy observations of live active attackers.
7. Choose independent defender assignments.
8. Record assignment contention.
9. Advance defenders under speed and acceleration limits.
10. Advance attackers under target-seeking maneuver dynamics.
11. Update dwell timers and mark kills.
12. Check segment-sphere breach crossings and mark breaches.

### 14.2 Why the Simulator Continues After Breach

The engine does not stop on the first breach. It continues until all attackers are resolved. This is essential because the repository studies not only binary mission success but also attrition, pass-through rates, completion times, and failure severity.

### 14.3 Computational Complexity

For each time step, the dominant operations scale roughly as:

```text
O(D * A)
```

where `D` is active defenders and `A` is live active attackers. Assignment scoring evaluates candidates for each defender, and dwell updates compute defender-attacker distances. This makes large Monte Carlo runs expensive. `simulate.py` therefore supports process-level parallelism through `ProcessPoolExecutor`.

The smoke-v4 log shows the 25-seed, 2,250-run job running with 8 jobs from 2026-05-28T05:15:25+00:00 to 2026-05-28T11:24:13+00:00.

### 14.4 Reproducibility

To recreate a smoke-v4-style run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m defensive_drones.simulate --runs 25 --seed 42 --out outputs_smoke_v4 --jobs 8
python scripts/generate_publication_figures.py \
  --input outputs_smoke_v4/per_run_results.csv \
  --out outputs_smoke_v4/publication_figures
```

To run the tests:

```bash
python -m unittest
```

## 15. Test Coverage and Verification Logic

The unit tests cover the main mechanics:

- Fixed seeds produce repeatable scenarios.
- Attackers have independently sampled speeds.
- Corridor assignment uses lateral `atan2(z, y)` angle.
- Default attackers are high-load and spawn near 5 miles.
- Full-sphere spawns occupy many corridors.
- Attackers target random points inside the breach radius.
- Reinforcements activate at their wave spawn time.
- Density rebalancing assigns more defenders to denser corridors.
- Future waves are not observed or targetable before spawn.
- Dwell kills require continuous time inside the kill radius.
- Dwell timers reset after leaving the kill radius.
- Segment-sphere breach detection catches between-step crossings.
- Simulation continues after the first breach.
- Contention metrics record duplicate assignments.
- Inactive defenders cannot be assigned and cannot kill.
- Global strategies can target outside defender corridors.
- Collaboration strategies keep own-corridor targets when available and assist only when local targets are absent.
- Output smoke tests verify CSV, PNG, matrix, log, and publication-figure generation.

This is good coverage for model mechanics and output production. It is not a validation against real flight data.

## 16. Limitations

The main scientific limitations are:

- Point-mass dynamics omit real aircraft constraints.
- No communication model exists, even though communication would be central to real deconfliction.
- There is no centralized assignment solver such as Hungarian matching, auction assignment, or min-cost flow.
- "Optimized" is a local receding-horizon heuristic, not a globally optimal controller.
- Observation noise is simple Gaussian noise with fixed standard deviations.
- Attackers do not coordinate, adapt, or react to defenders.
- Defenders do not run out of energy or suffer failures.
- The target is a sphere with a fixed breach radius.
- The kill model is a dwell-radius abstraction.
- The smoke-v4 result set has 25 scenario seeds, which is useful for stress testing but smaller than the 1,000-seed default described in the README.

The most important interpretation limitation is the naming of global strategies. `optimized_global` sounds as though it should be the strongest policy, but in this code it means independent global target choice. Without deconfliction, global choice creates target herding and severe underperformance.

## 17. Conclusions from `outputs_smoke_v4`

The smoke-v4 experiment is a saturation-regime stress test. No strategy prevents all breaches, but the policies differ sharply in how much of the swarm they stop.

The main conclusions are:

1. Binary success rate is saturated at 0.0% for all strategies in this result set.
2. Attrition metrics reveal meaningful differences despite universal mission failure.
3. `optimized_collab` is the best overall attrition strategy, with mean kill ratio 0.7655.
4. Collaboration improves optimized and nearest policies in paired matched scenarios.
5. Pure global visibility is harmful because defenders independently duplicate assignments.
6. Corridor-restricted policies avoid extreme herding and are far stronger than global policies here.
7. Increasing defender count greatly improves useful corridor and collaboration policies.
8. Even high defender counts still produce at least one breach under the tested load and dwell requirement.
9. Future gains likely require real deconfliction, not merely more permissive target visibility.

## 18. Recommended Future Work

The next research steps should target the actual failure modes seen in the data:

- Add central or distributed assignment deconfliction.
- Penalize targets that are already assigned, even in independent strategies.
- Compare local heuristics against Hungarian assignment, auction assignment, and min-cost flow.
- Run sensitivity sweeps for kill dwell time, kill radius, defender speed, defender acceleration, and reinforcement fraction.
- Add confidence intervals and repeat larger 1,000-seed experiments for publication-grade statistical power.
- Separate first-wave, second-wave, and third-wave policy behavior in more detail.
- Test whether `earliest_deadline_collab` becomes superior only above a defender-count threshold.
- Use the exact constant-speed intercept helper in assignment scoring and compare it against the current conservative distance-over-speed estimate.
- Model communication limits explicitly so deconfliction assumptions are transparent.
- Add attacker adaptation or decoys to test robustness.

## Appendix A: Strategy Result Table in Compact Form

| Rank by kill ratio | Strategy | Mean kill ratio | Mean breach rate | Mean contention |
| ---: | --- | ---: | ---: | ---: |
| 1 | `optimized_collab` | 0.7655 | 0.2345 | 0.3452 |
| 2 | `optimized` | 0.7460 | 0.2540 | 0.2716 |
| 3 | `nearest_collab` | 0.7398 | 0.2602 | 0.3466 |
| 4 | `nearest` | 0.7219 | 0.2781 | 0.2724 |
| 5 | `earliest_deadline_collab` | 0.6568 | 0.3432 | 0.4221 |
| 6 | `earliest_deadline` | 0.6562 | 0.3438 | 0.3535 |
| 7 | `optimized_global` | 0.1357 | 0.8643 | 0.8909 |
| 8 | `nearest_global` | 0.1195 | 0.8805 | 0.9118 |
| 9 | `earliest_deadline_global` | 0.0034 | 0.9966 | 0.9810 |

## Appendix B: Source of Results

All numeric results in this document were recomputed from:

```text
outputs_smoke_v4/per_run_results.csv
outputs_smoke_v4/wave_summary.csv
outputs_smoke_v4/run_status.log
```

The generated `outputs_smoke_v4/publication_figures/talking_points.md` is useful as a figure guide, but this document treats the raw CSVs as authoritative.
