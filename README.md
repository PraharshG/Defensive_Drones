# Defensive_Drones

A 3D Monte Carlo simulation for studying target allocation strategies for
defensive drones. The simulation is intentionally non-visual: it produces CSV
tables and PNG graphs that compare defensive strategies over randomized attack
scenarios.

This is an abstract research simulator. It models drones as point masses and is
intended for strategy comparison, not real flight control or weapons guidance.

## Scenario

Each run creates one randomized 3D attack scenario:

- 2-5 defensive drones protect a fixed target at `(0, 0, 0)`.
- 5-20 attack drones spawn 800-1500 m away.
- Attack drones come from the same approximate direction, sampled inside a
  12-degree cone.
- Each attack drone has its own sampled speed from 18-35 m/s.
- Attack drone velocity is stored as state. In this first version, speeds are
  constant, but the model is structured so future versions can update velocity
  over time.
- Defensive drones start near the target on a defensive ring.
- Each defensive drone owns one angular corridor and only engages attack drones
  inside that corridor.

The simulation has no hard time deadline. A run ends when either all attack
drones are down or at least one live attack drone reaches the target.

## Theory

### State Model

Each attack drone has:

- position `p_a` in 3D
- velocity `v_a`
- scalar speed
- assigned corridor
- live, killed, and breached status

Each defensive drone has:

- position `p_d` in 3D
- velocity `v_d`
- assigned corridor

Attack drones move directly toward the target with constant velocity:

```text
p_a(t + dt) = p_a(t) + v_a * dt
```

Defensive drones use a point-mass model with a maximum speed and acceleration:

```text
max speed        = 45 m/s
max acceleration = 15 m/s^2
time step        = 0.25 s
```

The controller chooses a desired velocity each time step, then clamps the
velocity change by the acceleration limit and clamps final velocity by the max
speed.

### Corridors

Corridors are angular sectors around the attack approach axis. The simulator
uses the lateral `y-z` angle of each attacker position:

```text
angle = atan2(z, y)
```

The full circle is split into one sector per defender. This creates hard
ownership: defender `i` can only target attackers whose lateral angle falls in
corridor `i`.

This keeps the first version simple and makes strategy comparison easier because
the assignment problem is local to each defender's corridor.

### Sensing

Strategies do not receive perfect state. At every time step, each live
attacker's observed position and velocity include Gaussian noise:

```text
position noise sigma = 1.0 m
velocity noise sigma = 0.5 m/s
```

The physics engine still updates the true underlying state.

### Kill and Failure Conditions

An attack drone is counted as killed when any defender remains within 5 m of it
for 3 continuous seconds:

```text
distance(defender, attacker) <= 5 m for 3.0 s
```

The dwell timer resets if the defender leaves the 5 m radius before the 3-second
requirement is met.

A scenario fails immediately when any live attack drone reaches the target
breach radius:

```text
distance(attacker, target) <= 5 m
```

Segment crossing is checked each time step so a fast attacker cannot skip past
the target between discrete updates.

### Strategies

The simulator compares three strategies on the same randomized scenarios:

1. `optimized`
   - Uses receding-horizon assignment.
   - Replans every time step from noisy observations.
   - Uses `scipy.optimize.linear_sum_assignment` for defender-to-attacker
     matching.
   - Scores targets by conservative rendezvous time plus target deadline risk.
   - Keeps a defender locked on a near-capture target to preserve dwell time.

2. `nearest`
   - Each defender chooses the nearest live attacker in its corridor.
   - This is a strong baseline because the kill condition requires sustained
     proximity.

3. `earliest_deadline`
   - Each defender chooses the live corridor attacker with the shortest
     estimated time to target.
   - This prioritizes imminent threats but can underperform when it causes long
     pursuit paths.

The term "optimized" here means optimized under this simplified simulation
model. It is not a proof of global optimality in real-world conditions.

## Outputs

The default run evaluates 1000 randomized scenarios across all three strategies,
for 3000 strategy runs total.

Outputs are written to `outputs/`:

- `per_run_results.csv`: one row per strategy run
- `summary.csv`: grouped aggregate metrics
- `success_rate_by_strategy.png`
- `success_rate_by_defender_count.png`
- `success_rate_by_attacker_count.png`
- `kill_ratio_distribution.png`
- `completion_time_distribution.png`

Key metrics include:

- success rate: fraction of runs where all attackers are killed
- kills: number of attack drones killed before the run ends
- breaches: number of attackers that reached the target
- completion time: simulated seconds until success or breach
- kill ratio: kills divided by total attackers

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Simulation

Run the default 1000-scenario simulation:

```bash
python -m defensive_drones.simulate --runs 1000 --seed 42 --out outputs
```

Equivalent command without activating the environment:

```bash
.venv/bin/python -m defensive_drones.simulate --runs 1000 --seed 42 --out outputs
```

Run a smaller smoke simulation:

```bash
python -m defensive_drones.simulate --runs 10 --seed 42 --out outputs_smoke
```

Run one strategy only:

```bash
python -m defensive_drones.simulate --runs 100 --seed 42 --out outputs_optimized --strategies optimized
```

Run multiple selected strategies:

```bash
python -m defensive_drones.simulate --runs 100 --seed 42 --out outputs_compare --strategies optimized nearest
```

## Run Tests

```bash
python -m unittest
```

Or without activating the environment:

```bash
.venv/bin/python -m unittest
```

The tests cover deterministic scenario generation, independent attacker speeds,
corridor assignment, dwell-time kill logic, target breach detection, and output
artifact generation.

## Current Baseline Result

The committed `outputs/` directory was generated with:

```bash
.venv/bin/python -m defensive_drones.simulate --runs 1000 --seed 42 --out outputs
```

Overall aggregate results from that run:

| Strategy | Runs | Success rate | Average kills | Average breaches |
| --- | ---: | ---: | ---: | ---: |
| optimized | 1000 | 0.012 | 1.696 | 1.007 |
| nearest | 1000 | 0.012 | 1.685 | 1.007 |
| earliest_deadline | 1000 | 0.010 | 0.720 | 1.006 |

The low success rates are expected under the current default assumptions: hard
corridor ownership, 5-20 attackers, 2-5 defenders, constant attacker motion
toward the target, and a strict 3-second dwell requirement. The optimized
strategy is slightly better than the nearest baseline on average kills in this
specific run, while success rate remains nearly identical.

## Project Layout

```text
defensive_drones/
  engine.py      simulation loop and physics updates
  geometry.py    vector math, corridor indexing, breach checks
  model.py       dataclasses and default configuration
  reporting.py   CSV and PNG output generation
  scenario.py    randomized scenario generation
  simulate.py    command line entry point
  strategies.py  target assignment strategies
tests/
  test_simulation.py
outputs/
  generated CSV and PNG results
```

## Limitations and Future Work

The first version deliberately excludes battery limits, communication delay,
collision avoidance, real flight dynamics, payload constraints, and visual
simulation.

Useful next improvements:

- allow attacker speeds and headings to change over time
- tune defender speed, acceleration, spawn distance, and dwell-time parameters
- add soft corridor handoff between neighboring defenders
- add layered defense corridors based on distance from target
- add richer summary tables for per-corridor load and kill timing
