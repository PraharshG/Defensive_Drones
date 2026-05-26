# Defensive_Drones

A 3D Monte Carlo simulation for studying target allocation strategies for
defensive drones. The simulation is intentionally non-visual: it produces CSV
tables and PNG graphs that compare defensive strategies over randomized attack
scenarios.

This is an abstract research simulator. It models drones as point masses and is
intended for strategy comparison, not real flight control or weapons guidance.

## Scenario

Each run creates one randomized 3D attack scenario:

- 10-100 defensive drones protect a fixed target at `(0, 0, 0)`, evaluated in
  steps of 10 defenders.
- Attack drones arrive in 2-3 waves. Each wave contains 250-350 attackers that
  spawn roughly 5 miles away, with randomized distance from 4.5-5.5 miles.
- Attack drones are distributed across a wide positive-side hemisphere rather
  than coming from one narrow approach area.
- Each attack drone has its own sampled speed from 18-25 m/s.
- Attack drone velocity is stored as state and changes over time in all three
  dimensions through bounded target-seeking maneuvers.
- Defensive drones start near the target on a defensive ring.
- Defensive drones do not communicate with each other. Each defender makes an
  independent target choice from its own observations, so multiple defenders can
  pursue the same attacker.
- Each defensive drone has an angular corridor label. Corridor strategies obey
  those labels; global strategies ignore them; collaboration strategies use a
  corridor-first fallback rule when a defender has no local targets.

The simulation has no hard time deadline. A run ends when all attack drones are
resolved: each attacker is either killed by a defender or passes through the
target breach radius.

## Theory

### State Model

Each attack drone has:

- position `p_a` in 3D
- velocity `v_a`
- scalar speed
- assigned corridor
- wave id and spawn time
- live, killed, and breached status

Each defensive drone has:

- position `p_d` in 3D
- velocity `v_d`
- assigned corridor

Attack drones maintain target-seeking radial progress while adding bounded
lateral maneuver velocity. This makes `x`, `y`, and `z` velocity components vary
over time:

```text
v_a(t) = target_direction * base_speed + bounded_lateral_maneuver(t)
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

The full circle is split into one sector per defender. Corridor-restricted
strategies let each defender consider only attackers whose lateral angle falls
in that defender's corridor. Global strategies let each defender consider every
observed active attacker. Collaboration strategies first use the defender's own
corridor and fall back to off-corridor targets only when the defender has no
local candidates.

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

An attack drone is counted as a breach when it reaches the target breach radius:

```text
distance(attacker, target) <= 5 m
```

Segment crossing is checked each time step so a fast attacker cannot skip past
the target between discrete updates. The simulation no longer terminates on the
first breach; it continues so the final result captures how many attackers pass
through. A scenario succeeds only when zero attackers breach.

### Strategies

The simulator compares nine strategy labels on the same randomized scenarios.
All strategies are autonomous: there is no shared assignment table and no target
deconfliction between defenders.

1. `optimized`
   - Uses receding-horizon assignment.
   - Replans every time step from noisy observations.
   - Each defender independently scores candidate attackers.
   - Scores targets by conservative rendezvous time plus target deadline risk.
   - Keeps a defender locked on a near-capture target to preserve dwell time.
   - Only considers attackers inside the defender's corridor.

2. `nearest`
   - Each defender chooses the nearest live attacker in its corridor.
   - This is a strong baseline because the kill condition requires sustained
     proximity.

3. `earliest_deadline`
   - Each defender chooses the live corridor attacker with the shortest
     estimated time to target.
   - This prioritizes imminent threats but can underperform when it causes long
     pursuit paths.

4. `optimized_global`
   - Same assignment objective as `optimized`, but it removes corridor limits.
   - Any defender can independently choose any live observed attacker.
   - Near-capture lock also ignores corridors so a defender does not abandon a
     dwell capture after crossing sector boundaries.

5. `nearest_global`
   - Same as `nearest`, but searches all live attackers rather than only the
     defender's corridor.

6. `earliest_deadline_global`
   - Same as `earliest_deadline`, but searches all live attackers rather than
     only the defender's corridor.

7. `optimized_collab`
   - Uses the optimized corridor assignment first.
   - If a defender's own corridor has no live attackers, that defender may
     independently choose an off-corridor attacker.
   - Assisted assignments use the same rendezvous/deadline cost with a small
     off-corridor collaboration penalty.

8. `nearest_collab`
   - Uses nearest-target corridor assignment first.
   - Defenders with empty corridors choose the nearest off-corridor target.

9. `earliest_deadline_collab`
   - Uses earliest-deadline corridor assignment first.
   - Defenders with empty corridors choose the most urgent off-corridor target.

The term "optimized" here means optimized under this simplified simulation
model. It is not a proof of global optimality in real-world conditions.

## Outputs

The default run evaluates 1000 randomized attacker-wave scenarios across 10
defender counts and all nine strategies, for 90,000 strategy runs total. Each
attacker-wave scenario is reused across the defender-count sweep so changes in
outcome can be attributed to defender resources and strategy behavior.

Outputs are written to `outputs/`:

- `per_run_results.csv`: one row per strategy run
- `wave_summary.csv`: one row per attacker wave per strategy run
- `summary.csv`: grouped aggregate metrics
- `success_rate_matrix.csv`: defense-drone count by attack-drone bucket
  success-rate matrix for each strategy
- `breach_rate_matrix.csv`: defense-drone count by attack-drone bucket
  pass-through-rate matrix for each strategy
- `run_status.log`: timestamped start, per-scenario, per-strategy, and finish
  status entries
- `success_rate_by_strategy.png`
- `success_rate_by_defender_count.png`
- `success_rate_by_attacker_count.png`
- `kill_ratio_distribution.png`
- `completion_time_distribution.png`
- `breach_rate_by_defender_count.png`
- `avg_breaches_by_defender_count.png`
- `breach_rate_distribution.png`
- `first_breach_time_distribution.png`
- `breach_rate_by_wave.png`
- `kill_ratio_by_wave.png`
- `contention_rate_by_strategy.png`
- `max_contention_distribution.png`

Key metrics include:

- success rate: fraction of runs where all attackers are killed
- kills: number of attack drones killed before the run ends
- breaches: number of attackers that reached the target
- breach rate: breaches divided by total attackers
- first breach time: simulated seconds until the first attacker passes through
- completion time: simulated seconds until all attackers are killed or breached
- kill ratio: kills divided by total attackers
- duplicate target assignments: extra defenders assigned to already selected
  targets in a time step
- contention rate: duplicate target assignments divided by total assignments
- max simultaneous defenders on target: peak number of defenders independently
  assigned to the same attacker

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
python -m defensive_drones.simulate --runs 1000 --seed 42 --out outputs --jobs 8
```

Equivalent command without activating the environment:

```bash
.venv/bin/python -m defensive_drones.simulate --runs 1000 --seed 42 --out outputs --jobs 8
```

Run a smaller smoke simulation:

```bash
python -m defensive_drones.simulate --runs 10 --seed 42 --out outputs_smoke
```

## Run on a Server

Check out this branch and run the server script:

```bash
git checkout praharsh_dev_v3
RUNS=1000 JOBS=$(nproc) OUT_DIR=outputs_server bash scripts/run_server_simulation.sh
```

Useful environment variables:

- `RUNS`: number of randomized attacker-wave scenarios, default `1000`
- `SEED`: master random seed, default `42`
- `JOBS`: parallel scenario cells, default `nproc`
- `OUT_DIR`: output directory, default `outputs_server`
- `GENERATE_FIGURES`: set to `0` to skip publication figure generation
- `STRATEGIES`: optional space-separated strategy list

Small server smoke check:

```bash
RUNS=1 JOBS=1 GENERATE_FIGURES=0 OUT_DIR=/tmp/defensive_drones_smoke bash scripts/run_server_simulation.sh
```

Write the status log to a custom path:

```bash
python -m defensive_drones.simulate --runs 100 --seed 42 --out outputs --log outputs/status_100.log
```

Run one strategy only:

```bash
python -m defensive_drones.simulate --runs 100 --seed 42 --out outputs_optimized --strategies optimized_global
```

Run multiple selected strategies:

```bash
python -m defensive_drones.simulate --runs 100 --seed 42 --out outputs_compare --strategies optimized optimized_global nearest_global
```

## Run Tests

```bash
python -m unittest
```

Or without activating the environment:

```bash
.venv/bin/python -m unittest
```

The tests cover deterministic scenario generation, wave activation, independent
attacker speeds, maneuvering velocity, autonomous duplicate assignments,
contention metrics, corridor/global/collaboration behavior, dwell-time kill
logic, target breach detection, and output artifact generation.

## Current Baseline Result

Regenerate the committed `outputs/` directory with:

```bash
.venv/bin/python -m defensive_drones.simulate --runs 1000 --seed 42 --out outputs --jobs 8
```

Then regenerate publication figures with:

```bash
.venv/bin/python scripts/generate_publication_figures.py --input outputs/per_run_results.csv --out outputs/publication_figures
```

The baseline now evaluates 2-3 waves of 250-350 attackers each, 10-100
defenders in steps of 10, and records success/kill, pass-through, per-wave, and
target-contention metrics for every strategy run.

## Project Layout

```text
defensive_drones/
  engine.py      simulation loop and physics updates
  geometry.py    vector math, corridor indexing, breach checks
  model.py       dataclasses and default configuration
  reporting.py   CSV and PNG output generation
  scenario.py    randomized scenario generation
  simulate.py    command line entry point
  strategies.py  autonomous target selection strategies
tests/
  test_simulation.py
outputs/
  generated CSV, PNG, and run status log results
```

## Limitations and Future Work

The simulation deliberately excludes battery limits, collision avoidance, real
flight dynamics, payload constraints, and visual simulation.

Useful next improvements:

- tune defender speed, acceleration, spawn distance, and dwell-time parameters
- add soft corridor handoff between neighboring defenders
- add layered defense corridors based on distance from target
- add richer summary tables for per-corridor load and kill timing
- add explicit communication-delay experiments as a comparison case
