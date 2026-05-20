from __future__ import annotations

import numpy as np

from defensive_drones.geometry import norm, segment_reaches_sphere, time_to_target, unit
from defensive_drones.model import Observation, RunResult, Scenario, SimulationConfig
from defensive_drones.strategies import choose_assignments


def simulate_scenario(
    initial_scenario: Scenario,
    config: SimulationConfig,
    strategy: str,
    run_id: int,
    observation_seed: int,
) -> RunResult:
    scenario = initial_scenario.copy()
    rng = np.random.default_rng(observation_seed)
    dwell_times = np.zeros((len(scenario.defenders), len(scenario.attackers)), dtype=float)
    assignments: dict[int, int] = {}
    elapsed_s = 0.0

    while True:
        if all(not attacker.alive for attacker in scenario.attackers):
            return _result(run_id, scenario, strategy, elapsed_s, True)

        observations = observe_attackers(scenario, config, rng)
        assignments = choose_assignments(
            strategy, scenario, observations, config, assignments, dwell_times
        )

        previous_attacker_positions = {
            attacker.id: attacker.position.copy()
            for attacker in scenario.attackers
            if attacker.alive
        }

        _advance_defenders(scenario, observations, assignments, config)
        _advance_attackers(scenario, config)
        elapsed_s += config.dt_s

        update_dwell_and_kills(scenario, dwell_times, config)

        breached = _mark_breaches(
            scenario, previous_attacker_positions, config
        )
        if breached:
            return _result(run_id, scenario, strategy, elapsed_s, False)


def observe_attackers(
    scenario: Scenario,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> dict[int, Observation]:
    observations: dict[int, Observation] = {}
    for attacker in scenario.attackers:
        if not attacker.alive:
            continue
        observations[attacker.id] = Observation(
            attacker_id=attacker.id,
            position=attacker.position
            + rng.normal(0.0, config.sensor_position_sigma_m, size=3),
            velocity=attacker.velocity
            + rng.normal(0.0, config.sensor_velocity_sigma_mps, size=3),
            corridor=attacker.corridor,
        )
    return observations


def update_dwell_and_kills(
    scenario: Scenario,
    dwell_times: np.ndarray,
    config: SimulationConfig,
) -> list[int]:
    killed_ids: list[int] = []
    for defender in scenario.defenders:
        for attacker in scenario.attackers:
            if not attacker.alive:
                dwell_times[defender.id, attacker.id] = 0.0
                continue
            if norm(defender.position - attacker.position) <= config.kill_radius_m:
                dwell_times[defender.id, attacker.id] += config.dt_s
            else:
                dwell_times[defender.id, attacker.id] = 0.0

            if dwell_times[defender.id, attacker.id] >= config.kill_dwell_s:
                attacker.alive = False
                attacker.killed = True
                killed_ids.append(attacker.id)
                dwell_times[:, attacker.id] = 0.0
    return killed_ids


def _advance_defenders(
    scenario: Scenario,
    observations: dict[int, Observation],
    assignments: dict[int, int],
    config: SimulationConfig,
) -> None:
    target = config.target_vector
    for defender in scenario.defenders:
        attacker_id = assignments.get(defender.id)
        observation = observations.get(attacker_id) if attacker_id is not None else None
        desired_velocity = np.zeros(3, dtype=float)

        if observation is not None:
            offset = observation.position - defender.position
            distance = norm(offset)
            if distance <= 20.0:
                desired_velocity = observation.velocity + 0.8 * offset
            else:
                lead_s = distance / max(config.defender_max_speed_mps, 1e-9)
                lead_s = min(
                    lead_s,
                    max(0.0, time_to_target(observation.position, observation.velocity, target)),
                )
                aim_point = observation.position + observation.velocity * lead_s
                desired_velocity = unit(aim_point - defender.position) * (
                    config.defender_max_speed_mps
                )

            desired_speed = norm(desired_velocity)
            if desired_speed > config.defender_max_speed_mps:
                desired_velocity = (
                    desired_velocity / desired_speed * config.defender_max_speed_mps
                )

        _apply_velocity_command(defender, desired_velocity, config)


def _apply_velocity_command(defender, desired_velocity: np.ndarray, config: SimulationConfig) -> None:
    max_delta_v = config.defender_max_accel_mps2 * config.dt_s
    delta_v = desired_velocity - defender.velocity
    delta_v_norm = norm(delta_v)
    if delta_v_norm > max_delta_v:
        delta_v = delta_v / delta_v_norm * max_delta_v

    defender.velocity = defender.velocity + delta_v
    speed = norm(defender.velocity)
    if speed > config.defender_max_speed_mps:
        defender.velocity = defender.velocity / speed * config.defender_max_speed_mps
    defender.position = defender.position + defender.velocity * config.dt_s


def _advance_attackers(scenario: Scenario, config: SimulationConfig) -> None:
    for attacker in scenario.attackers:
        if attacker.alive:
            attacker.position = attacker.position + attacker.velocity * config.dt_s


def _mark_breaches(
    scenario: Scenario,
    previous_attacker_positions: dict[int, np.ndarray],
    config: SimulationConfig,
) -> bool:
    target = config.target_vector
    breached_any = False
    for attacker in scenario.attackers:
        if not attacker.alive:
            continue
        previous_position = previous_attacker_positions[attacker.id]
        if segment_reaches_sphere(
            previous_position, attacker.position, target, config.target_breach_radius_m
        ):
            attacker.alive = False
            attacker.breached = True
            breached_any = True
    return breached_any


def _result(
    run_id: int,
    scenario: Scenario,
    strategy: str,
    elapsed_s: float,
    success: bool,
) -> RunResult:
    kills = sum(1 for attacker in scenario.attackers if attacker.killed)
    breaches = sum(1 for attacker in scenario.attackers if attacker.breached)
    return RunResult(
        run_id=run_id,
        scenario_seed=scenario.seed,
        strategy=strategy,
        defender_count=len(scenario.defenders),
        attacker_count=len(scenario.attackers),
        kills=kills,
        breaches=breaches,
        success=success,
        completion_time_s=elapsed_s,
    )
