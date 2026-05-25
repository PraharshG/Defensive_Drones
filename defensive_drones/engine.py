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
    first_breach_time_s: float | None = None

    while True:
        if all(not attacker.alive for attacker in scenario.attackers):
            success = all(not attacker.breached for attacker in scenario.attackers)
            return _result(
                run_id,
                scenario,
                strategy,
                elapsed_s,
                success,
                first_breach_time_s,
            )

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

        breached_ids = _mark_breaches(
            scenario, previous_attacker_positions, config
        )
        if breached_ids and first_breach_time_s is None:
            first_breach_time_s = elapsed_s


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
    if not scenario.defenders or not scenario.attackers:
        return []

    dead_ids = [attacker.id for attacker in scenario.attackers if not attacker.alive]
    if dead_ids:
        dwell_times[:, dead_ids] = 0.0

    live_attackers = [attacker for attacker in scenario.attackers if attacker.alive]
    if not live_attackers:
        return []

    defender_positions = np.array(
        [defender.position for defender in scenario.defenders], dtype=float
    )
    attacker_positions = np.array(
        [attacker.position for attacker in live_attackers], dtype=float
    )
    live_ids = np.array([attacker.id for attacker in live_attackers], dtype=int)
    distances = np.linalg.norm(
        defender_positions[:, np.newaxis, :] - attacker_positions[np.newaxis, :, :],
        axis=2,
    )
    live_dwell = dwell_times[:, live_ids]
    dwell_times[:, live_ids] = np.where(
        distances <= config.kill_radius_m,
        live_dwell + config.dt_s,
        0.0,
    )

    killed_ids = [
        int(attacker_id)
        for attacker_id in live_ids[
            np.any(dwell_times[:, live_ids] >= config.kill_dwell_s, axis=0)
        ]
    ]
    for attacker_id in killed_ids:
        scenario.attackers[attacker_id].alive = False
        scenario.attackers[attacker_id].killed = True
        dwell_times[:, attacker_id] = 0.0
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
) -> list[int]:
    target = config.target_vector
    breached_ids: list[int] = []
    for attacker in scenario.attackers:
        if not attacker.alive:
            continue
        previous_position = previous_attacker_positions[attacker.id]
        if segment_reaches_sphere(
            previous_position, attacker.position, target, config.target_breach_radius_m
        ):
            attacker.alive = False
            attacker.breached = True
            breached_ids.append(attacker.id)
    return breached_ids


def _result(
    run_id: int,
    scenario: Scenario,
    strategy: str,
    elapsed_s: float,
    success: bool,
    first_breach_time_s: float | None,
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
        first_breach_time_s=first_breach_time_s,
    )
