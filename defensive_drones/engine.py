from __future__ import annotations

from collections import Counter

import numpy as np

from defensive_drones.geometry import norm, segment_reaches_sphere, time_to_target, unit
from defensive_drones.model import (
    Observation,
    RunResult,
    Scenario,
    SimulationConfig,
    WaveResult,
)
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
    contention_tracker = AssignmentContentionTracker()

    while True:
        activate_attackers(scenario, elapsed_s)
        if all(not attacker.alive for attacker in scenario.attackers):
            success = all(not attacker.breached for attacker in scenario.attackers)
            return _result(
                run_id,
                scenario,
                strategy,
                elapsed_s,
                success,
                first_breach_time_s,
                contention_tracker,
            )
        if not _has_active_attackers(scenario):
            next_spawn_time_s = _next_spawn_time(scenario, elapsed_s)
            if next_spawn_time_s is None:
                raise RuntimeError("Scenario has unresolved attackers with no spawn time")
            elapsed_s = next_spawn_time_s
            assignments = {}
            for defender in scenario.defenders:
                defender.velocity = np.zeros(3, dtype=float)
            activate_attackers(scenario, elapsed_s)
            continue

        observations = observe_attackers(scenario, config, rng)
        assignments = choose_assignments(
            strategy, scenario, observations, config, assignments, dwell_times
        )
        contention_tracker.record(assignments)

        previous_attacker_positions = {
            attacker.id: attacker.position.copy()
            for attacker in scenario.attackers
            if attacker.alive and attacker.active
        }

        _advance_defenders(scenario, observations, assignments, config)
        _advance_attackers(scenario, config, elapsed_s)
        elapsed_s += config.dt_s

        update_dwell_and_kills(scenario, dwell_times, config, elapsed_s)

        breached_ids = _mark_breaches(
            scenario, previous_attacker_positions, config, elapsed_s
        )
        if breached_ids and first_breach_time_s is None:
            first_breach_time_s = elapsed_s


class AssignmentContentionTracker:
    def __init__(self) -> None:
        self.duplicate_target_assignments = 0
        self.total_assignments = 0
        self.max_simultaneous_defenders_on_target = 0

    def record(self, assignments: dict[int, int]) -> None:
        counts = Counter(assignments.values())
        self.total_assignments += len(assignments)
        self.duplicate_target_assignments += sum(
            count - 1 for count in counts.values() if count > 1
        )
        self.max_simultaneous_defenders_on_target = max(
            self.max_simultaneous_defenders_on_target,
            max(counts.values(), default=0),
        )


def activate_attackers(scenario: Scenario, elapsed_s: float) -> list[int]:
    activated: list[int] = []
    for attacker in scenario.attackers:
        if attacker.alive and not attacker.active and elapsed_s >= attacker.spawn_time_s:
            attacker.active = True
            activated.append(attacker.id)
    return activated


def observe_attackers(
    scenario: Scenario,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> dict[int, Observation]:
    observations: dict[int, Observation] = {}
    for attacker in scenario.attackers:
        if not attacker.alive or not attacker.active:
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
    elapsed_s: float | None = None,
) -> list[int]:
    if not scenario.defenders or not scenario.attackers:
        return []

    inactive_ids = [
        attacker.id
        for attacker in scenario.attackers
        if not attacker.alive or not attacker.active
    ]
    if inactive_ids:
        dwell_times[:, inactive_ids] = 0.0

    live_attackers = [
        attacker for attacker in scenario.attackers if attacker.alive and attacker.active
    ]
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
        scenario.attackers[attacker_id].active = False
        scenario.attackers[attacker_id].killed = True
        scenario.attackers[attacker_id].killed_time_s = elapsed_s
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


def _advance_attackers(
    scenario: Scenario,
    config: SimulationConfig,
    elapsed_s: float,
) -> None:
    update_attacker_velocities(scenario, config, elapsed_s)
    for attacker in scenario.attackers:
        if attacker.alive and attacker.active:
            attacker.position = attacker.position + attacker.velocity * config.dt_s


def update_attacker_velocities(
    scenario: Scenario,
    config: SimulationConfig,
    elapsed_s: float,
) -> None:
    target = config.target_vector
    for attacker in scenario.attackers:
        if not attacker.alive or not attacker.active:
            continue
        to_target = target - attacker.position
        distance = norm(to_target)
        if distance <= 1e-9:
            continue
        target_direction = to_target / distance
        lateral_a, lateral_b = _maneuver_basis(target_direction)
        time_since_spawn_s = max(0.0, elapsed_s - attacker.spawn_time_s)
        phase = (
            attacker.maneuver_phase_rad
            + attacker.maneuver_frequency_rad_s * time_since_spawn_s
        )
        secondary_phase = (
            attacker.maneuver_secondary_phase_rad
            + 0.73 * attacker.maneuver_frequency_rad_s * time_since_spawn_s
        )
        lateral_velocity = attacker.maneuver_amplitude_mps * (
            np.sin(phase) * lateral_a + np.cos(secondary_phase) * lateral_b
        )
        attacker.velocity = target_direction * attacker.speed_mps + lateral_velocity


def _maneuver_basis(target_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(target_direction, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    lateral_a = np.cross(target_direction, reference)
    lateral_a = lateral_a / max(norm(lateral_a), 1e-9)
    lateral_b = np.cross(target_direction, lateral_a)
    lateral_b = lateral_b / max(norm(lateral_b), 1e-9)
    return lateral_a, lateral_b


def _mark_breaches(
    scenario: Scenario,
    previous_attacker_positions: dict[int, np.ndarray],
    config: SimulationConfig,
    elapsed_s: float,
) -> list[int]:
    target = config.target_vector
    breached_ids: list[int] = []
    for attacker in scenario.attackers:
        if not attacker.alive or not attacker.active:
            continue
        previous_position = previous_attacker_positions[attacker.id]
        if segment_reaches_sphere(
            previous_position, attacker.position, target, config.target_breach_radius_m
        ):
            attacker.alive = False
            attacker.active = False
            attacker.breached = True
            attacker.breach_time_s = elapsed_s
            breached_ids.append(attacker.id)
    return breached_ids


def _result(
    run_id: int,
    scenario: Scenario,
    strategy: str,
    elapsed_s: float,
    success: bool,
    first_breach_time_s: float | None,
    contention_tracker: AssignmentContentionTracker,
) -> RunResult:
    kills = sum(1 for attacker in scenario.attackers if attacker.killed)
    breaches = sum(1 for attacker in scenario.attackers if attacker.breached)
    wave_results = _wave_results(run_id, scenario, strategy, elapsed_s)
    return RunResult(
        run_id=run_id,
        scenario_seed=scenario.seed,
        strategy=strategy,
        defender_count=len(scenario.defenders),
        attacker_count=len(scenario.attackers),
        wave_count=len(wave_results),
        kills=kills,
        breaches=breaches,
        success=success,
        completion_time_s=elapsed_s,
        first_breach_time_s=first_breach_time_s,
        duplicate_target_assignments=(
            contention_tracker.duplicate_target_assignments
        ),
        total_assignments=contention_tracker.total_assignments,
        max_simultaneous_defenders_on_target=(
            contention_tracker.max_simultaneous_defenders_on_target
        ),
        wave_results=wave_results,
    )


def _wave_results(
    run_id: int,
    scenario: Scenario,
    strategy: str,
    elapsed_s: float,
) -> tuple[WaveResult, ...]:
    wave_ids = sorted({attacker.wave_id for attacker in scenario.attackers})
    results: list[WaveResult] = []
    for wave_id in wave_ids:
        wave_attackers = [
            attacker for attacker in scenario.attackers if attacker.wave_id == wave_id
        ]
        if not wave_attackers:
            continue
        kills = sum(attacker.killed for attacker in wave_attackers)
        breaches = sum(attacker.breached for attacker in wave_attackers)
        spawn_time_s = min(attacker.spawn_time_s for attacker in wave_attackers)
        resolution_times = [
            time_s
            for attacker in wave_attackers
            for time_s in (attacker.killed_time_s, attacker.breach_time_s)
            if time_s is not None
        ]
        completion_time_s = (
            max(0.0, max(resolution_times) - spawn_time_s)
            if resolution_times
            else max(0.0, elapsed_s - spawn_time_s)
        )
        breach_times = [
            attacker.breach_time_s
            for attacker in wave_attackers
            if attacker.breach_time_s is not None
        ]
        first_breach_time_s = (
            max(0.0, min(breach_times) - spawn_time_s)
            if breach_times
            else None
        )
        results.append(
            WaveResult(
                run_id=run_id,
                scenario_seed=scenario.seed,
                strategy=strategy,
                defender_count=len(scenario.defenders),
                wave_id=wave_id,
                spawn_time_s=spawn_time_s,
                attacker_count=len(wave_attackers),
                kills=kills,
                breaches=breaches,
                success=breaches == 0,
                completion_time_s=completion_time_s,
                first_breach_time_s=first_breach_time_s,
            )
        )
    return tuple(results)


def _has_active_attackers(scenario: Scenario) -> bool:
    return any(attacker.alive and attacker.active for attacker in scenario.attackers)


def _next_spawn_time(scenario: Scenario, elapsed_s: float) -> float | None:
    spawn_times = [
        attacker.spawn_time_s
        for attacker in scenario.attackers
        if attacker.alive and not attacker.active and attacker.spawn_time_s > elapsed_s
    ]
    if not spawn_times:
        return None
    return min(spawn_times)
