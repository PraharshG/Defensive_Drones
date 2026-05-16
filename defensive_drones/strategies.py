from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from defensive_drones.geometry import norm, time_to_target
from defensive_drones.model import Observation, Scenario, SimulationConfig


STRATEGIES = ("optimized", "nearest", "earliest_deadline")
INF_COST = 1e9


def choose_assignments(
    strategy: str,
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    previous_assignments: dict[int, int],
    dwell_times: np.ndarray,
) -> dict[int, int]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    assignments: dict[int, int] = {}
    held_defenders = _hold_near_capture_assignments(
        scenario, observations, previous_assignments, dwell_times, assignments
    )

    if strategy == "nearest":
        _assign_nearest(scenario, observations, assignments, held_defenders)
    elif strategy == "earliest_deadline":
        _assign_earliest_deadline(
            scenario, observations, config, assignments, held_defenders
        )
    else:
        _assign_optimized(scenario, observations, config, assignments, held_defenders)

    return assignments


def _hold_near_capture_assignments(
    scenario: Scenario,
    observations: dict[int, Observation],
    previous_assignments: dict[int, int],
    dwell_times: np.ndarray,
    assignments: dict[int, int],
) -> set[int]:
    held_defenders: set[int] = set()
    live_ids = {attacker.id for attacker in scenario.attackers if attacker.alive}

    for defender in scenario.defenders:
        attacker_id = previous_assignments.get(defender.id)
        if attacker_id not in live_ids or attacker_id not in observations:
            continue
        attacker = scenario.attackers[attacker_id]
        if attacker.corridor != defender.corridor:
            continue
        is_dwelling = dwell_times[defender.id, attacker_id] > 0.0
        is_near_capture = norm(attacker.position - defender.position) <= 20.0
        if not is_dwelling and not is_near_capture:
            continue
        assignments[defender.id] = attacker_id
        held_defenders.add(defender.id)

    return held_defenders


def _corridor_candidates(
    defender_id: int,
    scenario: Scenario,
    observations: dict[int, Observation],
) -> list[int]:
    defender = scenario.defenders[defender_id]
    return [
        attacker.id
        for attacker in scenario.attackers
        if attacker.alive
        and attacker.corridor == defender.corridor
        and attacker.id in observations
    ]


def _assign_nearest(
    scenario: Scenario,
    observations: dict[int, Observation],
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    assigned_attackers = set(assignments.values())
    for defender in scenario.defenders:
        if defender.id in held_defenders:
            continue
        candidates = [
            attacker_id
            for attacker_id in _corridor_candidates(defender.id, scenario, observations)
            if attacker_id not in assigned_attackers
        ]
        if not candidates:
            continue
        target_id = min(
            candidates,
            key=lambda attacker_id: norm(
                observations[attacker_id].position - defender.position
            ),
        )
        assignments[defender.id] = target_id
        assigned_attackers.add(target_id)


def _assign_earliest_deadline(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    assigned_attackers = set(assignments.values())
    target = config.target_vector
    for defender in scenario.defenders:
        if defender.id in held_defenders:
            continue
        candidates = [
            attacker_id
            for attacker_id in _corridor_candidates(defender.id, scenario, observations)
            if attacker_id not in assigned_attackers
        ]
        if not candidates:
            continue
        target_id = min(
            candidates,
            key=lambda attacker_id: time_to_target(
                observations[attacker_id].position,
                observations[attacker_id].velocity,
                target,
            ),
        )
        assignments[defender.id] = target_id
        assigned_attackers.add(target_id)


def _assign_optimized(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    free_defenders = [
        defender for defender in scenario.defenders if defender.id not in held_defenders
    ]
    assigned_attackers = set(assignments.values())
    free_attackers = [
        attacker
        for attacker in scenario.attackers
        if attacker.alive
        and attacker.id not in assigned_attackers
        and attacker.id in observations
    ]

    if not free_defenders or not free_attackers:
        return

    costs = np.full((len(free_defenders), len(free_attackers)), INF_COST, dtype=float)
    target = config.target_vector
    for row, defender in enumerate(free_defenders):
        for col, attacker in enumerate(free_attackers):
            if attacker.corridor != defender.corridor:
                continue
            observation = observations[attacker.id]
            rendezvous_s = norm(observation.position - defender.position) / max(
                config.defender_max_speed_mps, 1e-9
            )
            deadline_s = time_to_target(
                observation.position, observation.velocity, target
            )
            finish_s = rendezvous_s + config.kill_dwell_s
            slack_s = deadline_s - finish_s
            if slack_s >= 0.0:
                costs[row, col] = rendezvous_s + 0.01 * deadline_s
            else:
                costs[row, col] = (
                    10_000.0
                    + 500.0 * abs(slack_s)
                    + rendezvous_s
                )

    row_ind, col_ind = linear_sum_assignment(costs)
    for row, col in zip(row_ind, col_ind):
        if costs[row, col] >= INF_COST:
            continue
        assignments[free_defenders[row].id] = free_attackers[col].id
