from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from defensive_drones.geometry import norm, time_to_target
from defensive_drones.model import Observation, Scenario, SimulationConfig


STRATEGIES = (
    "optimized",
    "nearest",
    "earliest_deadline",
    "optimized_global",
    "nearest_global",
    "earliest_deadline_global",
    "optimized_collab",
    "nearest_collab",
    "earliest_deadline_collab",
)
INF_COST = 1e9
COLLABORATION_PENALTY = 2.0


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

    base_strategy = _base_strategy(strategy)
    assignment_mode = _assignment_mode(strategy)
    use_corridors = assignment_mode == "corridor"
    assignments: dict[int, int] = {}
    held_defenders = _hold_near_capture_assignments(
        scenario,
        observations,
        previous_assignments,
        dwell_times,
        assignments,
        assignment_mode,
    )

    if base_strategy == "nearest":
        if assignment_mode == "collab":
            _assign_nearest_collab(scenario, observations, assignments, held_defenders)
        else:
            _assign_nearest(
                scenario, observations, assignments, held_defenders, use_corridors
            )
    elif base_strategy == "earliest_deadline":
        if assignment_mode == "collab":
            _assign_earliest_deadline_collab(
                scenario, observations, config, assignments, held_defenders
            )
        else:
            _assign_earliest_deadline(
                scenario,
                observations,
                config,
                assignments,
                held_defenders,
                use_corridors,
            )
    else:
        if assignment_mode == "collab":
            _assign_optimized_collab(
                scenario, observations, config, assignments, held_defenders
            )
        else:
            _assign_optimized(
                scenario,
                observations,
                config,
                assignments,
                held_defenders,
                use_corridors,
            )

    return assignments


def _base_strategy(strategy: str) -> str:
    for suffix in ("_global", "_collab"):
        if strategy.endswith(suffix):
            return strategy.removesuffix(suffix)
    return strategy


def _assignment_mode(strategy: str) -> str:
    if strategy.endswith("_global"):
        return "global"
    if strategy.endswith("_collab"):
        return "collab"
    return "corridor"


def _hold_near_capture_assignments(
    scenario: Scenario,
    observations: dict[int, Observation],
    previous_assignments: dict[int, int],
    dwell_times: np.ndarray,
    assignments: dict[int, int],
    assignment_mode: str,
) -> set[int]:
    held_defenders: set[int] = set()
    live_ids = {attacker.id for attacker in scenario.attackers if attacker.alive}

    for defender in scenario.defenders:
        attacker_id = previous_assignments.get(defender.id)
        if attacker_id not in live_ids or attacker_id not in observations:
            continue
        attacker = scenario.attackers[attacker_id]
        if assignment_mode == "corridor" and attacker.corridor != defender.corridor:
            continue
        is_dwelling = dwell_times[defender.id, attacker_id] > 0.0
        is_near_capture = norm(attacker.position - defender.position) <= 20.0
        if not is_dwelling and not is_near_capture:
            continue
        assignments[defender.id] = attacker_id
        held_defenders.add(defender.id)

    return held_defenders


def _candidate_ids(
    defender_id: int,
    scenario: Scenario,
    observations: dict[int, Observation],
    use_corridors: bool,
) -> list[int]:
    defender = scenario.defenders[defender_id]
    return [
        attacker.id
        for attacker in scenario.attackers
        if attacker.alive
        and attacker.id in observations
        and (not use_corridors or attacker.corridor == defender.corridor)
    ]


def _assign_nearest(
    scenario: Scenario,
    observations: dict[int, Observation],
    assignments: dict[int, int],
    held_defenders: set[int],
    use_corridors: bool,
) -> None:
    assigned_attackers = set(assignments.values())
    for defender in scenario.defenders:
        if defender.id in held_defenders:
            continue
        candidates = [
            attacker_id
            for attacker_id in _candidate_ids(
                defender.id, scenario, observations, use_corridors
            )
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


def _assign_nearest_collab(
    scenario: Scenario,
    observations: dict[int, Observation],
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    _assign_nearest(
        scenario, observations, assignments, held_defenders, use_corridors=True
    )
    _assign_collab_assists(
        scenario,
        observations,
        assignments,
        held_defenders,
        target_key=lambda defender, attacker_id, overload: (
            -overload,
            norm(observations[attacker_id].position - defender.position),
        ),
    )


def _assign_earliest_deadline(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
    use_corridors: bool,
) -> None:
    assigned_attackers = set(assignments.values())
    target = config.target_vector
    for defender in scenario.defenders:
        if defender.id in held_defenders:
            continue
        candidates = [
            attacker_id
            for attacker_id in _candidate_ids(
                defender.id, scenario, observations, use_corridors
            )
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


def _assign_earliest_deadline_collab(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    target = config.target_vector
    _assign_earliest_deadline(
        scenario,
        observations,
        config,
        assignments,
        held_defenders,
        use_corridors=True,
    )
    _assign_collab_assists(
        scenario,
        observations,
        assignments,
        held_defenders,
        target_key=lambda defender, attacker_id, overload: (
            -overload,
            time_to_target(
                observations[attacker_id].position,
                observations[attacker_id].velocity,
                target,
            ),
            norm(observations[attacker_id].position - defender.position),
        ),
    )


def _assign_optimized(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
    use_corridors: bool,
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

    target = config.target_vector
    defender_positions = np.array([defender.position for defender in free_defenders])
    defender_corridors = np.array([defender.corridor for defender in free_defenders])
    attacker_positions = np.array(
        [observations[attacker.id].position for attacker in free_attackers]
    )
    attacker_velocities = np.array(
        [observations[attacker.id].velocity for attacker in free_attackers]
    )
    attacker_corridors = np.array([attacker.corridor for attacker in free_attackers])

    rendezvous_s = (
        np.linalg.norm(
            defender_positions[:, np.newaxis, :] - attacker_positions[np.newaxis, :, :],
            axis=2,
        )
        / max(config.defender_max_speed_mps, 1e-9)
    )
    attacker_speeds = np.linalg.norm(attacker_velocities, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        deadline_s = np.linalg.norm(attacker_positions - target, axis=1) / attacker_speeds
    slack_s = deadline_s[np.newaxis, :] - (rendezvous_s + config.kill_dwell_s)
    costs = np.where(
        slack_s >= 0.0,
        rendezvous_s + 0.01 * deadline_s[np.newaxis, :],
        10_000.0 + 500.0 * np.abs(slack_s) + rendezvous_s,
    )
    if use_corridors:
        costs = np.where(
            defender_corridors[:, np.newaxis] == attacker_corridors[np.newaxis, :],
            costs,
            INF_COST,
        )
    costs = np.where(np.isfinite(costs), costs, INF_COST)

    row_ind, col_ind = linear_sum_assignment(costs)
    for row, col in zip(row_ind, col_ind):
        if costs[row, col] >= INF_COST:
            continue
        assignments[free_defenders[row].id] = free_attackers[col].id


def _assign_optimized_collab(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    _assign_optimized(
        scenario,
        observations,
        config,
        assignments,
        held_defenders,
        use_corridors=True,
    )
    _assign_optimized_collab_assists(
        scenario, observations, config, assignments, held_defenders
    )


def _assign_collab_assists(
    scenario: Scenario,
    observations: dict[int, Observation],
    assignments: dict[int, int],
    held_defenders: set[int],
    target_key,
) -> None:
    assigned_attackers = set(assignments.values())
    for defender in scenario.defenders:
        if defender.id in held_defenders or defender.id in assignments:
            continue
        if _own_corridor_has_live_attackers(defender.id, scenario, observations):
            continue

        overloads = _remaining_attacker_counts_by_corridor(
            scenario, observations, assigned_attackers
        )
        candidates = [
            attacker.id
            for attacker in scenario.attackers
            if attacker.alive
            and attacker.id in observations
            and attacker.id not in assigned_attackers
            and attacker.corridor != defender.corridor
            and overloads.get(attacker.corridor, 0) > 0
        ]
        if not candidates:
            continue

        target_id = min(
            candidates,
            key=lambda attacker_id: target_key(
                defender, attacker_id, overloads[scenario.attackers[attacker_id].corridor]
            ),
        )
        assignments[defender.id] = target_id
        assigned_attackers.add(target_id)


def _assign_optimized_collab_assists(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
    assignments: dict[int, int],
    held_defenders: set[int],
) -> None:
    assigned_attackers = set(assignments.values())
    assist_defenders = [
        defender
        for defender in scenario.defenders
        if defender.id not in held_defenders
        and defender.id not in assignments
        and not _own_corridor_has_live_attackers(defender.id, scenario, observations)
    ]
    overloads = _remaining_attacker_counts_by_corridor(
        scenario, observations, assigned_attackers
    )
    assist_attackers = [
        attacker
        for attacker in scenario.attackers
        if attacker.alive
        and attacker.id not in assigned_attackers
        and attacker.id in observations
        and overloads.get(attacker.corridor, 0) > 0
    ]

    if not assist_defenders or not assist_attackers:
        return

    target = config.target_vector
    defender_positions = np.array([defender.position for defender in assist_defenders])
    defender_corridors = np.array([defender.corridor for defender in assist_defenders])
    attacker_positions = np.array(
        [observations[attacker.id].position for attacker in assist_attackers]
    )
    attacker_velocities = np.array(
        [observations[attacker.id].velocity for attacker in assist_attackers]
    )
    attacker_corridors = np.array([attacker.corridor for attacker in assist_attackers])
    overload_bonus = np.array(
        [0.25 * overloads.get(attacker.corridor, 0) for attacker in assist_attackers]
    )

    rendezvous_s = (
        np.linalg.norm(
            defender_positions[:, np.newaxis, :] - attacker_positions[np.newaxis, :, :],
            axis=2,
        )
        / max(config.defender_max_speed_mps, 1e-9)
    )
    attacker_speeds = np.linalg.norm(attacker_velocities, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        deadline_s = np.linalg.norm(attacker_positions - target, axis=1) / attacker_speeds
    slack_s = deadline_s[np.newaxis, :] - (rendezvous_s + config.kill_dwell_s)
    costs = np.where(
        slack_s >= 0.0,
        rendezvous_s
        + 0.01 * deadline_s[np.newaxis, :]
        + COLLABORATION_PENALTY
        - overload_bonus[np.newaxis, :],
        10_000.0
        + 500.0 * np.abs(slack_s)
        + rendezvous_s
        + COLLABORATION_PENALTY
        - overload_bonus[np.newaxis, :],
    )
    costs = np.where(
        defender_corridors[:, np.newaxis] != attacker_corridors[np.newaxis, :],
        costs,
        INF_COST,
    )
    costs = np.where(np.isfinite(costs), costs, INF_COST)

    row_ind, col_ind = linear_sum_assignment(costs)
    for row, col in zip(row_ind, col_ind):
        if costs[row, col] >= INF_COST:
            continue
        assignments[assist_defenders[row].id] = assist_attackers[col].id


def _own_corridor_has_live_attackers(
    defender_id: int,
    scenario: Scenario,
    observations: dict[int, Observation],
) -> bool:
    return bool(
        _candidate_ids(
            defender_id, scenario, observations, use_corridors=True
        )
    )


def _remaining_attacker_counts_by_corridor(
    scenario: Scenario,
    observations: dict[int, Observation],
    assigned_attackers: set[int],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for attacker in scenario.attackers:
        if not attacker.alive:
            continue
        if attacker.id in assigned_attackers or attacker.id not in observations:
            continue
        counts[attacker.corridor] = counts.get(attacker.corridor, 0) + 1
    return counts
