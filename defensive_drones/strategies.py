from __future__ import annotations

import numpy as np

from defensive_drones.geometry import norm, time_to_target
from defensive_drones.model import Attacker, Defender, Observation, Scenario, SimulationConfig


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
    assignments: dict[int, int] = {}

    for defender in scenario.defenders:
        held_target = _held_target(
            defender,
            scenario,
            observations,
            previous_assignments,
            dwell_times,
            assignment_mode,
        )
        if held_target is not None:
            assignments[defender.id] = held_target
            continue

        candidates = _candidates_for_defender(
            defender, scenario, observations, assignment_mode
        )
        if not candidates:
            continue
        target = min(
            candidates,
            key=lambda attacker: (
                _score_candidate(
                    base_strategy,
                    assignment_mode,
                    defender,
                    attacker,
                    observations[attacker.id],
                    config,
                ),
                attacker.id,
            ),
        )
        assignments[defender.id] = target.id

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


def _held_target(
    defender: Defender,
    scenario: Scenario,
    observations: dict[int, Observation],
    previous_assignments: dict[int, int],
    dwell_times: np.ndarray,
    assignment_mode: str,
) -> int | None:
    attacker_id = previous_assignments.get(defender.id)
    if attacker_id is None or attacker_id not in observations:
        return None
    attacker = scenario.attackers[attacker_id]
    if not attacker.alive or not attacker.active:
        return None
    if assignment_mode == "corridor" and attacker.corridor != defender.corridor:
        return None

    is_dwelling = (
        defender.id < dwell_times.shape[0]
        and attacker_id < dwell_times.shape[1]
        and dwell_times[defender.id, attacker_id] > 0.0
    )
    is_near_capture = norm(attacker.position - defender.position) <= 20.0
    if is_dwelling or is_near_capture:
        return attacker_id
    return None


def _candidates_for_defender(
    defender: Defender,
    scenario: Scenario,
    observations: dict[int, Observation],
    assignment_mode: str,
) -> list[Attacker]:
    live_observed = [
        attacker
        for attacker in scenario.attackers
        if attacker.alive and attacker.active and attacker.id in observations
    ]
    if assignment_mode == "global":
        return live_observed

    own_corridor = [
        attacker for attacker in live_observed if attacker.corridor == defender.corridor
    ]
    if assignment_mode == "corridor":
        return own_corridor
    if own_corridor:
        return own_corridor
    return [attacker for attacker in live_observed if attacker.corridor != defender.corridor]


def _score_candidate(
    base_strategy: str,
    assignment_mode: str,
    defender: Defender,
    attacker: Attacker,
    observation: Observation,
    config: SimulationConfig,
) -> float:
    if base_strategy == "nearest":
        return norm(observation.position - defender.position)
    if base_strategy == "earliest_deadline":
        return time_to_target(
            observation.position,
            observation.velocity,
            config.target_vector,
        )

    rendezvous_s = norm(observation.position - defender.position) / max(
        config.defender_max_speed_mps, 1e-9
    )
    deadline_s = time_to_target(
        observation.position,
        observation.velocity,
        config.target_vector,
    )
    finish_s = rendezvous_s + config.kill_dwell_s
    slack_s = deadline_s - finish_s
    off_corridor_penalty = (
        COLLABORATION_PENALTY
        if assignment_mode == "collab" and attacker.corridor != defender.corridor
        else 0.0
    )
    if slack_s >= 0.0:
        return rendezvous_s + 0.01 * deadline_s + off_corridor_penalty
    return 10_000.0 + 500.0 * abs(slack_s) + rendezvous_s + off_corridor_penalty
