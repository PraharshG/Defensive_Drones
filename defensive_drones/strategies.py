from __future__ import annotations

import numpy as np

from defensive_drones.geometry import norm
from defensive_drones.model import Defender, Observation, Scenario, SimulationConfig


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
    observed = _observed_arrays(scenario, observations, config)
    if observed is None:
        return assignments

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

        target_id = _choose_target_for_defender(
            base_strategy,
            assignment_mode,
            defender,
            observed,
            config,
        )
        if target_id is None:
            continue
        assignments[defender.id] = target_id

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


class ObservedAttackers:
    def __init__(
        self,
        ids: np.ndarray,
        corridors: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        deadlines_s: np.ndarray,
    ) -> None:
        self.ids = ids
        self.corridors = corridors
        self.positions = positions
        self.velocities = velocities
        self.deadlines_s = deadlines_s


def _observed_arrays(
    scenario: Scenario,
    observations: dict[int, Observation],
    config: SimulationConfig,
) -> ObservedAttackers | None:
    attackers = [
        attacker
        for attacker in scenario.attackers
        if attacker.alive and attacker.active and attacker.id in observations
    ]
    if not attackers:
        return None

    ids = np.array([attacker.id for attacker in attackers], dtype=int)
    corridors = np.array([attacker.corridor for attacker in attackers], dtype=int)
    positions = np.array([observations[attacker.id].position for attacker in attackers])
    velocities = np.array([observations[attacker.id].velocity for attacker in attackers])
    speeds = np.linalg.norm(velocities, axis=1)
    target = config.target_vector
    with np.errstate(divide="ignore", invalid="ignore"):
        deadlines_s = np.linalg.norm(positions - target, axis=1) / speeds
    deadlines_s = np.where(np.isfinite(deadlines_s), deadlines_s, np.inf)
    return ObservedAttackers(ids, corridors, positions, velocities, deadlines_s)


def _choose_target_for_defender(
    base_strategy: str,
    assignment_mode: str,
    defender: Defender,
    observed: ObservedAttackers,
    config: SimulationConfig,
) -> int | None:
    if assignment_mode == "global":
        mask = np.ones(len(observed.ids), dtype=bool)
    else:
        own_corridor = observed.corridors == defender.corridor
        if assignment_mode == "corridor":
            mask = own_corridor
        elif np.any(own_corridor):
            mask = own_corridor
        else:
            mask = observed.corridors != defender.corridor

    if not np.any(mask):
        return None

    scores = _scores_for_defender(base_strategy, assignment_mode, defender, observed, config)
    masked_scores = np.where(mask, scores, np.inf)
    if not np.any(np.isfinite(masked_scores)):
        return None
    return int(observed.ids[int(np.argmin(masked_scores))])


def _scores_for_defender(
    base_strategy: str,
    assignment_mode: str,
    defender: Defender,
    observed: ObservedAttackers,
    config: SimulationConfig,
) -> np.ndarray:
    distances = np.linalg.norm(observed.positions - defender.position, axis=1)
    if base_strategy == "nearest":
        return distances
    if base_strategy == "earliest_deadline":
        return observed.deadlines_s

    rendezvous_s = distances / max(
        config.defender_max_speed_mps, 1e-9
    )
    slack_s = observed.deadlines_s - (rendezvous_s + config.kill_dwell_s)
    scores = np.where(
        slack_s >= 0.0,
        rendezvous_s + 0.01 * observed.deadlines_s,
        10_000.0 + 500.0 * np.abs(slack_s) + rendezvous_s,
    )
    if assignment_mode == "collab":
        scores = scores + np.where(
            observed.corridors != defender.corridor,
            COLLABORATION_PENALTY,
            0.0,
        )
    return np.where(np.isfinite(scores), scores, np.inf)
