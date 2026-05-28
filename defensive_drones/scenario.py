from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from defensive_drones.geometry import corridor_index_for_position
from defensive_drones.model import Attacker, Defender, Scenario, SimulationConfig


def generate_scenario(
    seed: int,
    config: SimulationConfig | None = None,
    defender_count: int | None = None,
    attacker_count: int | None = None,
) -> Scenario:
    config = config or SimulationConfig()
    rng = np.random.default_rng(seed)

    if defender_count is None:
        defender_count = int(rng.choice(config.defender_counts))
    if config.min_waves < 1:
        raise ValueError("min_waves must be at least 1")
    if config.max_waves < config.min_waves:
        raise ValueError("max_waves must be greater than or equal to min_waves")
    if config.wave_spacing_s < 0.0:
        raise ValueError("wave_spacing_s must be non-negative")
    if config.defender_reinforcement_fraction < 0.0:
        raise ValueError("defender_reinforcement_fraction must be non-negative")
    if defender_count <= 0:
        raise ValueError("defender_count must be positive")
    if not 0.0 <= config.approach_cone_degrees <= 180.0:
        raise ValueError("approach_cone_degrees must be between 0 and 180")

    wave_count = int(rng.integers(config.min_waves, config.max_waves + 1))

    attackers: list[Attacker] = []
    attacker_id = 0
    for wave_id in range(wave_count):
        wave_attacker_count = (
            attacker_count
            if attacker_count is not None
            else int(rng.integers(config.min_attackers, config.max_attackers + 1))
        )
        spawn_time_s = wave_id * config.wave_spacing_s
        for _ in range(wave_attacker_count):
            attackers.append(
                _make_attacker(
                    attacker_id,
                    defender_count,
                    wave_id,
                    spawn_time_s,
                    rng,
                    config,
                )
            )
            attacker_id += 1

    wave_zero_counts = _corridor_counts(
        [attacker for attacker in attackers if attacker.wave_id == 0],
        defender_count,
    )
    wave_zero_allocation = allocate_defenders_by_density(
        defender_count,
        wave_zero_counts,
    )

    defenders: list[Defender] = []
    defender_id = 0
    for corridor in _corridor_sequence(wave_zero_allocation):
        defenders.append(
            _make_defender(
                defender_id,
                corridor,
                defender_count,
                wave_id=0,
                spawn_time_s=0.0,
                active=True,
                config=config,
            )
        )
        defender_id += 1

    reinforcement_count = math.ceil(
        defender_count * config.defender_reinforcement_fraction
    )
    for wave_id in range(1, wave_count):
        spawn_time_s = wave_id * config.wave_spacing_s
        for _ in range(reinforcement_count):
            defenders.append(
                _make_defender(
                    defender_id,
                    corridor=0,
                    corridor_count=defender_count,
                    wave_id=wave_id,
                    spawn_time_s=spawn_time_s,
                    active=False,
                    config=config,
                )
            )
            defender_id += 1

    return Scenario(
        seed=seed,
        defenders=defenders,
        attackers=attackers,
        initial_defender_count=defender_count,
    )


def _make_defender(
    defender_id: int,
    corridor: int,
    corridor_count: int,
    wave_id: int,
    spawn_time_s: float,
    active: bool,
    config: SimulationConfig,
) -> Defender:
    position = (
        defender_position_for_corridor(corridor, corridor_count, config)
        if active
        else np.zeros(3, dtype=float)
    )
    return Defender(
        id=defender_id,
        position=position,
        velocity=np.zeros(3, dtype=float),
        corridor=corridor,
        wave_id=wave_id,
        spawn_time_s=spawn_time_s,
        active=active,
    )


def _make_attacker(
    attacker_id: int,
    defender_count: int,
    wave_id: int,
    spawn_time_s: float,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Attacker:
    cone_radians = math.radians(config.approach_cone_degrees)
    cos_theta = rng.uniform(math.cos(cone_radians), 1.0)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    distance = rng.uniform(config.min_spawn_distance_m, config.max_spawn_distance_m)
    target_point = _sample_target_point(rng, config)

    direction_from_target = np.array(
        [
            cos_theta,
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
        ],
        dtype=float,
    )
    position = direction_from_target * distance
    speed = float(
        rng.uniform(config.min_attacker_speed_mps, config.max_attacker_speed_mps)
    )
    target_offset = target_point - position
    target_distance = float(np.linalg.norm(target_offset))
    velocity = (
        target_offset / target_distance * speed
        if target_distance > 1e-9
        else -direction_from_target * speed
    )
    maneuver_amplitude = float(
        rng.uniform(
            config.min_attacker_maneuver_amplitude_mps,
            config.max_attacker_maneuver_amplitude_mps,
        )
    )
    maneuver_frequency = float(
        rng.uniform(
            config.min_attacker_maneuver_frequency_rad_s,
            config.max_attacker_maneuver_frequency_rad_s,
        )
    )
    return Attacker(
        id=attacker_id,
        position=position,
        velocity=velocity,
        speed_mps=speed,
        corridor=corridor_index_for_position(position, defender_count),
        wave_id=wave_id,
        spawn_time_s=spawn_time_s,
        active=spawn_time_s <= 0.0,
        maneuver_amplitude_mps=maneuver_amplitude,
        maneuver_frequency_rad_s=maneuver_frequency,
        maneuver_phase_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
        maneuver_secondary_phase_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
        target_point=target_point,
    )


def allocate_defenders_by_density(
    defender_total: int,
    attacker_counts: Sequence[int],
) -> list[int]:
    corridor_count = len(attacker_counts)
    if corridor_count == 0:
        return []
    if defender_total <= 0:
        return [0] * corridor_count

    counts = np.array(attacker_counts, dtype=float)
    nonempty = np.flatnonzero(counts > 0.0)
    if len(nonempty) == 0:
        base = defender_total // corridor_count
        allocation = [base] * corridor_count
        for corridor in range(defender_total % corridor_count):
            allocation[corridor] += 1
        return allocation

    allocation = np.zeros(corridor_count, dtype=int)
    if len(nonempty) >= defender_total:
        ranked = sorted(nonempty, key=lambda corridor: (-counts[corridor], corridor))
        for corridor in ranked[:defender_total]:
            allocation[int(corridor)] = 1
        return allocation.tolist()

    allocation[nonempty] = 1
    remaining = defender_total - len(nonempty)
    weights = counts[nonempty]
    quotas = remaining * weights / float(np.sum(weights))
    floors = np.floor(quotas).astype(int)
    allocation[nonempty] += floors

    leftover = remaining - int(np.sum(floors))
    ranked_remainders = sorted(
        range(len(nonempty)),
        key=lambda index: (
            -(quotas[index] - floors[index]),
            -weights[index],
            int(nonempty[index]),
        ),
    )
    for index in ranked_remainders[:leftover]:
        allocation[int(nonempty[index])] += 1

    return allocation.tolist()


def defender_position_for_corridor(
    corridor: int,
    corridor_count: int,
    config: SimulationConfig,
) -> np.ndarray:
    sector_width = 2.0 * math.pi / corridor_count
    angle = (corridor + 0.5) * sector_width
    return np.array(
        [
            0.0,
            config.defender_ring_radius_m * math.cos(angle),
            config.defender_ring_radius_m * math.sin(angle),
        ],
        dtype=float,
    )


def _corridor_counts(attackers: Sequence[Attacker], corridor_count: int) -> list[int]:
    counts = [0] * corridor_count
    for attacker in attackers:
        counts[attacker.corridor] += 1
    return counts


def _corridor_sequence(allocation: Sequence[int]) -> list[int]:
    return [
        corridor
        for corridor, count in enumerate(allocation)
        for _ in range(count)
    ]


def _sample_target_point(
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    center = config.target_vector
    radius = config.target_breach_radius_m
    if radius <= 0.0:
        return center.copy()

    direction = rng.normal(0.0, 1.0, size=3)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        direction = direction / length
    distance = radius * float(rng.random()) ** (1.0 / 3.0)
    return center + direction * distance
