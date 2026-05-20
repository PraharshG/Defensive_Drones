from __future__ import annotations

import math

import numpy as np

from defensive_drones.geometry import corridor_index_for_position
from defensive_drones.model import Attacker, Defender, Scenario, SimulationConfig


def generate_scenario(seed: int, config: SimulationConfig | None = None) -> Scenario:
    config = config or SimulationConfig()
    rng = np.random.default_rng(seed)

    defender_count = int(rng.integers(config.min_defenders, config.max_defenders + 1))
    attacker_count = int(rng.integers(config.min_attackers, config.max_attackers + 1))

    defenders = [
        _make_defender(defender_id, defender_count, config)
        for defender_id in range(defender_count)
    ]
    attackers = [
        _make_attacker(attacker_id, defender_count, rng, config)
        for attacker_id in range(attacker_count)
    ]
    return Scenario(seed=seed, defenders=defenders, attackers=attackers)


def _make_defender(
    defender_id: int,
    defender_count: int,
    config: SimulationConfig,
) -> Defender:
    sector_width = 2.0 * math.pi / defender_count
    angle = (defender_id + 0.5) * sector_width
    position = np.array(
        [
            0.0,
            config.defender_ring_radius_m * math.cos(angle),
            config.defender_ring_radius_m * math.sin(angle),
        ],
        dtype=float,
    )
    return Defender(
        id=defender_id,
        position=position,
        velocity=np.zeros(3, dtype=float),
        corridor=defender_id,
    )


def _make_attacker(
    attacker_id: int,
    defender_count: int,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Attacker:
    cone_radians = math.radians(config.approach_cone_degrees)
    cos_theta = rng.uniform(math.cos(cone_radians), 1.0)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    distance = rng.uniform(config.min_spawn_distance_m, config.max_spawn_distance_m)

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
    velocity = -direction_from_target * speed
    return Attacker(
        id=attacker_id,
        position=position,
        velocity=velocity,
        speed_mps=speed,
        corridor=corridor_index_for_position(position, defender_count),
    )
