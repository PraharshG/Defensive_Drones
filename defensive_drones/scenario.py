from __future__ import annotations

import math

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

    wave_count = int(rng.integers(config.min_waves, config.max_waves + 1))

    defenders = [
        _make_defender(defender_id, defender_count, config)
        for defender_id in range(defender_count)
    ]
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
    )
