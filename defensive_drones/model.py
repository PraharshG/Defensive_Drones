from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Vector = np.ndarray
MILE_TO_M = 1609.344


ATTACKER_BUCKET_RANGES = (
    (5, 10, "5-10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (250, 274, "250-274"),
    (275, 299, "275-299"),
    (300, 324, "300-324"),
    (325, 350, "325-350"),
    (500, 599, "500-599"),
    (600, 700, "600-700"),
    (750, 849, "750-849"),
    (850, 949, "850-949"),
    (950, 1050, "950-1050"),
)


@dataclass(frozen=True)
class SimulationConfig:
    min_defenders: int = 10
    max_defenders: int = 100
    defender_step: int = 10
    min_attackers: int = 250
    max_attackers: int = 350
    min_waves: int = 2
    max_waves: int = 3
    wave_spacing_s: float = 120.0
    defender_reinforcement_fraction: float = 0.5
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_cone_degrees: float = 180.0
    min_spawn_distance_m: float = 4.5 * MILE_TO_M
    max_spawn_distance_m: float = 5.5 * MILE_TO_M
    min_attacker_speed_mps: float = 18.0
    max_attacker_speed_mps: float = 25.0
    min_attacker_maneuver_amplitude_mps: float = 0.5
    max_attacker_maneuver_amplitude_mps: float = 4.0
    min_attacker_maneuver_frequency_rad_s: float = 0.02
    max_attacker_maneuver_frequency_rad_s: float = 0.08
    defender_ring_radius_m: float = 120.0
    defender_max_speed_mps: float = 45.0
    defender_max_accel_mps2: float = 15.0
    dt_s: float = 0.25
    sensor_position_sigma_m: float = 1.0
    sensor_velocity_sigma_mps: float = 0.5
    kill_radius_m: float = 5.0
    kill_dwell_s: float = 3.0
    target_breach_radius_m: float = 5.0

    @property
    def target_vector(self) -> Vector:
        return np.array(self.target, dtype=float)

    @property
    def defender_counts(self) -> tuple[int, ...]:
        if self.defender_step <= 0:
            raise ValueError("defender_step must be positive")
        return tuple(range(self.min_defenders, self.max_defenders + 1, self.defender_step))


@dataclass
class Attacker:
    id: int
    position: Vector
    velocity: Vector
    speed_mps: float
    corridor: int
    alive: bool = True
    killed: bool = False
    breached: bool = False
    wave_id: int = 0
    spawn_time_s: float = 0.0
    active: bool = True
    maneuver_amplitude_mps: float = 0.0
    maneuver_frequency_rad_s: float = 0.0
    maneuver_phase_rad: float = 0.0
    maneuver_secondary_phase_rad: float = 0.0
    killed_time_s: float | None = None
    breach_time_s: float | None = None
    target_point: Vector | None = None

    def copy(self) -> "Attacker":
        return Attacker(
            id=self.id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            speed_mps=self.speed_mps,
            corridor=self.corridor,
            alive=self.alive,
            killed=self.killed,
            breached=self.breached,
            wave_id=self.wave_id,
            spawn_time_s=self.spawn_time_s,
            active=self.active,
            maneuver_amplitude_mps=self.maneuver_amplitude_mps,
            maneuver_frequency_rad_s=self.maneuver_frequency_rad_s,
            maneuver_phase_rad=self.maneuver_phase_rad,
            maneuver_secondary_phase_rad=self.maneuver_secondary_phase_rad,
            killed_time_s=self.killed_time_s,
            breach_time_s=self.breach_time_s,
            target_point=(
                None if self.target_point is None else self.target_point.copy()
            ),
        )


@dataclass
class Defender:
    id: int
    position: Vector
    velocity: Vector
    corridor: int
    wave_id: int = 0
    spawn_time_s: float = 0.0
    active: bool = True

    def copy(self) -> "Defender":
        return Defender(
            id=self.id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            corridor=self.corridor,
            wave_id=self.wave_id,
            spawn_time_s=self.spawn_time_s,
            active=self.active,
        )


@dataclass
class Scenario:
    seed: int
    defenders: list[Defender] = field(default_factory=list)
    attackers: list[Attacker] = field(default_factory=list)
    initial_defender_count: int | None = None

    def copy(self) -> "Scenario":
        return Scenario(
            seed=self.seed,
            defenders=[defender.copy() for defender in self.defenders],
            attackers=[attacker.copy() for attacker in self.attackers],
            initial_defender_count=self.initial_defender_count,
        )


@dataclass(frozen=True)
class Observation:
    attacker_id: int
    position: Vector
    velocity: Vector
    corridor: int


@dataclass(frozen=True)
class WaveResult:
    run_id: int
    scenario_seed: int
    strategy: str
    defender_count: int
    wave_id: int
    spawn_time_s: float
    attacker_count: int
    kills: int
    breaches: int
    success: bool
    completion_time_s: float
    first_breach_time_s: float | None = None

    @property
    def kill_ratio(self) -> float:
        if self.attacker_count == 0:
            return 0.0
        return self.kills / self.attacker_count

    @property
    def breach_rate(self) -> float:
        if self.attacker_count == 0:
            return 0.0
        return self.breaches / self.attacker_count


@dataclass(frozen=True)
class RunResult:
    run_id: int
    scenario_seed: int
    strategy: str
    defender_count: int
    total_defender_count: int
    attacker_count: int
    wave_count: int
    kills: int
    breaches: int
    success: bool
    completion_time_s: float
    first_breach_time_s: float | None = None
    duplicate_target_assignments: int = 0
    total_assignments: int = 0
    max_simultaneous_defenders_on_target: int = 0
    wave_results: tuple[WaveResult, ...] = field(default_factory=tuple)

    @property
    def kill_ratio(self) -> float:
        if self.attacker_count == 0:
            return 0.0
        return self.kills / self.attacker_count

    @property
    def breach_rate(self) -> float:
        if self.attacker_count == 0:
            return 0.0
        return self.breaches / self.attacker_count

    @property
    def contention_rate(self) -> float:
        if self.total_assignments == 0:
            return 0.0
        return self.duplicate_target_assignments / self.total_assignments

    @property
    def attacker_bucket(self) -> str:
        return attacker_bucket_for_count(self.attacker_count)


def attacker_bucket_for_count(attacker_count: int) -> str:
    for lower, upper, label in ATTACKER_BUCKET_RANGES:
        if lower <= attacker_count <= upper:
            return label
    bucket_width = 25
    lower = (attacker_count // bucket_width) * bucket_width
    upper = lower + bucket_width - 1
    return f"{lower}-{upper}"


def attacker_bucket_sort_key(bucket: str) -> tuple[int, int]:
    lower, _, upper = bucket.partition("-")
    return int(lower), int(upper or lower)


def initial_defender_count_for_scenario(scenario: Scenario) -> int:
    if scenario.initial_defender_count is not None:
        return scenario.initial_defender_count
    initial_count = sum(1 for defender in scenario.defenders if defender.spawn_time_s <= 0.0)
    return initial_count if initial_count > 0 else len(scenario.defenders)
