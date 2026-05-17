from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Vector = np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    min_defenders: int = 2
    max_defenders: int = 5
    min_attackers: int = 5
    max_attackers: int = 20
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_cone_degrees: float = 12.0
    min_spawn_distance_m: float = 800.0
    max_spawn_distance_m: float = 1500.0
    min_attacker_speed_mps: float = 18.0
    max_attacker_speed_mps: float = 25.0
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
        )


@dataclass
class Defender:
    id: int
    position: Vector
    velocity: Vector
    corridor: int

    def copy(self) -> "Defender":
        return Defender(
            id=self.id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            corridor=self.corridor,
        )


@dataclass
class Scenario:
    seed: int
    defenders: list[Defender] = field(default_factory=list)
    attackers: list[Attacker] = field(default_factory=list)

    def copy(self) -> "Scenario":
        return Scenario(
            seed=self.seed,
            defenders=[defender.copy() for defender in self.defenders],
            attackers=[attacker.copy() for attacker in self.attackers],
        )


@dataclass(frozen=True)
class Observation:
    attacker_id: int
    position: Vector
    velocity: Vector
    corridor: int


@dataclass(frozen=True)
class RunResult:
    run_id: int
    scenario_seed: int
    strategy: str
    defender_count: int
    attacker_count: int
    kills: int
    breaches: int
    success: bool
    completion_time_s: float

    @property
    def kill_ratio(self) -> float:
        if self.attacker_count == 0:
            return 0.0
        return self.kills / self.attacker_count

    @property
    def attacker_bucket(self) -> str:
        if self.attacker_count <= 10:
            return "5-10"
        if self.attacker_count <= 15:
            return "11-15"
        return "16-20"
