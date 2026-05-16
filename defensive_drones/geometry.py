from __future__ import annotations

import math

import numpy as np

from defensive_drones.model import Vector


EPSILON = 1e-9


def norm(vector: Vector) -> float:
    return float(np.linalg.norm(vector))


def unit(vector: Vector) -> Vector:
    length = norm(vector)
    if length < EPSILON:
        return np.zeros_like(vector, dtype=float)
    return vector / length


def lateral_angle(position: Vector) -> float:
    """Angle around the approach axis, using the y-z plane."""
    angle = math.atan2(float(position[2]), float(position[1]))
    return angle % (2.0 * math.pi)


def corridor_index_for_position(position: Vector, defender_count: int) -> int:
    return corridor_index_for_angle(lateral_angle(position), defender_count)


def corridor_index_for_angle(angle: float, defender_count: int) -> int:
    if defender_count <= 0:
        raise ValueError("defender_count must be positive")
    sector_width = 2.0 * math.pi / defender_count
    return min(int((angle % (2.0 * math.pi)) / sector_width), defender_count - 1)


def time_to_target(position: Vector, velocity: Vector, target: Vector) -> float:
    speed = norm(velocity)
    if speed < EPSILON:
        return math.inf
    return norm(position - target) / speed


def estimate_intercept_time(
    defender_position: Vector,
    attacker_position: Vector,
    attacker_velocity: Vector,
    defender_max_speed: float,
) -> float:
    """Constant-speed intercept estimate for assignment scoring."""
    relative_position = attacker_position - defender_position
    a = float(np.dot(attacker_velocity, attacker_velocity) - defender_max_speed**2)
    b = float(2.0 * np.dot(relative_position, attacker_velocity))
    c = float(np.dot(relative_position, relative_position))

    if abs(a) < EPSILON:
        if abs(b) < EPSILON:
            return 0.0 if c < EPSILON else math.inf
        t = -c / b
        return t if t >= 0.0 else math.inf

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return math.inf

    root = math.sqrt(discriminant)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t >= 0.0]
    return min(candidates) if candidates else math.inf


def segment_reaches_sphere(
    start: Vector,
    end: Vector,
    center: Vector,
    radius: float,
) -> bool:
    """Return true if the segment from start to end touches a sphere."""
    if norm(start - center) <= radius or norm(end - center) <= radius:
        return True

    segment = end - start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq < EPSILON:
        return False

    t = float(np.dot(center - start, segment) / segment_len_sq)
    if t < 0.0 or t > 1.0:
        return False

    closest = start + t * segment
    return norm(closest - center) <= radius
