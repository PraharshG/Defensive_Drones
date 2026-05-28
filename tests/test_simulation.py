from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from defensive_drones.engine import (
    activate_defenders,
    activate_attackers,
    observe_attackers,
    rebalance_defender_corridors,
    simulate_scenario,
    update_attacker_velocities,
    update_dwell_and_kills,
)
from defensive_drones.geometry import corridor_index_for_position, norm, segment_reaches_sphere
from defensive_drones.model import (
    Attacker,
    Defender,
    MILE_TO_M,
    Observation,
    Scenario,
    SimulationConfig,
)
from defensive_drones.scenario import generate_scenario
from defensive_drones.simulate import run_monte_carlo
from defensive_drones.strategies import STRATEGIES, choose_assignments


class ScenarioGenerationTests(unittest.TestCase):
    def test_fixed_seed_is_repeatable(self) -> None:
        config = SimulationConfig(
            min_defenders=3,
            max_defenders=3,
            min_attackers=4,
            max_attackers=4,
            min_waves=2,
            max_waves=2,
        )
        first = generate_scenario(1234, config)
        second = generate_scenario(1234, config)

        self.assertEqual(len(first.defenders), len(second.defenders))
        self.assertEqual(len(first.attackers), len(second.attackers))
        for left, right in zip(first.attackers, second.attackers):
            np.testing.assert_allclose(left.position, right.position)
            np.testing.assert_allclose(left.velocity, right.velocity)
            self.assertEqual(left.speed_mps, right.speed_mps)
            self.assertEqual(left.corridor, right.corridor)
            self.assertEqual(left.wave_id, right.wave_id)
            self.assertEqual(left.spawn_time_s, right.spawn_time_s)
            self.assertEqual(left.active, right.active)

    def test_attackers_have_independent_sampled_speeds(self) -> None:
        config = SimulationConfig(
            min_attackers=12,
            max_attackers=12,
            min_waves=1,
            max_waves=1,
        )
        scenario = generate_scenario(99, config)
        speeds = {round(attacker.speed_mps, 6) for attacker in scenario.attackers}
        self.assertGreater(len(speeds), 1)

    def test_corridor_assignment_uses_lateral_angle(self) -> None:
        self.assertEqual(corridor_index_for_position(np.array([1.0, 1.0, 0.0]), 4), 0)
        self.assertEqual(corridor_index_for_position(np.array([1.0, 0.0, 1.0]), 4), 1)
        self.assertEqual(corridor_index_for_position(np.array([1.0, -1.0, 0.0]), 4), 2)
        self.assertEqual(corridor_index_for_position(np.array([1.0, 0.0, -1.0]), 4), 3)

    def test_default_attackers_are_high_load_and_spawn_near_five_miles(self) -> None:
        scenario = generate_scenario(2026, SimulationConfig(), defender_count=10)
        wave_ids = sorted({attacker.wave_id for attacker in scenario.attackers})

        self.assertIn(len(wave_ids), {2, 3})
        for wave_id in wave_ids:
            wave_attackers = [
                attacker for attacker in scenario.attackers if attacker.wave_id == wave_id
            ]
            self.assertGreaterEqual(len(wave_attackers), 250)
            self.assertLessEqual(len(wave_attackers), 350)
            self.assertEqual(
                {attacker.spawn_time_s for attacker in wave_attackers},
                {wave_id * SimulationConfig().wave_spacing_s},
            )
        for attacker in scenario.attackers:
            distance = norm(attacker.position)
            self.assertGreaterEqual(distance, 4.5 * MILE_TO_M)
            self.assertLessEqual(distance, 5.5 * MILE_TO_M)
        self.assertTrue(all(attacker.active for attacker in scenario.attackers if attacker.wave_id == 0))
        self.assertTrue(
            all(not attacker.active for attacker in scenario.attackers if attacker.wave_id > 0)
        )

    def test_full_sphere_spawn_spreads_attackers_across_corridors(self) -> None:
        config = SimulationConfig(
            min_attackers=350,
            max_attackers=350,
            min_waves=1,
            max_waves=1,
        )
        scenario = generate_scenario(2027, config, defender_count=10)

        self.assertTrue(any(attacker.position[0] < 0.0 for attacker in scenario.attackers))
        self.assertTrue(any(attacker.position[0] > 0.0 for attacker in scenario.attackers))
        self.assertGreaterEqual(
            len({attacker.corridor for attacker in scenario.attackers}),
            8,
        )

    def test_attackers_target_random_points_inside_breach_radius(self) -> None:
        config = SimulationConfig(
            min_attackers=50,
            max_attackers=50,
            min_waves=1,
            max_waves=1,
            target_breach_radius_m=25.0,
        )
        scenario = generate_scenario(2031, config, defender_count=5)
        target = config.target_vector

        for attacker in scenario.attackers:
            self.assertIsNotNone(attacker.target_point)
            self.assertLessEqual(norm(attacker.target_point - target), 25.0)

    def test_defender_reinforcements_activate_with_their_wave(self) -> None:
        config = SimulationConfig(
            min_attackers=1,
            max_attackers=1,
            min_waves=3,
            max_waves=3,
            wave_spacing_s=10.0,
            defender_reinforcement_fraction=0.5,
        )
        scenario = generate_scenario(2032, config, defender_count=4)

        self.assertEqual(len(scenario.defenders), 8)
        self.assertEqual(
            [defender.id for defender in scenario.defenders if defender.active],
            [0, 1, 2, 3],
        )
        self.assertEqual(activate_defenders(scenario, 9.99), [])
        self.assertEqual(activate_defenders(scenario, 10.0), [4, 5])
        self.assertTrue(scenario.defenders[4].active)
        self.assertTrue(scenario.defenders[5].active)

    def test_density_rebalance_assigns_more_defenders_to_dense_corridors(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=2033,
            initial_defender_count=4,
            defenders=[
                Defender(index, np.zeros(3), np.zeros(3), index)
                for index in range(4)
            ],
            attackers=[
                Attacker(
                    id=index,
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    speed_mps=0.0,
                    corridor=0 if index < 5 else 1,
                )
                for index in range(6)
            ],
        )

        rebalance_defender_corridors(scenario, config)

        defender_counts = Counter(defender.corridor for defender in scenario.defenders)
        self.assertEqual(defender_counts[0], 3)
        self.assertEqual(defender_counts[1], 1)
        self.assertEqual(defender_counts[2], 0)
        self.assertEqual(defender_counts[3], 0)

    def test_same_seed_reuses_attacker_swarm_across_defender_counts(self) -> None:
        config = SimulationConfig(
            min_attackers=25,
            max_attackers=25,
            min_waves=2,
            max_waves=2,
        )
        scenario_10 = generate_scenario(2028, config, defender_count=10)
        scenario_20 = generate_scenario(2028, config, defender_count=20)

        for attacker_10, attacker_20 in zip(
            scenario_10.attackers, scenario_20.attackers
        ):
            np.testing.assert_allclose(attacker_10.position, attacker_20.position)
            np.testing.assert_allclose(attacker_10.velocity, attacker_20.velocity)
            self.assertEqual(attacker_10.speed_mps, attacker_20.speed_mps)
            self.assertEqual(attacker_10.wave_id, attacker_20.wave_id)
            self.assertEqual(attacker_10.spawn_time_s, attacker_20.spawn_time_s)

    def test_future_waves_are_not_observed_or_targetable_before_spawn(self) -> None:
        config = SimulationConfig(
            min_defenders=1,
            max_defenders=1,
            min_attackers=2,
            max_attackers=2,
            min_waves=2,
            max_waves=2,
            wave_spacing_s=10.0,
            sensor_position_sigma_m=0.0,
            sensor_velocity_sigma_mps=0.0,
        )
        scenario = generate_scenario(2029, config, defender_count=1)
        observations = observe_attackers(scenario, config, np.random.default_rng(1))
        dwell = np.zeros((1, len(scenario.attackers)), dtype=float)

        assignments = choose_assignments(
            "nearest_global", scenario, observations, config, {}, dwell
        )

        self.assertEqual({scenario.attackers[target].wave_id for target in assignments.values()}, {0})
        self.assertEqual({attacker.wave_id for attacker in scenario.attackers if attacker.active}, {0})

    def test_future_wave_activates_at_spawn_time(self) -> None:
        config = SimulationConfig(
            min_attackers=1,
            max_attackers=1,
            min_waves=2,
            max_waves=2,
            wave_spacing_s=10.0,
        )
        scenario = generate_scenario(2030, config, defender_count=1)

        self.assertEqual(activate_attackers(scenario, 9.99), [])
        self.assertFalse(scenario.attackers[1].active)
        self.assertEqual(activate_attackers(scenario, 10.0), [1])
        self.assertTrue(scenario.attackers[1].active)


class SimulationMechanicsTests(unittest.TestCase):
    def test_kill_requires_three_continuous_seconds_inside_radius(self) -> None:
        config = replace(
            SimulationConfig(),
            dt_s=0.25,
            kill_radius_m=5.0,
            kill_dwell_s=3.0,
        )
        scenario = Scenario(
            seed=1,
            defenders=[
                Defender(
                    id=0,
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    corridor=0,
                )
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([4.0, 0.0, 0.0]),
                    velocity=np.zeros(3),
                    speed_mps=0.0,
                    corridor=0,
                )
            ],
        )
        dwell = np.zeros((1, 1), dtype=float)

        for _ in range(11):
            self.assertEqual(update_dwell_and_kills(scenario, dwell, config), [])
            self.assertTrue(scenario.attackers[0].alive)

        self.assertEqual(update_dwell_and_kills(scenario, dwell, config), [0])
        self.assertFalse(scenario.attackers[0].alive)
        self.assertTrue(scenario.attackers[0].killed)

    def test_dwell_resets_when_attacker_leaves_radius(self) -> None:
        config = replace(SimulationConfig(), dt_s=1.0, kill_dwell_s=3.0)
        scenario = Scenario(
            seed=1,
            defenders=[Defender(0, np.zeros(3), np.zeros(3), 0)],
            attackers=[Attacker(0, np.array([4.0, 0.0, 0.0]), np.zeros(3), 0.0, 0)],
        )
        dwell = np.zeros((1, 1), dtype=float)

        update_dwell_and_kills(scenario, dwell, config)
        scenario.attackers[0].position = np.array([10.0, 0.0, 0.0])
        update_dwell_and_kills(scenario, dwell, config)
        scenario.attackers[0].position = np.array([4.0, 0.0, 0.0])
        update_dwell_and_kills(scenario, dwell, config)

        self.assertTrue(scenario.attackers[0].alive)
        self.assertEqual(dwell[0, 0], 1.0)

    def test_segment_breach_detection(self) -> None:
        target = np.zeros(3)
        self.assertTrue(
            segment_reaches_sphere(
                np.array([10.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                target,
                5.0,
            )
        )
        self.assertFalse(
            segment_reaches_sphere(
                np.array([10.0, 10.0, 0.0]),
                np.array([6.0, 10.0, 0.0]),
                target,
                5.0,
            )
        )

    def test_attacker_velocity_maneuvers_in_three_dimensions(self) -> None:
        config = replace(
            SimulationConfig(),
            target=(0.0, 0.0, 0.0),
        )
        scenario = Scenario(
            seed=9,
            defenders=[],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([100.0, 40.0, 30.0]),
                    velocity=np.zeros(3),
                    speed_mps=20.0,
                    corridor=0,
                    maneuver_amplitude_mps=3.0,
                    maneuver_frequency_rad_s=0.4,
                    maneuver_phase_rad=0.3,
                    maneuver_secondary_phase_rad=1.1,
                )
            ],
        )

        update_attacker_velocities(scenario, config, elapsed_s=0.0)
        first_velocity = scenario.attackers[0].velocity.copy()
        update_attacker_velocities(scenario, config, elapsed_s=5.0)
        second_velocity = scenario.attackers[0].velocity.copy()
        target_direction = -scenario.attackers[0].position / norm(
            scenario.attackers[0].position
        )

        self.assertTrue(np.all(np.abs(first_velocity - second_velocity) > 1e-6))
        self.assertGreater(np.dot(second_velocity, target_direction), 0.0)

    def test_simulation_continues_after_first_breach(self) -> None:
        config = replace(
            SimulationConfig(),
            dt_s=1.0,
            target_breach_radius_m=0.1,
            sensor_position_sigma_m=0.0,
            sensor_velocity_sigma_mps=0.0,
        )
        scenario = Scenario(
            seed=8,
            defenders=[],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([1.0, 0.0, 0.0]),
                    velocity=np.array([-1.0, 0.0, 0.0]),
                    speed_mps=1.0,
                    corridor=0,
                ),
                Attacker(
                    id=1,
                    position=np.array([3.0, 0.0, 0.0]),
                    velocity=np.array([-1.0, 0.0, 0.0]),
                    speed_mps=1.0,
                    corridor=0,
                ),
            ],
        )

        result = simulate_scenario(
            scenario,
            config,
            strategy="optimized",
            run_id=0,
            observation_seed=1,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.breaches, 2)
        self.assertEqual(result.kills, 0)
        self.assertEqual(result.first_breach_time_s, 1.0)
        self.assertEqual(result.completion_time_s, 3.0)
        self.assertEqual(result.breach_rate, 1.0)

    def test_contention_metrics_record_duplicate_assignments(self) -> None:
        config = replace(
            SimulationConfig(),
            dt_s=1.0,
            kill_dwell_s=1.0,
            sensor_position_sigma_m=0.0,
            sensor_velocity_sigma_mps=0.0,
        )
        scenario = Scenario(
            seed=10,
            defenders=[
                Defender(0, np.zeros(3), np.zeros(3), 0),
                Defender(1, np.zeros(3), np.zeros(3), 0),
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([4.0, 0.0, 0.0]),
                    velocity=np.zeros(3),
                    speed_mps=0.0,
                    corridor=0,
                )
            ],
        )

        result = simulate_scenario(
            scenario,
            config,
            strategy="nearest",
            run_id=0,
            observation_seed=1,
        )

        self.assertEqual(result.duplicate_target_assignments, 1)
        self.assertEqual(result.total_assignments, 2)
        self.assertEqual(result.contention_rate, 0.5)
        self.assertEqual(result.max_simultaneous_defenders_on_target, 2)

    def test_inactive_defenders_are_not_assigned_or_counted_for_kills(self) -> None:
        config = replace(
            SimulationConfig(),
            dt_s=1.0,
            kill_radius_m=5.0,
            kill_dwell_s=1.0,
            sensor_position_sigma_m=0.0,
            sensor_velocity_sigma_mps=0.0,
        )
        scenario = Scenario(
            seed=11,
            initial_defender_count=2,
            defenders=[
                Defender(
                    id=0,
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    corridor=0,
                    wave_id=1,
                    spawn_time_s=10.0,
                    active=False,
                ),
                Defender(
                    id=1,
                    position=np.array([100.0, 0.0, 0.0]),
                    velocity=np.zeros(3),
                    corridor=0,
                ),
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    speed_mps=0.0,
                    corridor=0,
                )
            ],
        )
        observations = _observations_for(scenario)
        dwell = np.zeros((2, 1), dtype=float)

        assignment = choose_assignments(
            "nearest", scenario, observations, config, {}, dwell
        )
        killed_ids = update_dwell_and_kills(scenario, dwell, config)

        self.assertNotIn(0, assignment)
        self.assertEqual(assignment, {1: 0})
        self.assertEqual(killed_ids, [])
        self.assertTrue(scenario.attackers[0].alive)
        self.assertEqual(dwell[0, 0], 0.0)


class StrategyTests(unittest.TestCase):
    def test_global_strategy_can_assign_outside_defender_corridor(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=2,
            defenders=[
                Defender(
                    id=0,
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    corridor=0,
                )
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([100.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=0,
                ),
                Attacker(
                    id=1,
                    position=np.array([10.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=1,
                ),
            ],
        )
        observations = {
            attacker.id: Observation(
                attacker.id,
                attacker.position.copy(),
                attacker.velocity.copy(),
                attacker.corridor,
            )
            for attacker in scenario.attackers
        }
        dwell = np.zeros((1, 2), dtype=float)

        corridor_assignment = choose_assignments(
            "nearest", scenario, observations, config, {}, dwell
        )
        global_assignment = choose_assignments(
            "nearest_global", scenario, observations, config, {}, dwell
        )

        self.assertEqual(corridor_assignment, {0: 0})
        self.assertEqual(global_assignment, {0: 1})

    def test_global_near_capture_lock_ignores_corridor(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=3,
            defenders=[
                Defender(
                    id=0,
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    corridor=0,
                )
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([4.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=1,
                )
            ],
        )
        observations = {
            0: Observation(
                0,
                scenario.attackers[0].position.copy(),
                scenario.attackers[0].velocity.copy(),
                scenario.attackers[0].corridor,
            )
        }
        dwell = np.array([[1.0]], dtype=float)

        corridor_assignment = choose_assignments(
            "optimized", scenario, observations, config, {0: 0}, dwell
        )
        global_assignment = choose_assignments(
            "optimized_global", scenario, observations, config, {0: 0}, dwell
        )

        self.assertEqual(corridor_assignment, {})
        self.assertEqual(global_assignment, {0: 0})

    def test_collab_defender_keeps_own_corridor_when_targets_exist(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=4,
            defenders=[
                Defender(0, np.zeros(3), np.zeros(3), 0),
                Defender(1, np.zeros(3), np.zeros(3), 1),
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([100.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=0,
                ),
                Attacker(
                    id=1,
                    position=np.array([10.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=1,
                ),
            ],
        )
        observations = _observations_for(scenario)
        dwell = np.zeros((2, 2), dtype=float)

        assignment = choose_assignments(
            "nearest_collab", scenario, observations, config, {}, dwell
        )

        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[1], 1)

    def test_collab_idle_defender_assists_overloaded_corridor(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=5,
            defenders=[
                Defender(0, np.zeros(3), np.zeros(3), 0),
                Defender(1, np.zeros(3), np.zeros(3), 1),
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([20.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=1,
                ),
                Attacker(
                    id=1,
                    position=np.array([30.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=1,
                ),
            ],
        )
        observations = _observations_for(scenario)
        dwell = np.zeros((2, 2), dtype=float)

        assignment = choose_assignments(
            "nearest_collab", scenario, observations, config, {}, dwell
        )

        self.assertEqual(set(assignment), {0, 1})
        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[1], 0)

    def test_autonomous_defenders_can_duplicate_target_assignments(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=7,
            defenders=[
                Defender(0, np.zeros(3), np.zeros(3), 0),
                Defender(1, np.zeros(3), np.zeros(3), 0),
            ],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([20.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=0,
                ),
                Attacker(
                    id=1,
                    position=np.array([30.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=0,
                ),
            ],
        )
        observations = _observations_for(scenario)
        dwell = np.zeros((2, 2), dtype=float)

        assignment = choose_assignments(
            "nearest", scenario, observations, config, {}, dwell
        )

        self.assertEqual(assignment, {0: 0, 1: 0})

    def test_collab_near_capture_lock_can_hold_assisted_target(self) -> None:
        config = SimulationConfig()
        scenario = Scenario(
            seed=6,
            defenders=[Defender(0, np.zeros(3), np.zeros(3), 0)],
            attackers=[
                Attacker(
                    id=0,
                    position=np.array([4.0, 0.0, 0.0]),
                    velocity=np.array([-20.0, 0.0, 0.0]),
                    speed_mps=20.0,
                    corridor=1,
                )
            ],
        )
        observations = _observations_for(scenario)
        dwell = np.array([[1.0]], dtype=float)

        assignment = choose_assignments(
            "optimized_collab", scenario, observations, config, {0: 0}, dwell
        )

        self.assertEqual(assignment, {0: 0})


class OutputSmokeTests(unittest.TestCase):
    def test_run_monte_carlo_sweeps_defender_counts(self) -> None:
        config = SimulationConfig(
            min_defenders=10,
            max_defenders=30,
            defender_step=10,
            min_attackers=3,
            max_attackers=3,
            min_waves=1,
            max_waves=1,
            min_spawn_distance_m=20.0,
            max_spawn_distance_m=20.0,
            min_attacker_speed_mps=20.0,
            max_attacker_speed_mps=20.0,
            dt_s=1.0,
            sensor_position_sigma_m=0.0,
            sensor_velocity_sigma_mps=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            results = run_monte_carlo(
                runs=2,
                seed=7,
                out_dir=Path(directory),
                config=config,
                strategies=("optimized_global",),
            )

        self.assertEqual(len(results), 2 * 3)
        self.assertEqual({result.defender_count for result in results}, {10, 20, 30})

    def test_smoke_run_writes_csv_and_png_outputs(self) -> None:
        config = SimulationConfig(
            min_defenders=2,
            max_defenders=2,
            min_attackers=5,
            max_attackers=5,
            min_waves=1,
            max_waves=1,
            min_spawn_distance_m=30.0,
            max_spawn_distance_m=30.0,
            min_attacker_speed_mps=20.0,
            max_attacker_speed_mps=20.0,
            dt_s=1.0,
            sensor_position_sigma_m=0.0,
            sensor_velocity_sigma_mps=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            results = run_monte_carlo(runs=5, seed=7, out_dir=out_dir, config=config)

            self.assertEqual(len(results), 5 * len(STRATEGIES))
            per_run = out_dir / "per_run_results.csv"
            wave_summary = out_dir / "wave_summary.csv"
            summary = out_dir / "summary.csv"
            success_matrix = out_dir / "success_rate_matrix.csv"
            breach_matrix = out_dir / "breach_rate_matrix.csv"
            status_log = out_dir / "run_status.log"
            self.assertTrue(per_run.exists())
            self.assertTrue(wave_summary.exists())
            self.assertTrue(summary.exists())
            self.assertTrue(success_matrix.exists())
            self.assertTrue(breach_matrix.exists())
            self.assertTrue(status_log.exists())
            self.assertTrue((out_dir / "success_rate_by_strategy.png").exists())
            self.assertTrue((out_dir / "success_rate_by_defender_count.png").exists())
            self.assertTrue((out_dir / "success_rate_by_attacker_count.png").exists())
            self.assertTrue((out_dir / "kill_ratio_distribution.png").exists())
            self.assertTrue((out_dir / "completion_time_distribution.png").exists())
            self.assertTrue((out_dir / "breach_rate_by_defender_count.png").exists())
            self.assertTrue((out_dir / "avg_breaches_by_defender_count.png").exists())
            self.assertTrue((out_dir / "breach_rate_distribution.png").exists())
            self.assertTrue((out_dir / "first_breach_time_distribution.png").exists())
            self.assertTrue((out_dir / "breach_rate_by_wave.png").exists())
            self.assertTrue((out_dir / "kill_ratio_by_wave.png").exists())
            self.assertTrue((out_dir / "contention_rate_by_strategy.png").exists())
            self.assertTrue((out_dir / "max_contention_distribution.png").exists())

            with per_run.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5 * len(STRATEGIES))
            self.assertIn("success", rows[0])
            self.assertIn("wave_count", rows[0])
            self.assertIn("breach_rate", rows[0])
            self.assertIn("first_breach_time_s", rows[0])
            self.assertIn("contention_rate", rows[0])
            self.assertIn("max_simultaneous_defenders_on_target", rows[0])
            with wave_summary.open(newline="") as handle:
                wave_rows = list(csv.DictReader(handle))
            self.assertEqual(len(wave_rows), 5 * len(STRATEGIES))
            self.assertIn("wave_id", wave_rows[0])
            self.assertIn("breach_rate", wave_rows[0])
            with success_matrix.open(newline="") as handle:
                matrix_rows = list(csv.DictReader(handle))
            self.assertIn("defense_drones", matrix_rows[0])
            self.assertIn("attack_5_10", matrix_rows[0])
            with breach_matrix.open(newline="") as handle:
                breach_matrix_rows = list(csv.DictReader(handle))
            self.assertIn("attack_5_10", breach_matrix_rows[0])

            log_text = status_log.read_text(encoding="utf-8")
            self.assertIn("START runs=5", log_text)
            self.assertIn("SCENARIO_START run_id=0", log_text)
            self.assertIn("STRATEGY_DONE run_id=0", log_text)
            self.assertIn(f"FINISH strategy_runs={5 * len(STRATEGIES)}", log_text)


def _observations_for(scenario: Scenario) -> dict[int, Observation]:
    return {
        attacker.id: Observation(
            attacker.id,
            attacker.position.copy(),
            attacker.velocity.copy(),
            attacker.corridor,
        )
        for attacker in scenario.attackers
    }


if __name__ == "__main__":
    unittest.main()
