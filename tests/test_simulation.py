from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from defensive_drones.engine import simulate_scenario, update_dwell_and_kills
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
        config = SimulationConfig(min_defenders=3, max_defenders=3)
        first = generate_scenario(1234, config)
        second = generate_scenario(1234, config)

        self.assertEqual(len(first.defenders), len(second.defenders))
        self.assertEqual(len(first.attackers), len(second.attackers))
        for left, right in zip(first.attackers, second.attackers):
            np.testing.assert_allclose(left.position, right.position)
            np.testing.assert_allclose(left.velocity, right.velocity)
            self.assertEqual(left.speed_mps, right.speed_mps)
            self.assertEqual(left.corridor, right.corridor)

    def test_attackers_have_independent_sampled_speeds(self) -> None:
        config = SimulationConfig(min_attackers=12, max_attackers=12)
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

        self.assertGreaterEqual(len(scenario.attackers), 250)
        self.assertLessEqual(len(scenario.attackers), 350)
        for attacker in scenario.attackers:
            distance = norm(attacker.position)
            self.assertGreaterEqual(distance, 4.5 * MILE_TO_M)
            self.assertLessEqual(distance, 5.5 * MILE_TO_M)

    def test_wide_hemisphere_spawn_spreads_attackers_across_corridors(self) -> None:
        config = SimulationConfig(min_attackers=350, max_attackers=350)
        scenario = generate_scenario(2027, config, defender_count=10)

        self.assertTrue(all(attacker.position[0] >= 0.0 for attacker in scenario.attackers))
        self.assertGreaterEqual(
            len({attacker.corridor for attacker in scenario.attackers}),
            8,
        )

    def test_same_seed_reuses_attacker_swarm_across_defender_counts(self) -> None:
        config = SimulationConfig(min_attackers=25, max_attackers=25)
        scenario_10 = generate_scenario(2028, config, defender_count=10)
        scenario_20 = generate_scenario(2028, config, defender_count=20)

        for attacker_10, attacker_20 in zip(
            scenario_10.attackers, scenario_20.attackers
        ):
            np.testing.assert_allclose(attacker_10.position, attacker_20.position)
            np.testing.assert_allclose(attacker_10.velocity, attacker_20.velocity)
            self.assertEqual(attacker_10.speed_mps, attacker_20.speed_mps)


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
        self.assertEqual(set(assignment.values()), {0, 1})
        self.assertEqual(len(set(assignment.values())), len(assignment.values()))

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
        )
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            results = run_monte_carlo(runs=5, seed=7, out_dir=out_dir, config=config)

            self.assertEqual(len(results), 5 * len(STRATEGIES))
            per_run = out_dir / "per_run_results.csv"
            summary = out_dir / "summary.csv"
            success_matrix = out_dir / "success_rate_matrix.csv"
            breach_matrix = out_dir / "breach_rate_matrix.csv"
            status_log = out_dir / "run_status.log"
            self.assertTrue(per_run.exists())
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

            with per_run.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5 * len(STRATEGIES))
            self.assertIn("success", rows[0])
            self.assertIn("breach_rate", rows[0])
            self.assertIn("first_breach_time_s", rows[0])
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
