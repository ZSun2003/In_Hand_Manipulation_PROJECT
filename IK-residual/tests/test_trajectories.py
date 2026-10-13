"""Tests for the local canonical translation trajectory implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from g3g6 import trajectories as tr


def _sampled_arc_lengths(path: str, subdivisions: int = 200) -> np.ndarray:
    """Numerically integrate each of the 20 normalized-progress intervals."""

    # Include the N's two zero-radius corners explicitly.  Otherwise a
    # polyline chord can cut across a corner that lies between two numerical
    # samples and under-report that anchor interval despite exact arc-length
    # parameterization.
    corner_progress = (4.0 / 13.0, 9.0 / 13.0) if path == "N" else ()
    lengths = []
    for interval in range(tr.ANCHOR_INTERVALS):
        start = interval / tr.ANCHOR_INTERVALS
        stop = (interval + 1) / tr.ANCHOR_INTERVALS
        knots = [start]
        knots.extend(corner for corner in corner_progress if start < corner < stop)
        knots.append(stop)
        length = 0.0
        for left, right in zip(knots, knots[1:]):
            progress = np.linspace(left, right, subdivisions + 1, dtype=np.float64)
            points = tr.canonical_point_xy(path, progress)
            length += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        lengths.append(length)
    return np.asarray(lengths)


class CanonicalGeometryTests(unittest.TestCase):
    def test_names_aliases_and_exact_path_lengths(self) -> None:
        self.assertEqual(tr.PATH_NAMES, ("square", "circle", "S", "U", "N"))
        self.assertEqual(tr.normalize_path_name("square_100mm"), "square")
        self.assertEqual(tr.normalize_path_name("LETTER_S_300MM"), "S")
        self.assertEqual(tr.normalize_path_name(4), "N")
        np.testing.assert_allclose(
            tr.PATH_LENGTHS_M,
            (0.400, 0.100 * math.pi, 0.300, 0.300, 0.300),
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_authoritative_endpoints_and_corner_geometry(self) -> None:
        radius_s = 0.150 / (5.0 * math.pi / 4.0)
        s_terminal_x = radius_s / math.sqrt(2.0)
        s_terminal_y = radius_s * (1.0 + 1.0 / math.sqrt(2.0))
        u_vertical = (0.300 - math.pi * 0.045) / 2.0
        u_center_y = (0.045 - u_vertical) / 2.0
        n_scale = 0.300 / 13.0
        expected = {
            "square": ((0.050, 0.050), (0.050, 0.050)),
            "circle": ((0.050, 0.0), (0.050, 0.0)),
            "S": (
                (s_terminal_x, s_terminal_y),
                (-s_terminal_x, -s_terminal_y),
            ),
            "U": (
                (-0.045, u_center_y + u_vertical),
                (0.045, u_center_y + u_vertical),
            ),
            "N": (
                (-1.5 * n_scale, -2.0 * n_scale),
                (1.5 * n_scale, 2.0 * n_scale),
            ),
        }
        for path, (start, end) in expected.items():
            with self.subTest(path=path):
                np.testing.assert_allclose(
                    tr.canonical_point_xy(path, 0.0), start, rtol=0.0, atol=1.0e-14
                )
                np.testing.assert_allclose(
                    tr.canonical_point_xy(path, 1.0), end, rtol=0.0, atol=1.0e-14
                )

        square_corners = tr.canonical_point_xy(
            "square", np.array((0.0, 0.25, 0.50, 0.75, 1.0))
        )
        np.testing.assert_allclose(
            square_corners,
            (
                (0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, -0.05),
                (0.05, -0.05),
                (0.05, 0.05),
            ),
            rtol=0.0,
            atol=1.0e-14,
        )

        n_corners = tr.canonical_point_xy(
            "N", np.array((0.0, 4.0 / 13.0, 9.0 / 13.0, 1.0))
        )
        np.testing.assert_allclose(
            n_corners,
            (
                (-1.5 * n_scale, -2.0 * n_scale),
                (-1.5 * n_scale, 2.0 * n_scale),
                (1.5 * n_scale, -2.0 * n_scale),
                (1.5 * n_scale, 2.0 * n_scale),
            ),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_s_has_two_225_degree_arcs_with_a_tangent_continuous_join(self) -> None:
        self.assertAlmostEqual(tr.S_ARC_SWEEP_RAD, 5.0 * math.pi / 4.0)
        self.assertAlmostEqual(tr.S_RADIUS_M * tr.S_ARC_SWEEP_RAD, 0.150)
        points = tr.canonical_point_xy(
            "S", np.array((0.0, 0.5 - 1.0e-7, 0.5, 0.5 + 1.0e-7, 1.0))
        )
        np.testing.assert_array_equal(points[2], (0.0, 0.0))
        incoming = points[2] - points[1]
        outgoing = points[3] - points[2]
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        np.testing.assert_allclose(incoming, (1.0, 0.0), atol=5.0e-7)
        np.testing.assert_allclose(outgoing, (1.0, 0.0), atol=5.0e-7)
        np.testing.assert_allclose(points[0], -points[-1], atol=1.0e-15)

    def test_21_anchors_are_equal_arc_length(self) -> None:
        for path, path_length in zip(tr.PATH_NAMES, tr.PATH_LENGTHS_M):
            with self.subTest(path=path):
                anchors = tr.canonical_anchors_xy(path)
                self.assertEqual(anchors.shape, (21, 2))
                np.testing.assert_allclose(
                    anchors,
                    np.round(
                        tr.canonical_point_xy(path, np.linspace(0.0, 1.0, 21)),
                        decimals=12,
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                interval_lengths = _sampled_arc_lengths(path)
                np.testing.assert_allclose(
                    interval_lengths,
                    path_length / 20.0,
                    rtol=0.0,
                    # Polyline integration only underestimates at curved
                    # pieces and at a corner landing between tiny samples.
                    atol=2.0e-6,
                )

    def test_route_polyline_lengths_match_the_canonical_lengths(self) -> None:
        for path, expected in zip(tr.PATH_NAMES, tr.PATH_LENGTHS_M):
            with self.subTest(path=path):
                route = tr.route_points_xy(path)
                self.assertEqual(route.shape, (tr.ROUTE_TICKS + 1, 2))
                measured = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
                self.assertAlmostEqual(measured, expected, delta=3.0e-6)


class MinimumJerkAndTimingTests(unittest.TestCase):
    def test_minimum_jerk_endpoints_monotonicity_and_finiteness(self) -> None:
        tau = np.linspace(0.0, 1.0, 10_001)
        profile = tr.minimum_jerk(tau)
        self.assertEqual(profile[0], 0.0)
        self.assertEqual(profile[-1], 1.0)
        self.assertTrue(np.isfinite(profile).all())
        self.assertTrue(np.all(np.diff(profile) >= 0.0))
        # Zero endpoint slope and curvature are the reason corners can be
        # split into independent strokes without a velocity discontinuity.
        self.assertAlmostEqual(float(profile[1] - profile[0]), 0.0, delta=1.1e-11)
        self.assertAlmostEqual(float(profile[-1] - profile[-2]), 0.0, delta=1.1e-11)

    def test_stroke_tick_contract_and_progress_boundaries(self) -> None:
        self.assertEqual(tr.SQUARE_STROKE_TICKS, (450, 450, 450, 450))
        self.assertEqual(tr.N_STROKE_TICKS, (555, 690, 555))
        self.assertEqual(sum(tr.SQUARE_STROKE_TICKS), 1_800)
        self.assertEqual(sum(tr.N_STROKE_TICKS), 1_800)

        square = tr.route_progress("square")
        np.testing.assert_array_equal(
            square[[0, 450, 900, 1_350, 1_800]],
            np.array((0.0, 0.25, 0.50, 0.75, 1.0)),
        )
        n_path = tr.route_progress("N")
        np.testing.assert_allclose(
            n_path[[0, 555, 1_245, 1_800]],
            (0.0, 4.0 / 13.0, 9.0 / 13.0, 1.0),
            rtol=0.0,
            atol=5.1e-13,
        )
        for path in ("circle", "S", "U"):
            progress = tr.route_progress(path)
            self.assertEqual(progress.shape, (1_801,))
            self.assertEqual(progress[0], 0.0)
            self.assertEqual(progress[-1], 1.0)
            self.assertTrue(np.all(np.diff(progress) >= 0.0))

    def test_2425_tick_phase_boundaries_are_exact(self) -> None:
        self.assertEqual((tr.LEAD_IN_TICKS, tr.START_SETTLE_TICKS), (300, 300))
        self.assertEqual((tr.ROUTE_TICKS, tr.TERMINAL_HOLD_TICKS), (1_800, 25))
        self.assertEqual(tr.TOTAL_TICKS, 2_425)
        self.assertEqual(
            tr.PHASE_BOUNDS,
            {
                "lead_in": (0, 300),
                "settle": (300, 600),
                "route": (600, 2_400),
                "hold": (2_400, 2_425),
            },
        )
        self.assertEqual(tr.phase_for_tick(0), "lead_in")
        self.assertEqual(tr.phase_for_tick(299), "lead_in")
        self.assertEqual(tr.phase_for_tick(300), "settle")
        self.assertEqual(tr.phase_for_tick(599), "settle")
        self.assertEqual(tr.phase_for_tick(600), "route")
        self.assertEqual(tr.phase_for_tick(2_399), "route")
        self.assertEqual(tr.phase_for_tick(2_400), "hold")
        self.assertEqual(tr.phase_for_tick(2_424), "hold")

    def test_schedule_uses_inclusive_tables_without_duplicate_ticks(self) -> None:
        for path in tr.PATH_NAMES:
            with self.subTest(path=path):
                lead = tr.lead_in_points_xy(path)
                route = tr.route_points_xy(path)
                schedule = tr.build_schedule(path)
                self.assertEqual(lead.shape, (301, 2))
                self.assertEqual(schedule.shape, (2_425, 3))

                np.testing.assert_array_equal(schedule[:300, :2], lead[1:])
                np.testing.assert_array_equal(
                    schedule[300:600, :2],
                    np.repeat(route[0][None, :], 300, axis=0),
                )
                np.testing.assert_array_equal(schedule[600:2_400, :2], route[1:])
                np.testing.assert_array_equal(
                    schedule[2_400:, :2],
                    np.repeat(route[-1][None, :], 25, axis=0),
                )
                np.testing.assert_array_equal(schedule[:, 2], np.zeros(2_425))

        np.testing.assert_array_equal(
            tr.scheduled_path("circle", center_xyz=None),
            tr.build_schedule("circle"),
        )

    def test_every_xyz_axis_is_finite_and_z_remains_at_grasp_height(self) -> None:
        origin = np.array((0.125, -0.075, 0.360))
        for path in tr.PATH_NAMES:
            with self.subTest(path=path):
                schedule = tr.build_schedule(path, origin)
                self.assertTrue(np.isfinite(schedule).all(axis=0).all())
                np.testing.assert_array_equal(
                    schedule[:, 2], np.full(tr.TOTAL_TICKS, 0.360)
                )
                expected_start = origin[:2] + tr.canonical_point_xy(path, 0.0)
                expected_end = origin[:2] + tr.canonical_point_xy(path, 1.0)
                np.testing.assert_allclose(schedule[299, :2], expected_start, atol=5.1e-13)
                np.testing.assert_allclose(schedule[-1, :2], expected_end, atol=5.1e-13)

    def test_invalid_inputs_fail_closed(self) -> None:
        for bad_path in (True, -1, 5, "triangle", None):
            with self.subTest(path=bad_path):
                with self.assertRaises(ValueError):
                    tr.canonical_point_xy(bad_path, 0.0)  # type: ignore[arg-type]
        for bad_progress in (-0.01, 1.01, math.nan, math.inf):
            with self.subTest(progress=bad_progress):
                with self.assertRaises(ValueError):
                    tr.canonical_point_xy("circle", bad_progress)
        with self.assertRaises(ValueError):
            tr.build_schedule("square", (0.0, 0.0))
        with self.assertRaises(ValueError):
            tr.build_schedule("square", (0.0, math.nan, 0.36))
        for bad_tick in (-1, 2_425, 1.5, True):
            with self.subTest(tick=bad_tick):
                with self.assertRaises(ValueError):
                    tr.phase_for_tick(bad_tick)  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
