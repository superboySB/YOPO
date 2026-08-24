#!/usr/bin/env python3
"""Deterministic numerical tests for the controlled trajectory decoder ablation."""

import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "YOPO"))

from policy.poly_solver import MincoTraj, QuinticTraj  # noqa: E402

legacy_spec = importlib.util.spec_from_file_location(
    "legacy_poly_solver", ROOT / "YOPO" / "simple_runtime" / "policy" / "poly_solver.py")
legacy_poly_solver = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy_poly_solver)


def check_boundaries(traj, head, tail, total, atol=1e-6):
    for derivative, evaluator in enumerate((traj.position, traj.velocity, traj.acceleration)):
        np.testing.assert_allclose(evaluator(0.0), head[:, derivative], atol=atol)
        np.testing.assert_allclose(evaluator(total), tail[:, derivative], atol=atol)


def main():
    rng = np.random.default_rng(20260824)
    count = 7
    head = rng.normal(size=(count, 3, 3))
    tail = rng.normal(size=(count, 3, 3))
    inner = rng.normal(size=(count, 3))
    total = 5.0 / 3.0

    single = QuinticTraj().solve(head, tail, np.full(count, total))
    fixed = MincoTraj().solve(head, tail, inner, np.full((count, 2), total / 2.0))
    check_boundaries(single, head, tail, total)
    check_boundaries(fixed, head, tail, total)

    fractions = np.linspace(0.0, 1.0, 21)
    assert single.sample(fractions).shape == (count, len(fractions), 3)
    assert fixed.sample(fractions).shape == (count, len(fractions), 3)
    np.testing.assert_allclose(single.sample([0.0, 1.0])[:, 0], head[:, 0], atol=1e-8)
    np.testing.assert_allclose(single.sample([0.0, 1.0])[:, 1], tail[:, 0], atol=1e-7)
    np.testing.assert_allclose(fixed.sample([0.0, 1.0])[:, 0], head[:, 0], atol=1e-8)
    np.testing.assert_allclose(fixed.sample([0.0, 1.0])[:, 1], tail[:, 0], atol=1e-7)

    # A two-piece trajectory with its inner position placed on the equivalent single segment and
    # C1/C2 endpoints is not generally identical because MINCO also imposes C3/C4 continuity; it
    # must nevertheless interpolate the inner constraint exactly at the boundary.
    np.testing.assert_allclose(fixed.position(total / 2.0), inner, atol=1e-7)

    # The single-segment decoder is numerically equivalent to YOPO-Simple's original scalar solver,
    # not merely another curve satisfying the same endpoint constraints.
    times = np.linspace(0.0, total, 31)
    for batch_index in range(count):
        for axis in range(3):
            legacy = legacy_poly_solver.Poly5Solver(
                head[batch_index, 0, axis], head[batch_index, 1, axis], head[batch_index, 2, axis],
                tail[batch_index, 0, axis], tail[batch_index, 1, axis], tail[batch_index, 2, axis], total)
            np.testing.assert_allclose(single.position(times)[batch_index, :, axis],
                                       legacy.get_position(times), atol=1e-9)
            np.testing.assert_allclose(single.velocity(times)[batch_index, :, axis],
                                       legacy.get_velocity(times), atol=1e-9)
            np.testing.assert_allclose(single.acceleration(times)[batch_index, :, axis],
                                       legacy.get_acceleration(times), atol=1e-8)
    print("trajectory decoder tests: PASS")


if __name__ == "__main__":
    main()
