#!/usr/bin/env python3
"""Deterministic unit checks for MINCO jerk evaluation and online selection modes."""

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "YOPO"))

from policy.poly_solver import MincoTraj  # noqa: E402
from test_yopo_ros import YopoNet  # noqa: E402


def test_jerk_derivative():
    rng = np.random.default_rng(7)
    batch = 8
    head = rng.normal(size=(batch, 3, 3))
    tail = rng.normal(size=(batch, 3, 3))
    inner = rng.normal(size=(batch, 3))
    durations = rng.uniform(0.3, 1.2, size=(batch, 2))
    trajectory = MincoTraj().solve(head, tail, inner, durations)
    time, delta = 0.17, 1e-4
    finite_difference = (trajectory.acceleration(time + delta)
                         - trajectory.acceleration(time - delta)) / (2 * delta)
    np.testing.assert_allclose(trajectory.jerk(time), finite_difference, rtol=2e-6, atol=3e-6)
    assert trajectory.jerk(time).shape == (batch, 3)


def bare_selector(mode="none", topk=1):
    selector = YopoNet.__new__(YopoNet)
    selector.safe_mu = 0.1
    selector.topk = topk
    selector.continuity_mode = mode
    selector.best_inner_w = None
    selector.optimal_traj = None
    selector.ctrl_time = 0.1
    return selector


def test_score_and_inner_selection():
    scores = np.array([1.0, 3.0, 2.0])
    corridor = np.array([0.2, 0.2, 0.2])
    inner = np.array([[4.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    selector = bare_selector()
    assert selector.select_action(scores, inner, corridor) == (1, False)
    selector = bare_selector("inner", 2)
    selector.optimal_traj = object()
    selector.best_inner_w = np.zeros(3)
    assert selector.select_action(scores, inner, corridor) == (2, False)
    assert selector.select_action(scores, inner, np.zeros(3)) == (1, True)


def test_jerk_selection():
    rng = np.random.default_rng(11)
    count = 4
    head = np.zeros((3, 3))
    tail = rng.normal(size=(count, 3, 3))
    inner = rng.normal(size=(count, 3))
    durations = np.full((count, 2), 0.8)
    candidate = MincoTraj().solve(np.broadcast_to(head, (count, 3, 3)).copy(),
                                  tail, inner, durations)
    desired = 2
    target_jerk = candidate.jerk(0.0)[desired]

    class Prior:
        total_time = 1.0
        def jerk(self, _time):
            return target_jerk

    selector = bare_selector("jerk", count)
    selector.optimal_traj = Prior()
    selected, brake = selector.select_action(
        np.arange(count, dtype=float), inner, np.ones(count), head, tail, durations)
    assert (selected, brake) == (desired, False)


def test_corridor_multiplier():
    selector = bare_selector()
    selector.radius_num = 2
    selector.corridor_sigma = 2.0
    radius = np.array([[[[0.7]], [[0.5]], [[0.1]], [[0.2]]]])
    np.testing.assert_allclose(selector._corridor_mu_lo(radius), [0.1])


if __name__ == "__main__":
    test_jerk_derivative()
    test_score_and_inner_selection()
    test_jerk_selection()
    test_corridor_multiplier()
    print("MINCO selection tests: PASS")
