"""Tests for the r-concave bound implementation in rconcave.py.

Combines analytical property checks (monotonicity, trivial cases, Markov bound)
with seed-pinned golden values to catch silent numerical regressions.
"""

import pytest

from stably.rconcave import compute_D, compute_rconcave_threshold


def test_compute_D_trivial_t_zero():
    assert compute_D(0.1, 0.0, 50, -0.5) == 1.0


def test_compute_D_trivial_eta_zero():
    assert compute_D(0.0, 0.5, 50, -0.5) == 0.0


def test_compute_D_trivial_eta_ge_t():
    assert compute_D(0.6, 0.5, 50, -0.5) == 1.0
    assert compute_D(0.5, 0.5, 50, -0.5) == 1.0


def test_compute_D_bounded_by_one():
    assert 0.0 <= compute_D(0.3, 0.5, 50, -0.5) <= 1.0


def test_compute_D_monotone_in_eta():
    """For fixed (t, N, r), D should be monotone non-decreasing in eta."""
    t, N, r = 0.5, 50, -0.5
    etas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    values = [compute_D(eta, t, N, r) for eta in etas]
    for prev, curr in zip(values, values[1:]):
        assert curr + 1e-9 >= prev, f"D not monotone: {values}"


def test_compute_D_bounded_by_markov():
    """D(eta, t, N, r) is at most the Markov bound min(eta/t, 1)."""
    cases = [
        (0.05, 0.7, 50, -0.5),
        (0.1, 0.5, 50, -0.5),
        (0.2, 0.6, 100, -0.25),
        (0.05, 0.4, 50, -0.5),
    ]
    for eta, t, N, r in cases:
        markov = min(eta / t, 1.0)
        bound = compute_D(eta, t, N, r)
        assert bound <= markov + 1e-9, (
            f"D({eta}, {t}, {N}, {r}) = {bound} exceeds Markov bound {markov}"
        )


def test_compute_D_golden_values():
    """Seed-pinned golden values from the v0.3.0 implementation. If these change,
    investigate before updating — the r-concave bound is a numerical optimisation
    over distributions and silent drift would invalidate every PFER claim the
    package has ever made."""
    assert compute_D(0.1, 0.5, 50, -0.5) == pytest.approx(0.04540068059019453, abs=1e-10)
    assert compute_D(0.05, 0.7, 50, -0.5) == pytest.approx(0.008176610008194252, abs=1e-10)
    assert compute_D(0.2, 0.6, 100, -0.25) == pytest.approx(0.06813698604766172, abs=1e-10)


def test_compute_rconcave_threshold_tighter_than_mb():
    """The Shah & Samworth r-concave threshold should be lower (tighter) than
    the Meinshausen & Buhlmann worst-case threshold for representative inputs.
    This is the central reason to use S&S over M&B."""
    tau_rc, _, tau_mb = compute_rconcave_threshold(q=50, p=500, B=50, pfer_target=10)
    assert tau_rc < tau_mb

    tau_rc, _, tau_mb = compute_rconcave_threshold(q=20, p=200, B=25, pfer_target=5)
    assert tau_rc < tau_mb


def test_compute_rconcave_threshold_satisfies_pfer():
    """The threshold returned must produce a PFER bound at-or-below the target."""
    pfer_target = 10
    _, bound, _ = compute_rconcave_threshold(q=50, p=500, B=50, pfer_target=pfer_target)
    assert bound <= pfer_target


def test_compute_rconcave_threshold_in_unit_interval():
    tau, _, _ = compute_rconcave_threshold(q=50, p=500, B=50, pfer_target=10)
    assert 0.0 < tau <= 1.0


def test_compute_rconcave_threshold_golden():
    """Golden values for two representative shapes."""
    tau, bound, mb = compute_rconcave_threshold(q=50, p=500, B=50, pfer_target=10)
    assert tau == pytest.approx(0.55, abs=1e-9)
    assert bound == pytest.approx(9.60747946881746, abs=1e-6)
    assert mb == pytest.approx(0.75, abs=1e-9)

    tau, bound, mb = compute_rconcave_threshold(q=20, p=200, B=25, pfer_target=5)
    assert tau == pytest.approx(0.52, abs=1e-9)
    assert bound == pytest.approx(4.971542343223834, abs=1e-6)
    assert mb == pytest.approx(0.7, abs=1e-9)
