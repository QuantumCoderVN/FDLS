import numpy as np

from fdls5bus import case5, solve_fdls


def test_fdls_case5_converges():
    result = solve_fdls(case5(), tolerance=1e-8, max_iterations=50)
    assert result.converged
    assert result.iterations <= 10


def test_fdls_case5_voltage_solution():
    result = solve_fdls(case5(), tolerance=1e-8, max_iterations=50)
    expected_vm = np.array([1.0, 0.98926124, 1.0, 1.0, 1.0])
    expected_va_deg = np.array([3.27336086, -0.75926928, -0.49225867, 0.0, 4.11203100])
    assert np.allclose(result.vm, expected_vm, atol=1e-6)
    assert np.allclose(result.va_deg, expected_va_deg, atol=1e-6)


def test_fdls_case5_mismatch_small_on_solved_equations():
    result = solve_fdls(case5(), tolerance=1e-8, max_iterations=50)
    non_slack = np.array([0, 1, 2, 4])
    pq = np.array([1])
    assert np.max(np.abs(result.p_mismatch[non_slack])) < 1e-8
    assert np.max(np.abs(result.q_mismatch[pq])) < 1e-8
