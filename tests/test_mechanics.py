import numpy as np
import pytest

from bg_exp001.mechanics import B, checked_solve, response, response_and_jacobian, stiffness


def test_assembly_energy_and_support_rank():
    theta = np.log(np.linspace(0.4, 2.0, B.shape[0]))
    k = stiffness(theta)
    u = np.array([0.2, -0.1, 0.4, 0.3, -0.2, 0.8])
    expected = np.sum(np.exp(theta) * (B @ u) ** 2)
    assert np.isclose(u @ k @ u, expected)
    assert np.linalg.matrix_rank(B) == 6
    assert np.linalg.eigvalsh(k)[0] > 0


def test_analytic_log_stiffness_jacobian_matches_central_difference():
    rng = np.random.default_rng(103)
    theta = rng.normal(0, 0.5, B.shape[0])
    loads = rng.normal(size=(4, 6))
    u, jac, _, _ = response_and_jacobian(theta, loads)
    assert np.allclose(u, response(theta, loads))
    step = 1e-5
    for e in range(len(theta)):
        shift = np.eye(len(theta))[e] * step
        finite = (response(theta + shift, loads) - response(theta - shift, loads)) / (2 * step)
        assert np.allclose(finite, jac[:, :, e], rtol=3e-7, atol=2e-9)


@pytest.mark.parametrize("bad", [
    np.diag([1., 1., 1., 1., 1., 0.]),
    np.diag([1., 1., 1., 1., 1., -0.1]),
    np.eye(6) * 1e-14,
])
def test_singular_and_indefinite_rejected(bad):
    with pytest.raises(ValueError, match="singular|indefinite"):
        checked_solve(bad, np.ones(6))


def test_nonsymmetric_rejected():
    bad = np.eye(6)
    bad[0, 1] = 0.3
    with pytest.raises(ValueError, match="nonsymmetric"):
        checked_solve(bad, np.ones(6))
