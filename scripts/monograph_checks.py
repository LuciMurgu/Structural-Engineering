#!/usr/bin/env python3
"""Independent reconstruction of the five checks named in monograph Appendix B."""

import numpy as np


def run_checks() -> None:
    # B.1: two-storey stiffness and exact energy distances.
    k = np.array([[2.0, -1.0], [-1.0, 1.0]])
    a, b = np.array([1.0, 1.0]), np.array([1.0, -1.0])
    d = lambda x, y: np.sqrt((x - y) @ k @ (x - y))
    assert np.isclose(d(a, np.zeros(2)), 1)
    assert np.isclose(d(b, np.zeros(2)), np.sqrt(5))
    assert np.isclose(d(a, b), 2)

    # B.2: graph B.T B energy and fractional nullspace for a tiny path.
    incidence = np.array([[1., -1., 0.], [0., 1., -1.]])
    lap = incidence.T @ incidence
    v = np.array([1.0, -2.0, 0.5])
    assert np.isclose(v @ lap @ v, np.sum((incidence @ v) ** 2))
    eig, q = np.linalg.eigh(lap)
    # A mathematical zero may be returned as a tiny positive eigenvalue;
    # fractional powers amplify that roundoff unless null modes are thresholded.
    eig[np.abs(eig) < 1e-12 * max(eig)] = 0.0
    for alpha in (0.25, 0.5, 0.75, 1.0):
        lap_alpha = (q * np.maximum(eig, 0) ** alpha) @ q.T
        assert np.linalg.eigvalsh(lap_alpha)[0] > -1e-9
        assert np.allclose(lap_alpha @ np.ones(3), 0, atol=1e-9)

    # B.3: observability Gramian of A=diag(-1,-2), C=[1,1], horizon T=2.
    rate = np.array([1.0, 2.0])
    horizon = 2.0
    sums = rate[:, None] + rate[None, :]
    gramian = (1 - np.exp(-sums * horizon)) / sums
    delta = np.array([0.4, -0.8])
    t = np.linspace(0, horizon, 10001)
    output = np.exp(-t[:, None] * rate[None, :]) @ delta
    integral = np.trapezoid(output**2, t)
    assert np.isclose(delta @ gramian @ delta, integral, rtol=1e-7)

    # B.4: Jacobian pullback numeric identity.
    jac = np.array([[1.0, 0.2], [0.0, 3.0]])
    weight = np.diag([1.0, 0.25])
    perturbation = np.array([0.0, 0.5])
    metric = jac.T @ weight @ jac
    assert np.isclose(perturbation @ metric @ perturbation, 0.5725)
    assert np.isclose(perturbation @ metric @ perturbation,
                      (jac @ perturbation) @ weight @ (jac @ perturbation))

    # B.5: Gaussian-load expected squared response distance versus Monte Carlo.
    other_k = np.array([[3.0, -1.5], [-1.5, 2.0]])
    covariance = np.array([[1.0, 0.2], [0.2, 0.5]])
    transfer_difference = np.linalg.inv(k) - np.linalg.inv(other_k)
    expected = np.trace(transfer_difference @ covariance @ transfer_difference.T)
    loads = np.random.default_rng(7123).multivariate_normal(np.zeros(2), covariance, size=40000)
    difference = loads @ transfer_difference.T
    estimate = np.mean(np.sum(difference**2, axis=1))
    assert np.isclose(estimate, expected, rtol=0.03)
    print("Five reconstructed Appendix B checks passed.")


if __name__ == "__main__":
    run_checks()
