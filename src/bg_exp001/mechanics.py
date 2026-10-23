"""Physically assembled six-DOF scalar spring network and exact sensitivities."""

from __future__ import annotations

import numpy as np

N_DOF = 6
# -1 denotes the fixed ground. Each row is an extension u_j-u_i, with u_ground=0.
EDGES: tuple[tuple[int, int], ...] = (
    (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 2), (1, 3), (2, 4), (3, 5), (5, -1),
)


def incidence(edges: tuple[tuple[int, int], ...] = EDGES, n_dof: int = N_DOF) -> np.ndarray:
    """Return B such that edge extension is B @ free_node_displacement."""
    b = np.zeros((len(edges), n_dof), dtype=float)
    for row, (i, j) in enumerate(edges):
        if i == j or any(node < -1 or node >= n_dof for node in (i, j)):
            raise ValueError("invalid spring edge")
        if i >= 0:
            b[row, i] -= 1.0
        if j >= 0:
            b[row, j] += 1.0
    return b


B = incidence()


def stiffness(theta: np.ndarray, b: np.ndarray = B) -> np.ndarray:
    """Assemble K=B.T diag(exp(theta)) B from positive spring stiffnesses."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (b.shape[0],) or not np.all(np.isfinite(theta)):
        raise ValueError("log-stiffness vector has wrong shape or nonfinite entries")
    k = np.exp(theta)
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("spring stiffness must be positive and finite")
    return b.T @ (k[:, None] * b)


def checked_solve(k_matrix: np.ndarray, loads: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Solve K u=f after checking symmetry and strict positive definiteness.

    Returns (u, smallest eigenvalue, spectral condition number); loads have shape
    (n_cases, n_dof). A one-dimensional load is also accepted.
    """
    matrix = np.asarray(k_matrix, dtype=float)
    f = np.asarray(loads, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("stiffness matrix must be square")
    if f.shape[-1] != matrix.shape[0] or f.ndim not in (1, 2):
        raise ValueError("load shape is inconsistent with the operator")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(f)):
        raise ValueError("nonfinite stiffness or load")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("nonsymmetric stiffness operator")
    eig = np.linalg.eigvalsh(matrix)
    if eig[0] <= max(1e-12, 1e-10 * eig[-1]):
        raise ValueError("singular, nearly singular, or indefinite stiffness operator")
    u = np.linalg.solve(matrix, f.T).T
    return u, float(eig[0]), float(eig[-1] / eig[0])


def response(theta: np.ndarray, loads: np.ndarray, b: np.ndarray = B) -> np.ndarray:
    return checked_solve(stiffness(theta, b), loads)[0]


def response_and_jacobian(
    theta: np.ndarray, loads: np.ndarray, b: np.ndarray = B
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return u and du/dtheta for every load; Jacobian shape is (load, dof, edge).

    dK/dtheta_e = k_e b_e b_e.T and du/dtheta_e = -K^-1 (dK/dtheta_e) u.
    """
    f = np.atleast_2d(np.asarray(loads, dtype=float))
    k_matrix = stiffness(theta, b)
    u, min_eigenvalue, condition = checked_solve(k_matrix, f)
    spring_k = np.exp(theta)
    extensions = u @ b.T  # (load, edge)
    rhs = -spring_k[None, None, :] * extensions[:, None, :] * b.T[None, :, :]
    n_load, n_dof, n_edge = rhs.shape
    packed = rhs.transpose(1, 0, 2).reshape(n_dof, n_load * n_edge)
    jac = np.linalg.solve(k_matrix, packed).reshape(n_dof, n_load, n_edge).transpose(1, 0, 2)
    return u, jac, min_eigenvalue, condition
