"""Matched coordinate, global, response-profile, and local distances."""

from __future__ import annotations

import numpy as np

from .mechanics import response, response_and_jacobian


def pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, 1)


def euclidean(theta: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    i, j = pairs
    return np.linalg.norm(theta[i] - theta[j], axis=1)


def standardize_fit(theta_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = theta_train.mean(axis=0)
    scale = theta_train.std(axis=0, ddof=1)
    if np.any(scale <= 0):
        raise ValueError("constant log-stiffness coordinate")
    return mean, scale


def response_profiles(theta: np.ndarray, loads: np.ndarray) -> np.ndarray:
    return np.stack([response(row, loads) for row in theta])


def profile_distances(profiles: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    i, j = pairs
    delta = profiles[i] - profiles[j]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))


def local_tensors(theta: np.ndarray, loads: np.ndarray) -> tuple[np.ndarray, dict]:
    tensors = []
    min_eigenvalues = []
    conditions = []
    for row in theta:
        _, jac, min_eig, condition = response_and_jacobian(row, loads)
        tensors.append(np.einsum("lie,lif->ef", jac, jac) / len(loads))
        min_eigenvalues.append(min_eig)
        conditions.append(condition)
    return np.stack(tensors), {
        "min_stiffness_eigenvalue": float(np.min(min_eigenvalues)),
        "max_stiffness_condition": float(np.max(conditions)),
        "min_tensor_eigenvalue": float(min(np.linalg.eigvalsh(m)[0] for m in tensors)),
    }


def local_distances(
    theta: np.ndarray, tensors: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Endpoint-averaged pullback *dissimilarity*, not asserted a global metric."""
    i, j = pairs
    delta = theta[i] - theta[j]
    m = 0.5 * (tensors[i] + tensors[j])
    q = np.einsum("pi,pij,pj->p", delta, m, delta)
    if np.min(q) < -1e-9:
        raise ValueError("negative local quadratic distance")
    return np.sqrt(np.maximum(q, 0.0))


def global_sensitivity_matrix(tensors_train: np.ndarray) -> np.ndarray:
    return tensors_train.mean(axis=0)


def quadratic_distances(
    theta: np.ndarray, matrix: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    i, j = pairs
    delta = theta[i] - theta[j]
    q = np.einsum("pi,ij,pj->p", delta, matrix, delta)
    if np.min(q) < -1e-8:
        raise ValueError("global matrix is not positive semidefinite")
    return np.sqrt(np.maximum(q, 0.0))


def pair_design(z: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Design for squared Mahalanobis distances, one column per M[i,j], i<=j."""
    i, j = pairs
    delta = z[i] - z[j]
    columns = []
    for a in range(z.shape[1]):
        for b in range(a, z.shape[1]):
            columns.append(delta[:, a] * delta[:, b] * (1 if a == b else 2))
    return np.column_stack(columns)


def fit_mahalanobis(
    theta: np.ndarray,
    target_distances: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    ridge_ratio: float,
) -> tuple[np.ndarray, dict]:
    """Ridge fit on training squared response distances; project to PSD."""
    z = (theta - mean) / scale
    x = pair_design(z, pairs)
    gram = x.T @ x
    ridge = ridge_ratio * np.trace(gram) / len(gram)
    coeff = np.linalg.solve(gram + ridge * np.eye(len(gram)), x.T @ target_distances**2)
    matrix = np.zeros((z.shape[1], z.shape[1]), dtype=float)
    t = 0
    for a in range(z.shape[1]):
        for b in range(a, z.shape[1]):
            matrix[a, b] = matrix[b, a] = coeff[t]
            t += 1
    eig, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.maximum(eig, 0)) @ vectors.T
    return matrix, {
        "ridge_ratio": ridge_ratio,
        "unprojected_min_eigenvalue": float(eig[0]),
        "projected_rank": int(np.sum(eig > 1e-10)),
    }


def full_distances(
    theta: np.ndarray,
    reference_loads: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    fitted_mahalanobis: np.ndarray,
    global_sensitivity: np.ndarray,
    include_controls: bool = False,
    control_seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, float | dict]]:
    """Compute all predictions from inputs/reference loads only, never oracle loads."""
    import time

    pairs = pair_indices(len(theta))
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float | dict] = {}
    start = time.perf_counter()
    outputs["euclidean"] = euclidean(theta, pairs)
    timings["euclidean"] = time.perf_counter() - start
    start = time.perf_counter()
    z = (theta - mean) / scale
    outputs["standardized_euclidean"] = euclidean(z, pairs)
    timings["standardized_euclidean"] = time.perf_counter() - start
    start = time.perf_counter()
    outputs["global_mahalanobis"] = quadratic_distances(z, fitted_mahalanobis, pairs)
    timings["global_mahalanobis"] = time.perf_counter() - start
    start = time.perf_counter()
    outputs["global_sensitivity"] = quadratic_distances(theta, global_sensitivity, pairs)
    timings["global_sensitivity"] = time.perf_counter() - start
    start = time.perf_counter()
    profiles = response_profiles(theta, reference_loads)
    outputs["direct_response_profile"] = profile_distances(profiles, pairs)
    timings["direct_response_profile"] = time.perf_counter() - start
    start = time.perf_counter()
    tensors, stability = local_tensors(theta, reference_loads)
    outputs["local_jacobian_pullback"] = local_distances(theta, tensors, pairs)
    timings["local_jacobian_pullback"] = time.perf_counter() - start
    timings["stability"] = stability
    if include_controls:
        rng = np.random.default_rng(control_seed)
        scrambled = np.stack([row[rng.permutation(row.size)] for row in theta])
        start = time.perf_counter()
        scramble_profile = response_profiles(scrambled, reference_loads)
        outputs["scrambled_response_profile"] = profile_distances(scramble_profile, pairs)
        timings["scrambled_response_profile"] = time.perf_counter() - start
        start = time.perf_counter()
        scramble_tensors, _ = local_tensors(scrambled, reference_loads)
        outputs["scrambled_local_pullback"] = local_distances(scrambled, scramble_tensors, pairs)
        timings["scrambled_local_pullback"] = time.perf_counter() - start
    for name, values in outputs.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"invalid predicted distances: {name}")
    return outputs, timings
