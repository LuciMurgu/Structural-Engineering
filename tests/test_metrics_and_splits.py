import numpy as np

from bg_exp001.data import generate_seed
from bg_exp001.mechanics import B, response
from bg_exp001.metrics import (
    local_distances, local_tensors, pair_indices, profile_distances,
    response_profiles,
)


def test_split_streams_reproducible_disjoint_and_independent():
    cfg = {
        "base_stiffness": [1.0] * B.shape[0],
        "log_stiffness_global_sd": 0.35,
        "log_stiffness_edge_sd": 0.55,
        "load_node_sd": [1.0] * 6,
        "load_common_sd": 0.35,
        "n_structures": {"train": 5, "development": 4, "test": 3},
        "n_loads": {"train": 4, "development": 3, "test": 5},
    }
    first, second = generate_seed(41, cfg), generate_seed(41, cfg)
    for split in first:
        for field in ("theta", "loads"):
            assert np.array_equal(first[split][field], second[split][field])
    assert not np.array_equal(first["train"]["theta"][0], first["test"]["theta"][0])
    assert not np.array_equal(first["train"]["loads"][0], first["test"]["loads"][0])


def test_local_tensor_is_psd_and_first_order_correct():
    rng = np.random.default_rng(84)
    theta = rng.normal(0, 0.2, B.shape[0])
    loads = rng.normal(size=(10, 6))
    tensor = local_tensors(theta[None, :], loads)[0][0]
    assert np.linalg.eigvalsh(tensor)[0] > -1e-10
    direction = rng.normal(size=len(theta))
    step = 1e-4
    actual = response(theta + step * direction, loads) - response(theta, loads)
    predicted_squared = step**2 * direction @ tensor @ direction
    actual_squared = np.mean(np.sum(actual**2, axis=1))
    assert np.isclose(actual_squared, predicted_squared, rtol=1e-3)


def test_profile_uses_only_supplied_loads_and_local_pair_is_symmetric():
    rng = np.random.default_rng(99)
    theta = rng.normal(size=(3, B.shape[0])) * 0.2
    loads = rng.normal(size=(5, 6))
    profiles = response_profiles(theta, loads)
    pair = pair_indices(3)
    distances = profile_distances(profiles, pair)
    assert np.all(distances >= 0)
    assert not np.allclose(distances, profile_distances(response_profiles(theta, loads * 2), pair))
    tensors = local_tensors(theta, loads)[0]
    forward = local_distances(theta, tensors, (np.array([0]), np.array([1])))
    reverse = local_distances(theta, tensors, (np.array([1]), np.array([0])))
    assert np.allclose(forward, reverse)
