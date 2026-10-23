import json

import numpy as np
import pytest

from bg_exp001.evaluation import develop, finalize, prepare, weighted_spearman


def test_development_does_not_read_test_responses_or_parameters(tmp_path):
    cfg = {
        "experiment_id": "boundary-check",
        "seeds": [73],
        "n_structures": {"train": 15, "development": 8, "test": 9},
        "n_loads": {"train": 4, "development": 5, "test": 6},
        "base_stiffness": [1.0] * 11,
        "log_stiffness_global_sd": 0.35,
        "log_stiffness_edge_sd": 0.55,
        "load_node_sd": [1.0] * 6,
        "load_common_sd": 0.35,
        "fit_pair_count": 100,
        "mahalanobis_ridge_ratios": [0.01, 1.0],
        "neighbor_k": 3,
    }
    prepare(tmp_path, cfg)
    (tmp_path / "data" / "seed_73_test.npz").unlink()
    develop(tmp_path, cfg)
    result = json.loads((tmp_path / "results" / "development.json").read_text())
    assert result["selected_simple_baseline"]
    with pytest.raises(FileNotFoundError):
        finalize(tmp_path, cfg)  # A protocol lock is mandatory before test evaluation.


def test_weighted_spearman_ranks_and_pair_weights():
    a = np.array([1., 2., 3., 4.])
    w = np.array([1., 2., 0., 1.])
    assert np.isclose(weighted_spearman(a, a, w), 1)
    assert np.isclose(weighted_spearman(a, -a, w), -1)
