"""Protocol-locked development, final evaluation, and structure-aware inference."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import read_json, read_split, sha256, write_dataset, write_json
from .metrics import (
    fit_mahalanobis, full_distances, global_sensitivity_matrix, pair_indices,
    profile_distances, quadratic_distances, response_profiles, local_tensors,
    standardize_fit,
)

METHODS = (
    "euclidean", "standardized_euclidean", "global_mahalanobis",
    "global_sensitivity", "direct_response_profile", "local_jacobian_pullback",
)
SIMPLE_BASELINES = (
    "direct_response_profile", "global_sensitivity", "global_mahalanobis",
    "standardized_euclidean", "euclidean",
)
CONTROLS = ("scrambled_response_profile", "scrambled_local_pullback", "permuted_target_mahalanobis")
LOCKED_CODE = (
    "configs/exp001.json", "pyproject.toml", "requirements.txt",
    "src/bg_exp001/mechanics.py", "src/bg_exp001/data.py",
    "src/bg_exp001/metrics.py", "src/bg_exp001/evaluation.py",
    "scripts/run_experiment.py", "scripts/monograph_checks.py", "manifests/PROTOCOL.md",
    "tests/test_mechanics.py", "tests/test_metrics_and_splits.py", "tests/test_evaluation_boundary.py",
    "references/Geometry_That_Understands_Behavior_extracted.txt",
)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return weighted_spearman(x, y, np.ones(len(x)))


def _weighted_midrank(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    ranks = np.empty(len(x), dtype=float)
    start = 0
    cumulative = 0.0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and xs[stop] == xs[start]:
            stop += 1
        mass = float(ws[start:stop].sum())
        ranks[order[start:stop]] = cumulative + mass / 2
        cumulative += mass
        start = stop
    return ranks


def weighted_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    active = weights > 0
    x, y, w = np.asarray(x)[active], np.asarray(y)[active], np.asarray(weights)[active]
    if len(x) < 3:
        return float("nan")
    rx, ry = _weighted_midrank(x, w), _weighted_midrank(y, w)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    covariance = np.sum(w * (rx - mx) * (ry - my))
    denominator = np.sqrt(np.sum(w * (rx - mx) ** 2) * np.sum(w * (ry - my) ** 2))
    return float(covariance / denominator) if denominator > 0 else float("nan")


def neighbor_precision(predicted: np.ndarray, oracle: np.ndarray, n: int, k: int) -> np.ndarray:
    if k >= n:
        raise ValueError("k must be smaller than number of structures")
    i, j = pair_indices(n)
    p = np.full((n, n), np.inf)
    o = np.full((n, n), np.inf)
    p[i, j] = p[j, i] = predicted
    o[i, j] = o[j, i] = oracle
    pred_near = np.argsort(p, axis=1)[:, :k]
    true_near = np.argsort(o, axis=1)[:, :k]
    return np.array([len(set(a) & set(b)) / k for a, b in zip(pred_near, true_near)])


def score_predictions(predictions: dict[str, np.ndarray], oracle: np.ndarray, n: int, k: int) -> dict:
    return {
        name: {
            "spearman": spearman(values, oracle),
            "precision_at_k": float(neighbor_precision(values, oracle, n, k).mean()),
            "nonfinite_count": int(np.size(values) - np.isfinite(values).sum()),
        }
        for name, values in predictions.items()
    }


def prepare(root: Path, cfg: dict) -> None:
    manifest = write_dataset(cfg, root / "data")
    write_json(root / "manifests" / "splits.json", manifest)


def _train_fit(root: Path, cfg: dict, manifest: dict, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    theta, loads = read_split(root / "data", manifest, seed, "train")
    mean, scale = standardize_fit(theta)
    start = time.perf_counter()
    profiles = response_profiles(theta, loads)
    tensors, train_stability = local_tensors(theta, loads)
    global_matrix = global_sensitivity_matrix(tensors)
    train_preprocess_seconds = time.perf_counter() - start
    all_pairs = pair_indices(len(theta))
    sample_rng = np.random.default_rng(seed + 100_003)
    selected = sample_rng.choice(len(all_pairs[0]), min(cfg["fit_pair_count"], len(all_pairs[0])), replace=False)
    fit_pairs = (all_pairs[0][selected], all_pairs[1][selected])
    targets = profile_distances(profiles, fit_pairs)
    dev_theta, dev_loads = read_split(root / "data", manifest, seed, "development")
    dev_pairs = pair_indices(len(dev_theta))
    # Only development loads supply development reference distances.
    dev_oracle = profile_distances(response_profiles(dev_theta, dev_loads), dev_pairs)
    candidates = {}
    candidate_scores = {}
    fit_times = {}
    for ridge in cfg["mahalanobis_ridge_ratios"]:
        start = time.perf_counter()
        fitted, info = fit_mahalanobis(theta, targets, fit_pairs, mean, scale, ridge)
        fit_times[str(ridge)] = time.perf_counter() - start
        z_dev = (dev_theta - mean) / scale
        predicted = quadratic_distances(z_dev, fitted, dev_pairs)
        candidates[str(ridge)] = (fitted, info)
        candidate_scores[str(ridge)] = spearman(predicted, dev_oracle)
    # Ties go to the earlier, smaller ridge ratio in the fixed config order.
    selected_ridge = max(cfg["mahalanobis_ridge_ratios"], key=lambda r: candidate_scores[str(r)])
    selected_matrix, selected_info = candidates[str(selected_ridge)]
    dev_predictions, dev_timing = full_distances(
        dev_theta, loads, mean, scale, selected_matrix, global_matrix
    )
    dev_scores = score_predictions(dev_predictions, dev_oracle, len(dev_theta), cfg["neighbor_k"])
    # Label-permutation control retains pair covariates and the selected regularizer.
    permutation = np.random.default_rng(seed + 200_003).permutation(len(targets))
    permuted_matrix, _ = fit_mahalanobis(
        theta, targets[permutation], fit_pairs, mean, scale, selected_ridge
    )
    return {
        "seed": seed,
        "mean": mean.tolist(), "scale": scale.tolist(),
        "global_sensitivity_matrix": global_matrix.tolist(),
        "mahalanobis_matrix": selected_matrix.tolist(),
        "permuted_target_matrix": permuted_matrix.tolist(),
        "selected_ridge_ratio": selected_ridge,
        "selected_mahalanobis_info": selected_info,
        "ridge_development_spearman": candidate_scores,
        "development_scores": dev_scores,
        "development_timing_seconds": dev_timing,
        "train_preprocess_seconds": train_preprocess_seconds,
        "mahalanobis_fit_seconds": fit_times,
        "train_stability": train_stability,
    }, dev_oracle, dev_theta


def develop(root: Path, cfg: dict) -> None:
    manifest = read_json(root / "manifests" / "splits.json")
    records = []
    for seed in cfg["seeds"]:
        record, _, _ = _train_fit(root, cfg, manifest, seed)
        records.append(record)
    average = {name: float(np.mean([r["development_scores"][name]["spearman"] for r in records]))
               for name in METHODS}
    selected_baseline = max(SIMPLE_BASELINES, key=lambda name: average[name])
    write_json(root / "results" / "development.json", {
        "seeds": records,
        "mean_spearman": average,
        "selected_simple_baseline": selected_baseline,
        "selection_rule": "highest mean development Spearman among simple baselines; fixed tuple order resolves ties",
    })


def freeze(root: Path) -> None:
    if not (root / "results" / "development.json").exists():
        raise FileNotFoundError("development results are required before protocol freeze")
    paths = list(LOCKED_CODE) + ["manifests/splits.json", "results/development.json"]
    write_json(root / "manifests" / "protocol_lock.json", {
        "status": "frozen_before_test_oracle_evaluation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {name: sha256(root / name) for name in paths},
    })


def verify_lock(root: Path) -> dict:
    lock = read_json(root / "manifests" / "protocol_lock.json")
    for name, expected in lock["sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen protocol file changed: {name}")
    return lock


def _node_bootstrap(records: list[dict], methods: tuple[str, ...], baseline: str, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n_rep = cfg["bootstrap_replicates"]
    ranking_samples = {name: np.empty(n_rep) for name in methods}
    retrieval_samples = {name: np.empty(n_rep) for name in methods}
    delta_samples = np.empty(n_rep)
    for b in range(n_rep):
        chosen_seeds = rng.integers(0, len(records), len(records))
        per_method_rank = {name: [] for name in methods}
        per_method_retrieval = {name: [] for name in methods}
        for idx in chosen_seeds:
            record = records[int(idx)]
            n = record["n_structures"]
            counts = np.bincount(rng.integers(0, n, n), minlength=n)
            i, j = pair_indices(n)
            pair_weight = counts[i] * counts[j]
            for name in methods:
                value = weighted_spearman(record["predictions"][name], record["oracle"], pair_weight)
                per_method_rank[name].append(value)
                per_method_retrieval[name].append(float(np.average(record["retrieval_by_structure"][name], weights=counts)))
        for name in methods:
            ranking_samples[name][b] = np.mean(per_method_rank[name])
            retrieval_samples[name][b] = np.mean(per_method_retrieval[name])
        delta_samples[b] = ranking_samples["local_jacobian_pullback"][b] - ranking_samples[baseline][b]
    interval = lambda a: [float(v) for v in np.quantile(a, [0.025, 0.975])]
    return {
        "method_spearman_95pct": {name: interval(ranking_samples[name]) for name in methods},
        "method_precision_at_k_95pct": {name: interval(retrieval_samples[name]) for name in methods},
        "local_minus_selected_baseline_spearman_95pct": interval(delta_samples),
        "resampling": "hierarchical: sample seeds, then structures within each seed; pair weight=count_i*count_j; weighted midranks",
    }


def _decision(mean_delta: float, ci: list[float], per_seed_delta: list[float], control_gap: float, cfg: dict) -> str:
    threshold = cfg["meaningful_spearman_gain"]
    if (mean_delta >= threshold and ci[0] > 0 and sum(d > 0 for d in per_seed_delta) >= 4
            and control_gap >= threshold):
        return "SUPPORTED"
    if ci[1] <= threshold:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def finalize(root: Path, cfg: dict) -> dict:
    verify_lock(root)
    if (root / "results" / "final.json").exists():
        raise FileExistsError("final evaluation already exists; use a clean copy to reproduce")
    manifest = read_json(root / "manifests" / "splits.json")
    development = read_json(root / "results" / "development.json")
    selected = development["selected_simple_baseline"]
    records = []
    summaries = []
    for model_record in development["seeds"]:
        seed = model_record["seed"]
        theta, test_loads = read_split(root / "data", manifest, seed, "test")
        _, train_loads = read_split(root / "data", manifest, seed, "train")
        mean = np.array(model_record["mean"])
        scale = np.array(model_record["scale"])
        fitted = np.array(model_record["mahalanobis_matrix"])
        global_matrix = np.array(model_record["global_sensitivity_matrix"])
        # Prediction uses test configurations and TRAIN loads, before the test oracle is computed.
        predictions, timing = full_distances(
            theta, train_loads, mean, scale, fitted, global_matrix,
            include_controls=True, control_seed=seed + 300_003,
        )
        pair = pair_indices(len(theta))
        z = (theta - mean) / scale
        predictions["permuted_target_mahalanobis"] = quadratic_distances(
            z, np.array(model_record["permuted_target_matrix"]), pair
        )
        # This is the only test-response computation. Its output is evaluation-only.
        oracle = profile_distances(response_profiles(theta, test_loads), pair)
        scores = score_predictions(predictions, oracle, len(theta), cfg["neighbor_k"])
        retrieval = {
            name: neighbor_precision(pred, oracle, len(theta), cfg["neighbor_k"])
            for name, pred in predictions.items()
        }
        np.savez(root / "results" / f"seed_{seed}_pairs.npz", oracle=oracle, **predictions)
        records.append({
            "seed": seed, "n_structures": len(theta), "oracle": oracle,
            "predictions": predictions, "retrieval_by_structure": retrieval,
        })
        summaries.append({"seed": seed, "scores": scores, "timing_seconds": timing})
    mean_scores = {
        name: {
            key: float(np.mean([s["scores"][name][key] for s in summaries]))
            for key in ("spearman", "precision_at_k")
        }
        for name in METHODS + CONTROLS
    }
    ci_methods = METHODS + ("scrambled_local_pullback",)
    intervals = _node_bootstrap(records, ci_methods, selected, cfg)
    per_seed_delta = [s["scores"]["local_jacobian_pullback"]["spearman"] - s["scores"][selected]["spearman"] for s in summaries]
    mean_delta = float(np.mean(per_seed_delta))
    control_gap = mean_scores["local_jacobian_pullback"]["spearman"] - mean_scores["scrambled_local_pullback"]["spearman"]
    decision = _decision(mean_delta, intervals["local_minus_selected_baseline_spearman_95pct"], per_seed_delta, control_gap, cfg)
    result = {
        "experiment_id": cfg["experiment_id"],
        "selected_simple_baseline": selected,
        "test_seed_summaries": summaries,
        "mean_scores": mean_scores,
        "uncertainty": intervals,
        "primary_mean_local_minus_baseline_spearman": mean_delta,
        "primary_per_seed_differences": per_seed_delta,
        "local_minus_scrambled_control_mean_spearman": control_gap,
        "decision": decision,
    }
    write_json(root / "results" / "final.json", result)
    return result
