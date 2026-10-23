"""Deterministic, disjoint structural configurations and independent load banks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .mechanics import B, EDGES, N_DOF

SPLITS = ("train", "development", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(a.tobytes()).hexdigest()


def generate_seed(seed: int, cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """Six separately spawned RNG streams per seed prevent split/load overlap."""
    streams = np.random.SeedSequence(seed).spawn(2 * len(SPLITS))
    base = np.log(np.asarray(cfg["base_stiffness"], dtype=float))
    load_sd = np.asarray(cfg["load_node_sd"], dtype=float)
    result = {}
    for split_idx, split in enumerate(SPLITS):
        struct_rng = np.random.default_rng(streams[2 * split_idx])
        load_rng = np.random.default_rng(streams[2 * split_idx + 1])
        n_struct = cfg["n_structures"][split]
        n_load = cfg["n_loads"][split]
        common = struct_rng.normal(0, cfg["log_stiffness_global_sd"], (n_struct, 1))
        edge = struct_rng.normal(0, cfg["log_stiffness_edge_sd"], (n_struct, len(base)))
        theta = base[None, :] + common + edge
        loads = load_rng.normal(size=(n_load, N_DOF)) * load_sd[None, :]
        loads += load_rng.normal(0, cfg["load_common_sd"], (n_load, 1))
        result[split] = {"theta": theta.astype("<f8"), "loads": loads.astype("<f8")}
    return result


def write_dataset(cfg: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "units": {"stiffness": "arbitrary consistent force/displacement", "load": "force", "response": "displacement"},
        "dof": N_DOF,
        "edges": [list(edge) for edge in EDGES],
        "incidence_rank": int(np.linalg.matrix_rank(B)),
        "splits": {},
    }
    all_theta_hashes: set[str] = set()
    all_load_hashes: set[str] = set()
    for seed in cfg["seeds"]:
        for split, arrays in generate_seed(seed, cfg).items():
            path = out / f"seed_{seed}_{split}.npz"
            np.savez(path, **arrays)
            theta_hashes = [array_hash(row) for row in arrays["theta"]]
            load_hashes = [array_hash(row) for row in arrays["loads"]]
            if set(theta_hashes) & all_theta_hashes or set(load_hashes) & all_load_hashes:
                raise AssertionError("cross-split or cross-seed duplicate structure/load")
            all_theta_hashes.update(theta_hashes)
            all_load_hashes.update(load_hashes)
            manifest["splits"][f"{seed}:{split}"] = {
                "file": path.name,
                "sha256": sha256(path),
                "structure_ids": [f"s{seed}_{split}_{i:03d}" for i in range(len(theta_hashes))],
                "structure_parameter_hashes": theta_hashes,
                "load_ids": [f"l{seed}_{split}_{i:03d}" for i in range(len(load_hashes))],
                "load_vector_hashes": load_hashes,
            }
    return manifest


def read_split(data_dir: Path, manifest: dict, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    item = manifest["splits"][f"{seed}:{split}"]
    path = data_dir / item["file"]
    if sha256(path) != item["sha256"]:
        raise ValueError(f"dataset hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as dataset:
        return dataset["theta"].copy(), dataset["loads"].copy()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
