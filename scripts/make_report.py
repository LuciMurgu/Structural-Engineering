#!/usr/bin/env python3
"""Render frozen-test plots and the scientific report from saved result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bg_exp001.data import sha256
from bg_exp001.evaluation import METHODS, verify_lock


ROOT = Path(__file__).resolve().parents[1]


def fmt(number: float) -> str:
    return f"{number:.3f}"


def main() -> None:
    verify_lock(ROOT)
    cfg = json.loads((ROOT / "configs" / "exp001.json").read_text())
    development = json.loads((ROOT / "results" / "development.json").read_text())
    final = json.loads((ROOT / "results" / "final.json").read_text())
    reproduction_path = ROOT / "results" / "reproduction_check.json"
    reproduction_note = (
        "A fresh temporary-directory rerun passed 11 unit tests and five reconstructed monograph checks, "
        "then matched every split-file hash, the selected baseline, the primary estimate and interval, "
        "and the decision. See `results/reproduction_check.json`."
        if reproduction_path.exists() else
        "A fresh-directory reproduction check can be run with the commands in `README.md`."
    )
    plot_dir = ROOT / "plots"
    plot_dir.mkdir(exist_ok=True)
    seed = cfg["seeds"][0]
    with np.load(ROOT / "results" / f"seed_{seed}_pairs.npz", allow_pickle=False) as pair_data:
        oracle = pair_data["oracle"]
        rng = np.random.default_rng(819)
        keep = rng.choice(len(oracle), min(750, len(oracle)), replace=False)
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
        for ax, name in zip(axes.ravel(), METHODS):
            predicted = pair_data[name]
            ax.scatter(oracle[keep], predicted[keep], s=11, alpha=0.27, linewidths=0)
            rho = next(s for s in final["test_seed_summaries"] if s["seed"] == seed)["scores"][name]["spearman"]
            ax.set_title(f"{name.replace('_', ' ')}\nSpearman = {rho:.3f}")
            ax.set_xlabel("Oracle distance: independent test loads")
            ax.set_ylabel("Predicted distance")
            ax.grid(alpha=0.2)
        fig.suptitle(f"Held-out structural pairs, seed {seed}; 750 presampled pairs", fontsize=14)
        fig.savefig(plot_dir / "predicted_vs_oracle.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    names = list(METHODS)
    means = [final["mean_scores"][name]["spearman"] for name in names]
    intervals = [final["uncertainty"]["method_spearman_95pct"][name] for name in names]
    lower = [m - ci[0] for m, ci in zip(means, intervals)]
    upper = [ci[1] - m for m, ci in zip(means, intervals)]
    x = np.arange(len(names))
    ax.bar(x, means, color=["#72839a", "#72839a", "#72839a", "#72839a", "#49a0a4", "#e79c4b"])
    ax.errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor="#1f2937", capsize=4)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Mean test-pair Spearman correlation")
    ax.set_title("Five seeds; 95% hierarchical structure-bootstrap intervals")
    ax.set_ylim(min(0, min(ci[0] for ci in intervals) - 0.05), min(1.02, max(ci[1] for ci in intervals) + 0.05))
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(plot_dir / "method_comparison.png", dpi=160)
    plt.close(fig)

    baseline = final["selected_simple_baseline"]
    ci = final["uncertainty"]["local_minus_selected_baseline_spearman_95pct"]
    gain = final["primary_mean_local_minus_baseline_spearman"]
    method_rows = []
    for name in METHODS + ("scrambled_response_profile", "scrambled_local_pullback", "permuted_target_mahalanobis"):
        score = final["mean_scores"][name]
        ci_text = final["uncertainty"]["method_spearman_95pct"].get(name)
        interval = f"[{fmt(ci_text[0])}, {fmt(ci_text[1])}]" if ci_text else "—"
        method_rows.append(f"| {name} | {fmt(score['spearman'])} | {interval} | {fmt(score['precision_at_k'])} |")
    seed_rows = []
    for summary, difference in zip(final["test_seed_summaries"], final["primary_per_seed_differences"]):
        seed_rows.append(
            f"| {summary['seed']} | {fmt(summary['scores']['local_jacobian_pullback']['spearman'])} "
            f"| {fmt(summary['scores'][baseline]['spearman'])} | {fmt(difference)} |"
        )
    time_rows = []
    for name in METHODS:
        samples = [s["timing_seconds"][name] for s in final["test_seed_summaries"]]
        time_rows.append(f"| {name} | {np.median(samples)*1000:.1f} |")
    train_prepare_ms = np.median([s["train_preprocess_seconds"] for s in development["seeds"]]) * 1000
    ridge_search_ms = np.median([sum(s["mahalanobis_fit_seconds"].values()) for s in development["seeds"]]) * 1000
    stability = [s["timing_seconds"]["stability"] for s in final["test_seed_summaries"]]
    max_condition = max(s["max_stiffness_condition"] for s in stability)
    min_eigenvalue = min(s["min_stiffness_eigenvalue"] for s in stability)
    min_tensor_eigenvalue = min(s["min_tensor_eigenvalue"] for s in stability)
    nonfinite = sum(sum(v["nonfinite_count"] for v in s["scores"].values()) for s in final["test_seed_summaries"])
    original_pdf_hash = "fcea08ff2ad755782cb783a0023726a8b75870099b0d59296c17bddb9971fef4"
    text = f"""# EXP-001 Scientific Report — Behavioral Geometry on Spring Networks

**Decision: {final['decision']}.** The prespecified primary comparison of the local Jacobian pullback against the development-selected strongest simple baseline, **{baseline}**, gave mean test Spearman difference **{gain:+.3f}**, with a hierarchical 95% interval **[{ci[0]:+.3f}, {ci[1]:+.3f}]**. The meaningful-gain threshold was +{cfg['meaningful_spearman_gain']:.2f}. This decision follows the frozen rule in `manifests/PROTOCOL.md`.

## Question and design

We tested whether a response-sensitivity-induced local geometry predicts independent-load structural behavior better than matched simple distances. This is a controlled simulator experiment, not evidence about cells, learned embeddings, topology extrapolation, or real structures. Five independent generator seeds each produced 120 training, 50 development, and 60 test stiffness configurations, with 24/32/128 independently sampled load cases. The topology and load distribution were fixed. Every configuration is a six-DOF constrained scalar spring network with eleven positive springs.

The final oracle was the root-mean-square displacement difference under **test loads only**. Predictions on test configurations used the training load bank; training and development responses supplied fitting and selection. The final runner verified a hash lock on configuration, source, splits, development results, and protocol before it computed any test response. No test response was used for fitting, feature choice, or hyperparameter choice. The selected simple comparator was chosen by mean development Spearman among raw Euclidean, standardized Euclidean, fitted global Mahalanobis, global sensitivity, and direct response profile.

## Mathematical checks and numerical validation

The assembly identity `K=B.T diag(exp(theta)) B` and derivative `du/dtheta_e=-K^-1 (exp(theta_e) b_e b_e.T)u` are exact for this specified linear spring model. `M(theta)=mean_train_load J.T J` is positive semidefinite by construction. The endpoint-averaged quadratic pullback is a local first-order dissimilarity; it is **not** claimed to be a global metric or exact finite response distance. Unit tests compare the analytic derivatives with central differences, test energy assembly, detect singular and indefinite operators, check split integrity, and verify the local first-order approximation. `monograph_checks.py` independently reconstructs five checks described in Appendix B. The separately named companion script was unavailable; these reconstructed checks are not represented as an inspection of its code.

## Held-out results

All Spearman scores use every unordered pair of the 60 test structures per seed. Confidence intervals resample seeds and structures; pairs sharing a structure receive dependent weights instead of being treated as independent observations. Precision@5 is the mean fraction of oracle neighbors recovered among the five nearest structures.

| Method | Mean Spearman | 95% interval | Precision@5 |
|---|---:|---:|---:|
{chr(10).join(method_rows)}

The selected baseline comparison by seed was:

| Seed | Local Spearman | Baseline Spearman | Local minus baseline |
|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

The primary interval is based on {cfg['bootstrap_replicates']} hierarchical bootstrap replicates with fixed bootstrap seed {cfg['bootstrap_seed']}. The local-minus-scrambled-coordinate control mean Spearman gap was {fmt(final['local_minus_scrambled_control_mean_spearman'])}. Coordinate scrambling preserves each structure's stiffness multiset but breaks spring-edge identity; permuting training response-distance labels probes whether global fitting benefits from the correct targets. These are diagnostic controls and do not establish causal superiority by themselves.

![Predicted versus oracle behavioral distances](../plots/predicted_vs_oracle.png)

![Method comparison with uncertainty](../plots/method_comparison.png)

## Numerical stability and cost

There were {nonfinite} nonfinite predicted distances across methods and seeds. The smallest test stiffness eigenvalue was {min_eigenvalue:.4g}; the largest spectral condition number was {max_condition:.2f}. The smallest computed local tensor eigenvalue was {min_tensor_eigenvalue:.4g}. `checked_solve` rejects nonsymmetric, singular, near-singular, and indefinite operators. The generator produced no rejected configuration.

Median per-seed test prediction time, including a method's test-structure preprocessing and all pair distances, measured on this machine:

| Method | Median milliseconds per seed |
|---|---:|
{chr(10).join(time_rows)}

Training response profiles and Jacobian tensors shared by the trained methods took a median {train_prepare_ms:.1f} ms per seed. Fitting all five Mahalanobis ridge candidates took a median {ridge_search_ms:.1f} ms per seed, excluding development scoring. Timing is descriptive and can vary by machine and BLAS configuration. It is not part of the scientific decision rule.

## Interpretation and limits

The direct response-profile baseline is especially strong in this setting because the exact simulator is available and the training and test load banks come from the same distribution. For this linear model, with `T=K_a^-1-K_b^-1`, squared profile distance is exactly `tr(T C T.T)`, where `C` is the empirical mean of `f f.T` over the chosen load bank. Training and test banks have different empirical `C`, but estimate the same load distribution. The local Jacobian uses an exact derivative but approximates finite differences between distinct structures; the scatter plot shows larger deviations for widely separated structures. Better local approximation for infinitesimal changes is a mathematical statement; better held-out ranking or retrieval is an empirical hypothesis adjudicated above. This benchmark has one fixed network topology, one load family, arbitrary consistent units, only five generator seeds, and no measurement noise or learned response-model error. The reported uncertainty is conditional on this generator and cannot support broad claims about physical structures or intervention embeddings.

## Reproducibility and provenance

- Run the commands in `README.md` from a clean copy. Code, configuration, split manifest, protocol lock, pairwise distances, development selection, and final scores are stored here.
- {reproduction_note}
- Environment used for the recorded run: Python/NumPy/Matplotlib/Pytest versions are listed in `results/environment.json`; set `OPENBLAS_NUM_THREADS=1` for comparable timing.
- Original monograph: *Geometry That Understands Behavior*, version 1.0, 23 October 2026; PDF SHA-256 `{original_pdf_hash}`. The extracted text retained in `references/` has SHA-256 `{sha256(ROOT / 'references' / 'Geometry_That_Understands_Behavior_extracted.txt')}`. The PDF disappeared from the source directory after extraction, and the named companion script was not found locally or through the connected file search.
"""
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "EXP_001_SCIENTIFIC_REPORT.md").write_text(text)
    print(f"Wrote report and plots; decision: {final['decision']}")


if __name__ == "__main__":
    main()
