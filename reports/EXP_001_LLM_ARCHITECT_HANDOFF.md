# BG-EXP-001 — Handoff to the LLM Architect

**Repository:** [LuciMurgu/Structural-Engineering](https://github.com/LuciMurgu/Structural-Engineering)

**Frozen experiment snapshot:** [`9d5f8ac`](https://github.com/LuciMurgu/Structural-Engineering/tree/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8)
**Scientific decision:** **NOT_SUPPORTED** for a local Jacobian pullback advantage over the strongest simpler baseline in this bounded spring-network experiment.

## Read these Git files first

1. [Full scientific report](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/reports/EXP_001_SCIENTIFIC_REPORT.md) — complete methods, results, uncertainty, cost, interpretation, and limits.
2. [Prespecified protocol](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/manifests/PROTOCOL.md) and [protocol hash lock](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/manifests/protocol_lock.json) — the selection and decision rules frozen before final-test responses were examined.
3. [Final results](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/results/final.json), [development selection](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/results/development.json), and [split manifest](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/manifests/splits.json) — numerical evidence and input provenance.
4. [Predicted-versus-oracle plot](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/plots/predicted_vs_oracle.png) and [method comparison plot](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/plots/method_comparison.png) — visual diagnostics.

The [README](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/README.md) has commands to reproduce the benchmark. The exact [configuration](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/configs/exp001.json), [mechanics](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/src/bg_exp001/mechanics.py), [metric implementations](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/src/bg_exp001/metrics.py), [evaluation and bootstrap code](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/src/bg_exp001/evaluation.py), and [tests](https://github.com/LuciMurgu/Structural-Engineering/tree/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/tests) are available for audit. Saved pairwise predictions and oracle distances are in the [results directory](https://github.com/LuciMurgu/Structural-Engineering/tree/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/results).

## Research question and controlled system

The question was whether a response-sensitivity-induced geometry ranks held-out structural pairs by behavior more accurately than simpler matched distances. Each object is a **six-degree-of-freedom constrained scalar spring network** with a fixed topology and eleven positive springs. Log stiffness coordinates `theta` define `K(theta)=B.T diag(exp(theta)) B`; the simulator solves `K(theta) u=f`. The six displacements are the observable response. The experiment uses exact known mechanics, no deep learning, fractional operators, or geodesic optimization.

Five independent seeds each generated 120 training, 50 development, and 60 test stiffness configurations, with 24, 32, and 128 independently sampled load cases. Structure and load identities are disjoint across splits. Methods fit or use training data and the training load bank; development responses choose the global Mahalanobis regularizer and the strongest simple comparator. **Only test-load responses define the final oracle.** The code verifies the pretest hash lock before that calculation.

The comparisons are raw Euclidean, training-standardized Euclidean, fitted PSD global Mahalanobis, global mean sensitivity, direct response profile on training loads, and an endpoint-averaged local Jacobian pullback. The direct profile uses the **same exact simulator and training load bank** as the local construction. It is therefore the essential simple baseline. The local endpoint average is a symmetric nonnegative **dissimilarity**; triangle inequality was not established.

## Held-out evidence

| Method | Mean test-pair Spearman | 95% structure-aware interval | Oracle top-5 neighbor precision |
|---|---:|---:|---:|
| Raw Euclidean | 0.469 | [0.412, 0.529] | 0.294 |
| Standardized Euclidean | 0.468 | [0.406, 0.532] | 0.286 |
| Fitted global Mahalanobis | 0.761 | [0.703, 0.808] | 0.381 |
| Global sensitivity | 0.886 | [0.868, 0.907] | 0.692 |
| **Direct response profile** | **0.998** | **[0.997, 0.999]** | **0.914** |
| Local Jacobian pullback | 0.987 | [0.982, 0.990] | 0.824 |

The development-selected comparator was **direct response profile**. The prespecified primary difference, local minus comparator, was **−0.01146 Spearman**, with a **95% hierarchical interval [−0.01573, −0.00877]**. All five seed-specific differences were negative. The practical-support threshold had been set at **+0.02**. Under the frozen rule, this is **NOT_SUPPORTED**, rather than an inconclusive positive trend.

Uncertainty was estimated with 600 hierarchical bootstrap replicates: seeds were resampled, then test structures within seeds; pair weights accounted for shared structures. The seed is the independent load-bank replication unit. These intervals are conditional on this generator and only five seeds.

Two semantic controls behaved as expected: scrambling stiffness-to-edge coordinates reduced direct-profile Spearman to 0.403 and local-pullback Spearman to 0.407; permuting training pair response-distance targets reduced fitted Mahalanobis Spearman to 0.416. They support the evaluation pipeline's sensitivity to correct structure/response correspondence, but do not make the local method superior.

## Mathematics versus the empirical claim

For this model, `K=B.T diag(exp(theta)) B` and `du/dtheta_e=-K^-1 (exp(theta_e) b_e b_e.T)u` are **exact identities**. The analytic derivative passed central finite-difference checks. `M(theta)=mean_load J.T J` is positive semidefinite by construction. Its quadratic form describes first-order output change near one structure; it does not equal the finite response difference between arbitrary structures.

The direct response-profile distance is an exact finite response comparison on its chosen load bank. For a pair with `T=K_a^-1-K_b^-1`, its square is `tr(T C T.T)`, with `C=mean(f f.T)` for that bank. Training and test banks independently estimate the same load distribution, explaining why the simple direct baseline was so accurate here. The scatter plot shows the local first-order construction deviating most for widely separated pairs. Those observations are empirical; they are not a theorem that local geometry is generally inferior.

## Validation, cost, and scope limits

The test suite passed **11 unit tests**, including energy assembly, singular/indefinite detection, analytic-versus-central-difference sensitivity, split integrity, and first-order behavior. Five Appendix B examples from the source monograph were independently reconstructed and passed. A fresh temporary-directory run reproduced every split-file hash, the selected baseline, the primary estimate and interval, and the decision. See [reproduction check](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/results/reproduction_check.json).

No predicted distance was nonfinite. The smallest test stiffness eigenvalue was 0.06897; the largest spectral condition number was 70.46. Median per-seed test prediction time was **3.8 ms** for direct profile and **5.0 ms** for local pullback on the recorded machine; these timings are descriptive.

The result covers **one fixed topology, one load family, exact simulator access, no measurement noise, and no learned response-model error**. It tests new stiffness configurations under independently sampled loads from an observed family, not new topology or load-family extrapolation. It does not establish a result for cellular perturbations, VCC submissions, or learned embeddings.

## Recommendation to the Architect

Treat direct response-profile distance as the benchmark to beat whenever a response model or simulator is available. Do **not** claim that this experiment validates curvature, a learned local metric, or a cross-domain architecture. If proposing another experiment, specify a regime in which direct profiling is limited (for example sparse or costly response measurements, an independently shifted load family, or an imperfect surrogate), freeze its oracle and comparison rule before seeing its test outcomes, and retain the same direct-profile control. The current negative result is a useful design constraint, not a reason to tune the final test.

## Source provenance

The read source was *Geometry That Understands Behavior*, version 1.0, 23 October 2026. Its [extracted text](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/references/Geometry_That_Understands_Behavior_extracted.txt) is stored in Git; the original PDF SHA-256 is recorded in the report. The separately named `behavioral_metric_demo.py` was unavailable. [Monograph checks](https://github.com/LuciMurgu/Structural-Engineering/blob/9d5f8acd2ec20bd94bbfccee0dcbd77de840f1a8/scripts/monograph_checks.py) are a reconstruction from Appendix B, not the missing original script.
