# BG-EXP-001 frozen protocol definition

This file and the implementation are hashed in `protocol_lock.json` after development selection and before any test response is computed. The test runner refuses changed files.

## Physical system and data

Six free scalar displacement coordinates are joined by eleven positive linear springs. Ground is fixed; the incidence matrix in `mechanics.py` has rank six. Stiffness is `K(theta)=B.T diag(exp(theta)) B`, force and displacement have consistent arbitrary units, and output is the six-coordinate static displacement under a load. All operators must be symmetric positive definite to solve. Structures vary only in their eleven log stiffnesses, on a fixed topology. This tests new stiffness configurations on an observed topology, not topology extrapolation.

Five independent top-level seeds generate train, development, and test structural configurations and load banks from six separately spawned streams per seed. Structure counts are 120/50/60 and load counts are 24/32/128. A structure has one log stiffness vector. Load vectors are sampled independently per split from the same zero-mean family with node-specific and shared components. Hashes and IDs in `splits.json` demonstrate disjoint configurations and load vectors. The target is interpolation to independent loads from this family, not distribution shift.

## Information boundary and methods

Training loads and training structures may fit standardization and a supervised global Mahalanobis matrix. For predictions on development and test structures, only the **training load bank** is used. Development responses under development loads select one ridge ratio for Mahalanobis per seed and the strongest simple baseline by mean development Spearman. Test responses under test loads are computed only in `finalize`, after the verified protocol lock, exclusively to define the oracle and score predictions. Test load responses never fit metrics or select methods, features, or hyperparameters.

The oracle for structures a,b is `sqrt(mean_s ||u_a(f_s)-u_b(f_s)||_2^2)` over the 128 test loads. Predictions use:

1. Raw Euclidean distance between log stiffness vectors.
2. Euclidean distance after training-only coordinate standardization.
3. A full PSD global Mahalanobis matrix, ridge fitted to squared training response distances over 4,096 presampled training pairs, PSD projected, with ridge selected by development loads.
4. A global sensitivity matrix equal to the training-structure average of the exact Jacobian pullback tensor, using training loads.
5. Direct response-profile distance over training loads.
6. The local Jacobian pullback **dissimilarity** `sqrt(delta_theta.T (M_a+M_b)/2 delta_theta)`, with `M_i=mean_train_load J_i.T J_i`. This symmetric nonnegative construction need not satisfy the triangle inequality; no geodesic claim is made.

The direct profile and global sensitivity controls use the same known physics and training load bank as the local method. No learned surrogate is used. Thus the study isolates whether local first-order geometry helps this exact simulator task; it does not test neural model errors.

## Evaluation and decision

Primary outcome: mean across five seed-specific Spearman correlations over all unordered pairs of held-out structures. The primary comparison is local pullback minus the strongest simple baseline selected by mean development Spearman from methods 1–5. A meaningful gain is at least +0.02 Spearman.

Secondary outcomes: oracle top-5 neighbor overlap (precision@5), all method Spearman scores, per-seed stability, minimum operator eigenvalue, maximum condition number, nonfinite distance count, and wall-clock computation time. Pairwise observations share structures. A 600-replicate hierarchical bootstrap samples seeds with replacement, then test structures with replacement within each selected seed; a pair receives weight equal to the product of the two structure counts. Weighted Spearman recomputes weighted midranks. Retrieval uncertainty resamples structure anchors in the same hierarchy. Report 2.5/97.5 percentiles for each method and the paired primary difference. These intervals describe variation over the five seeds and sampled structures, conditional on the declared data generator and load family; five seeds limit precision.

Negative controls: independently permute stiffness coordinates inside each test structure before applying direct profile and local pullback methods; permute squared response-distance labels over training pairs before fitting a matched global Mahalanobis matrix. Coordinate scrambling preserves each structure's stiffness value multiset but breaks edge identity. These controls test semantic mapping and label fit, not every possible confound.

Decision rule, fixed before final outcomes:

- `SUPPORTED` if the primary mean gain is at least +0.02, the 95% interval lower bound exceeds zero, at least four of five seed differences are positive, and the local method exceeds its scrambled-coordinate control by at least +0.02 mean Spearman.
- `NOT_SUPPORTED` if the primary interval upper bound is at most +0.02.
- `INCONCLUSIVE` otherwise.

No deep learning, fractional operators, or geodesic optimization are in scope. Computation time is reported as a tradeoff, not a decision gate.
