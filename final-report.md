# Auto-Train Final Report — Kaggle TPS January 2021

**Status: CONVERGED** · Winner: `exp_tree3` · Submission: `.auto-trainer/submission.csv`

---

## Section 1 — Objective Recap
- **Dataset:** ./train.csv (300,000 rows) · ./test.csv (200,000 rows)
- **Target:** `target` (continuous regression) · **Metric:** RMSE (minimize)
- **Submission:** id_column=`id`, prediction_column=`target`
- **Constraints:** max_iterations=15, architecture_classes_minimum=3

## Section 2 — Data Quality Summary (data-quality-report.json: status=PASS, all 8 checks PASS)
- Shape: 300000 × 14 (samples/feature=21428.57)
- Missing: 0 cols · Duplicates: 0 · Outliers flagged: 31 (0.01%)
- Target variance: 0.5374 · Zero-variance cols: [] · Redundant pairs: [] · Leakage suspects: []
- Domain: synthetic CTGAN-generated; linear signal near-zero (max |r|~0.067) so signal lives in non-linear interactions.
- **Feature engineering:** 17 engineered features (row aggregates + interaction/collinear-cluster terms) → 31 total model features (14 raw + 17 engineered), verified row-count-preserving and leakage-free.

## Section 3 — Exploration Summary
- Experiments: 8 exploration nodes across **3 architecture classes** (linear, tree_based, neural_net) + 1 post-convergence ensemble
- Rounds: 4 — R1 divergence (tree_based+neural_net+linear), R2–R3 depth refinement, R4 stability confirmation
- Max depth: 3

| exp_id | class | model | depth | val_rmse | params | eval | review |
|---|---|---|---|---|---|---|---|
| exp_000 | linear | Ridge | 0 | 0.72155 | 32 | BASELINE | - |
| exp_tree1 | tree_based | LGBMRegressor | 1 | 0.69951 | 9300 | ACCEPT | KEEP_WITH_CONCERNS |
| exp_tree2 | tree_based | LGBMRegressor | 2 | 0.69924 | 9300 | ACCEPT | KEEP_WITH_CONCERNS |
| exp_tree3 | tree_based | LGBMRegressor | 3 | 0.69901 | 9300 | ACCEPT | KEEP_WITH_CONCERNS |
| exp_nn1 | neural_net | MLPRegressor | 1 | 0.71609 | 12417 | REJECT | - |
| exp_lin1 | linear | Lasso | 1 | 0.72181 | 32 | INCONCLUSIVE | - |
| exp_lin2 | linear | ElasticNet | 2 | 0.72169 | 32 | INCONCLUSIVE | - |
| exp_lin3 | linear | Ridge | 3 | 0.72161 | 32 | INCONCLUSIVE | - |
| exp_ensemble | ensemble | CaruanaBlend | 4 | 0.69901 | 9300 | ACCEPT | - |

## Section 4 — Pareto Front Evolution (objective: minimize rmse & params)
| step | pareto_front |
|---|---|
| snapshot 0 | ['exp_000'] |
| snapshot 1 | ['exp_000', 'exp_tree1'] |
| snapshot 2 | ['exp_000', 'exp_tree2'] |
| snapshot 3 | ['exp_000', 'exp_tree3'] |
| snapshot 4 | ['exp_000', 'exp_tree3'] |
| snapshot 5 | ['exp_000', 'exp_tree3'] |

Final Pareto front: **['exp_000', 'exp_tree3']** — the low-param linear baseline and the low-RMSE LightGBM winner form the non-dominated trade-off.

## Section 5 — Winner Analysis: `exp_tree3`
- **Model:** LGBMRegressor (class `tree_based`) — n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=80, feature_fraction=0.7
- **Validation RMSE:** 0.699012 (MAE 0.58674, R² 0.08379)
- **Baseline (exp_000 Ridge) RMSE:** 0.721553
- **Improvement:** 0.02254 absolute / **3.12%** relative
- **trainable_params:** 9300 · config_hash 3e707c633be2… · sha 7554da0ea4ea…
- **Lineage:** exp_000 → exp_tree1 → exp_tree2 → exp_tree3
- **Independent reviewer:** re-ran run.sh from scratch → reproduced RMSE 0.699012 exactly (abs diff 0.0, deterministic seed=42).

## Section 6 — Runner-up Comparison (top 3 exploration nodes by RMSE)
| exp_id | class | val_rmse | params |
|---|---|---|---|
| exp_tree3 | tree_based | 0.69901 | 9300 |
| exp_tree2 | tree_based | 0.69924 | 9300 |
| exp_tree1 | tree_based | 0.69951 | 9300 |

Tree-class variants dominate linear/NN on this interaction-driven signal; winner = lowest-RMSE node on the stable Pareto front.

## Section 7 — Two-Tier Convergence Evidence
- **Tier 1 (check_class_exhaustion.py):**
  - linear → EXHAUSTED (diminishing returns, depth≥2: 0.72155→0.72181→0.72169, all <1%)
  - tree_based → EXHAUSTED (diminishing returns, depths 1→2→3: 0.69951→0.69924→0.69901, all <1%)
  - neural_net → EXHAUSTED (Pareto-dominated by tree_based: 0.71609 RMSE at 12417 params vs tree 0.69901 at 9300)
- **Tier 2 (check_cross_class_coverage.py):** explored_classes=3 ≥ minimum 3 ✓ · exploring_classes=[] ✓ · Pareto front stable 2 consecutive rounds ✓ → **CONVERGED**
- The Caruana ensemble node is post-convergence (status ENSEMBLE, excluded from exploration accounting).

## Section 8 — Integrity Summary
- **Merkle chain (verify_merkle_chain.py):** valid=True, nodes_checked=9, mismatches=[].
- **Independent winner re-eval:** metric reproduced exactly.
- **Ensemble (caruana_ensemble.py):** greedy selection over Pareto front chose only `exp_tree3` @ weight 1.0; ensemble_score 0.699012 == best_single 0.699012; beats_best_single=False → **DONE_WITH_CONCERNS (tie)**. The weak linear member added no blend value; recommended solution is the single winner.
- No discrepancies flagged.

## Section 9 — Reproducibility
- **Winner worktree:** .auto-trainer/worktrees/exp_tree3/
- **Reproduce:** `cd .auto-trainer/worktrees/exp_tree3 && bash run.sh`  (preflight → train → eval; LightGBM seed=42, 80/20 split random_state=42, deterministic)
- **Shared modules:** .auto-trainer/features.py (locked, hashed in feature-manifest.json) and .auto-trainer/harness.py (single fixed split shared by all variants → aligned val predictions for ensembling)
- **Submission:** .auto-trainer/submission.csv — 200,000 rows, columns [id, target], from the winner's eval.py on test.csv.

---
*Every metric traces to an executed script (evaluate_layers.py, compute_pareto.py, check_class_exhaustion.py, check_cross_class_coverage.py, verify_merkle_chain.py, caruana_ensemble.py) or a variant eval.py run — none from memory.*
