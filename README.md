# Kaggle Tabular Playground Series -- January 2021

This is an autonomous run by my [auto-model-trainer](https://github.com/Hook12aaa/auto-model-trainer) plugin. I pointed it at the objective file and let it explore on its own. It worked through 3 architecture classes -- linear, tree_based, and neural_net -- across 8 experiments before converging on a winner.

The dataset is synthetic, generated with CTGAN, and it shows: there's almost no linear signal in the raw features (the strongest correlation with the target is only about |r| ~0.067). All the useful structure lives in non-linear interactions between features. That's exactly why LightGBM pulled away from the pack. On top of the 14 raw features I added 17 engineered ones -- row-level aggregates and interaction terms -- to give the tree models more to work with.

## Competition

- Task: regression, predict a continuous `target`
- Metric: RMSE (minimize)
- Data: 300K training rows, 200K test rows
- Link: https://www.kaggle.com/competitions/tabular-playground-series-jan-2021

## Results

- Public LB: **0.70308**
- Private LB: **0.70176**
- Best model: `exp_tree3` (LightGBM), CV RMSE **0.69901**
- Baseline: Ridge, CV RMSE 0.72155
- Improvement over baseline: **3.12%**

The ensemble step ended up identical to the best single model, since LightGBM dominated every other class and there was nothing worth blending in.

| Experiment | Class | Model | CV RMSE |
|---|---|---|---|
| exp_tree3 | tree_based | LGBMRegressor | 0.69901 |
| exp_tree2 | tree_based | LGBMRegressor | 0.69924 |
| exp_tree1 | tree_based | LGBMRegressor | 0.69951 |
| exp_nn1 | neural_net | MLPRegressor | 0.71609 |
| exp_lin3 | linear | Ridge | 0.72161 |
| exp_lin2 | linear | ElasticNet | 0.72169 |
| exp_lin1 | linear | Lasso | 0.72181 |
| exp_000 | linear | Ridge (baseline) | 0.72155 |

## Project Structure

```
objective.yaml      the objective I handed to auto-model-trainer
features.py          the 17 engineered features (row aggregates + interactions)
final-report.md      the full autonomous run report
submission.csv       the winning submission (exp_tree3)
.auto-trainer/       experiment tree, worktrees, and run state
```

## Usage

This was produced with my auto-model-trainer plugin. To reproduce:

```
/auto-train objective.yaml
```

The plugin handles data validation, feature engineering, baseline creation, experiment exploration, convergence detection, and the final report on its own. Plugin lives here: https://github.com/Hook12aaa/auto-model-trainer
