# Predicting Concrete Compressive Strength with Supervised Machine Learning

Course project for **ISEN 613: Engineering Data Analysis**, Texas A&M University.

## Overview

This project develops and compares supervised learning models to predict the compressive strength of high-performance concrete from its mix design, and separately builds classification models to determine whether a given mix meets a 40 MPa strength threshold.

**Dataset:** 1,030 concrete samples, 8 mix-design features (cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, curing age) and 1 continuous target (compressive strength in MPa). No missing values.

## Exploratory Data Analysis

- Strength is positively correlated with cement content (r = 0.50), superplasticizer (r = 0.37), and curing age (r = 0.33); negatively correlated with water content (r = -0.44).
- Water and superplasticizer are strongly negatively correlated (r = -0.66), consistent with superplasticizers being used as water-reducing admixtures.
- Strength rises with curing age up to roughly 91 days, then plateaus/declines.
- Scatter plots and residual analysis from an initial linear fit indicated a clearly non-linear relationship between strength and several predictors, motivating the modeling approach below.

## Part 1: Regression (Predicting Strength)

Four modeling approaches were compared using an 80/20 train-test split:

| Model | Test MSE |
|---|---|
| Multiple Linear Regression | 129.68 |
| **Polynomial Regression (4th-order, subset-selected)** | **46.75** |
| Lasso Regression | 129.89 |
| Ridge Regression | 129.89 |
| 3-Nearest Neighbors Regression | 71.28 |

**Approach:**
- 10-fold cross-validation over polynomial degrees 1–9 identified a 4th-degree multivariate polynomial as optimal, reducing cross-validated MSE from ~105 (linear) to ~41.
- Since a full 4th-degree expansion requires 33 coefficients, forward/backward stepwise subset selection (scored by adjusted R²) was used to select a 12-term model, retaining polynomial terms in `age`, `superplasticizer`, and `water`, plus linear terms in `cement`, `slag`, and `ash`. This model achieved a test MSE of 46.75 — a 64% reduction from the linear baseline.
- Lasso and Ridge regression (tuned via 5-fold CV) did not outperform the linear baseline, and did not shrink any coefficients to zero — indicating all 8 predictors contribute meaningfully and that the limitation is model flexibility (linearity), not variable redundancy.
- K-Nearest Neighbors regression (K tuned via cross-validation, optimal K=3) outperformed the linear models but underperformed the polynomial model, consistent with non-linear but locally smooth structure in the data.

## Part 2: Classification (Strength ≥ 40 MPa)

The continuous strength target was binarized (`1` if strength > 40 MPa, `0` otherwise; 63%/37% class split) and three classifiers were compared:

| Model | Test Accuracy |
|---|---|
| QDA (subset-selected, 5 features) | 81.07% |
| **Decision Tree (cost-complexity pruned via 10-fold CV)** | **91.26%** |
| Logistic Regression (subset-selected) | 83.50% |

**Approach:**
- QDA's normality assumption was checked informally via mean/median comparison per class and per feature; forward subset selection improved its test accuracy from 79.6% to 81.1% using 5 of 8 predictors (cement, slag, ash, water, age).
- A decision tree was grown to full depth and pruned using cost-complexity pruning with the penalty parameter selected via 10-fold cross-validation, yielding the best-performing classifier at 91.3% accuracy (84 terminal nodes).
- Logistic regression achieved 83.5% accuracy; forward subset selection did not improve on the full 8-predictor model, suggesting all predictors contribute some information despite two (`coarseagg`, `fineagg`) having high p-values in the fitted GLM.

## Key Trade-off: Accuracy vs. Interpretability

The best-performing models in both tasks (4th-degree polynomial regression; pruned decision tree) are also the least interpretable — the polynomial model requires 12 coefficients across multiple transformed terms, and the pruned tree has 84 terminal nodes. Logistic regression and low-order linear models remain the most interpretable options, at a meaningful cost in accuracy.

## Limitations

- The 4th-degree polynomial can behave erratically outside the observed data range (unbounded growth/decay at the extremes), and cannot capture non-polynomial relationships (e.g., logarithmic or exponential effects).
- QDA's normality assumption is only approximately satisfied by the data.
- The pruned decision tree, despite its accuracy, is not practically interpretable given its size.
- Deep learning approaches (e.g., feedforward neural networks) were identified as a promising direction for future work, given the clear non-linear structure in the data.

## Repository Structure

```
├── notebooks/
│   └── concrete_strength_analysis.ipynb   # Full EDA, regression, and classification pipeline
├── data/
│   └── concrete.csv
├── README.md
```

## Running the Notebook

The notebook uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, and the [ISLP](https://www.statlearning.com/) package (companion library to *An Introduction to Statistical Learning*). Install dependencies and run top to bottom; all models are fit and evaluated inline with the accompanying analysis.

## Notes

- This repository is shared for portfolio purposes.
- Feedback and questions welcome — feel free to open an issue.
