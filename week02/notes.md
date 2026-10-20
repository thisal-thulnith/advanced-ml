# Week 02 – Linear Regression

## Key concepts
- **Variable types**: quantitative (continuous vs. discrete) and qualitative
  (nominal vs. ordinal).
- **EDA passes**: univariate → bivariate → multivariate → correlation analysis.
- **Multicollinearity**: when the `X` variables are correlated with each other,
  which makes individual coefficients unreliable.
- **Linear regression assumptions**: normally distributed variables, linear
  relationship between features and target, and low multicollinearity.
- **Regression metrics**: RMSE, MSE and MAE — MAE is in the target's own units
  and is less sensitive to outliers than RMSE/MSE.

## What I learnt
- Cleaning a dataset before modelling: reading with `na_values='?'`, dropping
  columns that are not usable as-is, `dropna()`, checking `duplicated()`, and
  coercing object columns with `pd.to_numeric(..., errors='coerce')`.
- Running univariate EDA with histograms, violin plots and box plots, and
  reading `skew()` to judge how far a variable is from normal.
- Using `regplot`, `pairplot` and a `heatmap` of `corr()` to inspect
  feature–target relationships and spot multicollinearity.
- Removing outliers with an IQR rule (`q1 - 1.5*iqr`, `q3 + 1.5*iqr`) applied
  across features, then re-plotting to confirm the effect.
- Fitting `LinearRegression`, reading `coef_` / `intercept_`, and tabulating
  coefficients against feature names to interpret the model.
- <!-- Add any further lecture takeaways here. -->

## Questions / things to revisit
- The California housing intercept is around `-3,254,735`, which is not
  meaningful on its own — worth revisiting how to interpret an intercept when
  features (e.g. longitude/latitude) are far from zero.
- Neither model scales or normalises the features before fitting — when does
  that start to matter for linear regression?
- `ocean_proximity` (housing) and `origin` / `cylinders` (auto-mpg) were dropped
  rather than encoded. Revisit how to encode categorical variables and whether
  they would improve the fit.
- <!-- Anything else unclear you want to come back to. -->

## Assignment

Two linear regression implementations, each following the full EDA → clean →
model → evaluate workflow.

### 1. `linear_regression.ipynb` — auto-mpg

Predicts `mpg` from `displacement`, `horsepower`, `weight`, `acceleration` and
`model year`. An 80/20 train/test split over 392 rows after cleaning
(313 train / 79 test).

| Metric | Value |
| --- | --- |
| RMSE | 3.25 |
| MSE  | 10.58 |
| MAE  | 2.51 |

### 2. `californiahousingdataset.ipynb` — California housing

Predicts `median_house_value` from the eight quantitative features, with IQR
outlier removal applied first and a 70/30 train/test split.

| Metric | Value |
| --- | --- |
| RMSE | 67,734.97 |
| MSE  | 4,588,026,420.86 |
| MAE  | 49,495.98 |

### Files

| File | Description |
| --- | --- |
| `linear_regression.ipynb` | auto-mpg linear regression |
| `californiahousingdataset.ipynb` | California housing linear regression |
| `auto-mpg.csv` | auto-mpg dataset |
| `housing.csv` | California housing dataset |
| `pyproject.toml` | dependencies for the week |

Run with `uv sync` (or install `pyproject.toml` into a virtualenv), then open
the notebooks from inside the `week02/` folder — both read their CSV by
relative path.
