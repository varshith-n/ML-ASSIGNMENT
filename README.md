# Polynomial regression assignment

Two problems, two polynomial models. Everything here can be rerun from the repo root.

| problem | model | CV MSE | hold-out MSE / R2 |
|---|---|---|---|
| var1 (6 features) | degree 5, lasso picks terms, plain least squares refit (70 terms) | 0.323 | 0.326 / 0.964 |
| var2 (3 features) | degree 12, ridge alpha = 0.2 (454 terms) | 0.253 | 0.201 / 0.995 |

## Files

- `degree_sweep.py` - 5-fold CV over polynomial degree (`python degree_sweep.py 1 8`, `python degree_sweep.py 2 20`), writes `results/sweep_var*.csv`
- `train_var1.py` - picks the lasso penalty by repeated CV, fits, writes `predictions/IMT2024044_pred_var1.csv`
- `train_var2.py` - picks degree and ridge penalty by repeated CV, fits, writes `predictions/IMT2024044_pred_var2.csv`
- `fit_gap.py` - train error vs CV error per degree (the under/overfitting table in the report), e.g. `python fit_gap.py 2 2 4 6 8 10 12 16 20`
- `experiments/` - the side runs mentioned in the report (lasso/ridge sweeps, OMP, backward elimination, Legendre basis, paired degree comparison); `alt.py` has a very slow backward-elimination loop
- `Polynomial_Regression_Report.pdf` - the report
- `shift_check.py` - re-scores var1 candidates with weights that mimic the test set (see report)
- `make_figures.py` - plots used in the report
- `data/` - the four csv files, `predictions/` - the two submission files

## Run

```
pip install -r requirements.txt
python degree_sweep.py 1 8
python degree_sweep.py 2 20
python train_var1.py
python train_var2.py
python shift_check.py
python fit_gap.py 1 2 3 4 5 6
python fit_gap.py 2 2 4 6 8 10 12 16 20
python make_figures.py
```

Prediction files have a single `y` column in the same row order as the test file, like the sample submission.
