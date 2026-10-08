import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, lasso_path
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# train error vs cross-validated error for a few degrees, to show under/overfitting
ROLL = "IMT2024044"
V = int(sys.argv[1])
degs = [int(a) for a in sys.argv[2:]]
tr = pd.read_csv(f"data/{ROLL}_train_var{V}.csv")
X, y = tr.drop(columns="y").values, tr.y.values
folds = list(KFold(5, shuffle=True, random_state=0).split(X))
rows = []

for d in degs:
    P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    if V == 1:
        # lasso picks terms, plain refit (same recipe as train_var1.py)
        al = np.logspace(-0.8, -2.4, 20)
        err = np.zeros(len(al))
        for a_, b_ in folds:
            sc = StandardScaler().fit(P[a_]); Za, Zb = sc.transform(P[a_]), sc.transform(P[b_])
            mu = y[a_].mean()
            _, co, _ = lasso_path(Za, y[a_] - mu, alphas=al, max_iter=100000)
            for k in range(len(al)):
                s = np.flatnonzero(co[:, k])
                p = LinearRegression().fit(Za[:, s], y[a_]).predict(Zb[:, s]) if len(s) else np.full(len(b_), mu)
                err[k] += np.sum((p - y[b_]) ** 2)
        k = err.argmin()
        sc = StandardScaler().fit(P); Z = sc.transform(P)
        _, co, _ = lasso_path(Z, y - y.mean(), alphas=[al[k]], max_iter=100000)
        s = np.flatnonzero(co[:, 0])
        fit_mse = np.mean((LinearRegression().fit(Z[:, s], y).predict(Z[:, s]) - y) ** 2) if len(s) else y.var()
        rows.append(dict(degree=d, terms=P.shape[1], used=len(s), train_mse=fit_mse, cv_mse=err[k] / len(y)))
    else:
        al = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 1, 3]
        cv = {a: np.mean([np.mean((Ridge(alpha=a).fit(P[i], y[i]).predict(P[j]) - y[j]) ** 2) for i, j in folds]) for a in al}
        a = min(cv, key=cv.get)
        fit_mse = np.mean((Ridge(alpha=a).fit(P, y).predict(P) - y) ** 2)
        rows.append(dict(degree=d, terms=P.shape[1], used=P.shape[1], train_mse=fit_mse, cv_mse=cv[a]))
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(f"results/fit_gap_var{V}.csv", index=False)
