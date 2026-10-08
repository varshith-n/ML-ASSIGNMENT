import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, lasso_path
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024044"
V = int(sys.argv[1])
MAXD = int(sys.argv[2])

tr = pd.read_csv(f"data/{ROLL}_train_var{V}.csv")
X = tr.drop(columns="y").values
y = tr.y.values
folds = list(KFold(5, shuffle=True, random_state=0).split(X))
alphas = [1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10]
lasso_al = np.logspace(-0.5, -2.6, 25)  # already descending

rows = []
for d in range(1, MAXD + 1):
    P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    p = P.shape[1]
    ols, rdg = [], {a: [] for a in alphas}
    for a_, b_ in folds:
        if p < 0.9 * len(a_):
            m = LinearRegression().fit(P[a_], y[a_])
            ols.append(np.mean((m.predict(P[b_]) - y[b_]) ** 2))
        for a in alphas:
            m = Ridge(alpha=a).fit(P[a_], y[a_])
            rdg[a].append(np.mean((m.predict(P[b_]) - y[b_]) ** 2))
    best = min(alphas, key=lambda a: np.mean(rdg[a]))
    row = dict(degree=d, n_terms=p, ols=np.mean(ols) if ols else np.nan,
               ridge=np.mean(rdg[best]), ridge_alpha=best, relaxed_lasso=np.nan)
    # relaxed lasso only for var1, it's slow and only helped there
    if V == 1 and 3 <= d <= 6:
        Z = StandardScaler().fit_transform(P)
        err = np.zeros(len(lasso_al))
        for a_, b_ in folds:
            mu = y[a_].mean()
            _, co, _ = lasso_path(Z[a_], y[a_] - mu, alphas=lasso_al, max_iter=50000)
            for k in range(len(lasso_al)):
                s = np.flatnonzero(co[:, k])
                if len(s) == 0:
                    err[k] += np.sum((mu - y[b_]) ** 2)
                    continue
                m = LinearRegression().fit(Z[a_][:, s], y[a_])
                err[k] += np.sum((m.predict(Z[b_][:, s]) - y[b_]) ** 2)
        row["relaxed_lasso"] = err.min() / len(y)
    rows.append(row)
    print(row, flush=True)

pd.DataFrame(rows).to_csv(f"results/sweep_var{V}.csv", index=False)
