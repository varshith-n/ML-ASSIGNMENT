import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, lasso_path
from sklearn.model_selection import RepeatedKFold
rk = lambda n: list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=7).split(np.zeros(n)))
# var2: ridge on raw monomials, finer alpha grid, degrees 8..14
tr = pd.read_csv("data/IMT2024044_train_var2.csv")
X = tr.drop(columns="y").values; y = tr.y.values; sp = rk(len(y))
alphas = [0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.8, 1.6]
print("var2 ridge, repeated 5x3 CV")
for deg in range(8, 15):
    P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    sc = {a: [] for a in alphas}
    for a_, b_ in sp:
        for a in alphas:
            m = Ridge(alpha=a).fit(P[a_], y[a_]); sc[a].append(np.mean((m.predict(P[b_]) - y[b_]) ** 2))
    b = min(alphas, key=lambda a: np.mean(sc[a])); f = np.array(sc[b])
    print(f"  deg {deg:2d} best a={b:<6} mse={f.mean():.4f} +- {f.std()/np.sqrt(len(f)):.4f}", flush=True)
# var1: relaxed lasso deg 5 vs plain OLS on a few degrees under same repeated CV
tr = pd.read_csv("data/IMT2024044_train_var1.csv")
X = tr.drop(columns="y").values; y = tr.y.values; sp = rk(len(y))
P = PolynomialFeatures(5, include_bias=False).fit_transform(X); Z = StandardScaler().fit_transform(P)
# alphas below are descending, which is the order lasso_path returns its columns in
alphas = np.logspace(-0.6, -2.6, 17); res = np.zeros(len(alphas)); nn = np.zeros(len(alphas))
for a_, b_ in sp:
    ym = y[a_].mean(); _, co, _ = lasso_path(Z[a_], y[a_] - ym, alphas=alphas, max_iter=50000)
    for k in range(len(alphas)):
        s = np.flatnonzero(co[:, k]); nn[k] += len(s)
        m = Ridge(alpha=1e-6).fit(Z[a_][:, s], y[a_]); res[k] += np.mean((m.predict(Z[b_][:, s]) - y[b_]) ** 2)
res /= len(sp); nn /= len(sp)
print("var1 deg5 relaxed lasso, repeated CV")
for a, r, n in zip(alphas, res, nn): print(f"  a={a:.4f} mse={r:.4f} nnz={n:.0f}")
