import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso, LinearRegression, Ridge, lasso_path
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024044"
tr = pd.read_csv(f"data/{ROLL}_train_var1.csv")
te = pd.read_csv(f"data/{ROLL}_test_var1.csv")
X, y = tr.drop(columns="y").values, tr.y.values
kt = (np.abs(X) == 1).sum(1)
ke = (np.abs(te.values) == 1).sum(1)
w_k = np.array([(ke == k).mean() / max((kt == k).mean(), 1e-9) for k in range(7)])
w = w_k[kt]
print("weights by k", w_k.round(2))
splits = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=3).split(X))
wm = lambda e, ww: np.sum(e * ww) / np.sum(ww)
al = np.logspace(-0.8, -2.2, 15)


def run(deg, fit_weighted):
    P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    U, W = np.zeros(len(al)), np.zeros(len(al))
    for a_, b_ in splits:
        sc = StandardScaler().fit(P[a_]); Za, Zb = sc.transform(P[a_]), sc.transform(P[b_])
        mu = y[a_].mean()
        if fit_weighted:
            sw = w[a_] / w[a_].mean(); coefs = []
            m = Lasso(alpha=al[0], max_iter=50000, warm_start=True)
            for a in al:
                m.set_params(alpha=a); m.fit(Za, y[a_] - mu, sample_weight=sw); coefs.append(m.coef_.copy())
            coefs = np.array(coefs).T
        else:
            _, coefs, _ = lasso_path(Za, y[a_] - mu, alphas=al, max_iter=50000)
        for k in range(len(al)):
            s = np.flatnonzero(coefs[:, k])
            if len(s) == 0: continue
            lr = LinearRegression().fit(Za[:, s], y[a_], sample_weight=(w[a_] if fit_weighted else None))
            e = (lr.predict(Zb[:, s]) - y[b_]) ** 2
            U[k] += e.mean(); W[k] += wm(e, w[b_])
    U /= len(splits); W /= len(splits)
    k = W.argmin()
    print(f"relaxed lasso deg {deg} fit_weighted={fit_weighted}: best test-weighted mse {W[k]:.4f} (plain {U[k]:.4f}) alpha {al[k]:.4f}", flush=True)

for deg in (4, 5, 6):
    run(deg, False)
run(5, True)

for deg in (5, 6, 7):
    P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    for a in (1, 3, 10):
        U = W = 0
        for a_, b_ in splits:
            e = (Ridge(alpha=a).fit(P[a_], y[a_]).predict(P[b_]) - y[b_]) ** 2
            U += e.mean(); W += wm(e, w[b_])
        print(f"ridge deg {deg} a={a}: test-weighted mse {W/len(splits):.4f} (plain {U/len(splits):.4f})", flush=True)
