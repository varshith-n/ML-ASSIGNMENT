import pandas as pd, numpy as np, warnings, sys, time
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge, lasso_path
from sklearn.model_selection import KFold
v = int(sys.argv[1]); degs = [int(a) for a in sys.argv[2:]]
tr = pd.read_csv(f"data/IMT2024044_train_var{v}.csv")
X = tr.drop(columns="y").values; y = tr.y.values
kf = list(KFold(5, shuffle=True, random_state=2).split(X))
for deg in degs:
    t = time.time()
    P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    sc = StandardScaler().fit(P); Z = sc.transform(P)
    alphas = np.logspace(0, -3.5, 25)  # descending, as lasso_path returns
    lasso_mse = np.zeros(len(alphas)); rel_mse = np.zeros(len(alphas)); nnz = np.zeros(len(alphas))
    for a_, b_ in kf:
        ym = y[a_].mean()
        _, coefs, _ = lasso_path(Z[a_], y[a_] - ym, alphas=alphas, max_iter=50000)
        for k in range(len(alphas)):
            c = coefs[:, k]; s = np.flatnonzero(c)
            lasso_mse[k] += np.sum((Z[b_] @ c + ym - y[b_]) ** 2)
            nnz[k] += len(s) / 5
            if len(s) == 0 or len(s) > 0.8 * len(a_):
                rel_mse[k] += lasso_mse[k] * 0 + np.sum((ym - y[b_]) ** 2); continue
            m = Ridge(alpha=1e-6).fit(Z[a_][:, s], y[a_])
            rel_mse[k] += np.sum((m.predict(Z[b_][:, s]) - y[b_]) ** 2)
    lasso_mse /= len(y); rel_mse /= len(y)
    i, j = lasso_mse.argmin(), rel_mse.argmin()
    print(f"var{v} deg {deg:2d} p={P.shape[1]:5d} | lasso best {lasso_mse[i]:.4f} (a={alphas[i]:.4f}, nnz~{nnz[i]:.0f}) | relaxed best {rel_mse[j]:.4f} (a={alphas[j]:.4f}, nnz~{nnz[j]:.0f}) | {time.time()-t:.0f}s", flush=True)
