# exploratory runs behind the "other things I tried" paragraph. run from the repo root.
# the backward-elimination loop below is very slow (it only finished the first threshold for me).
import pandas as pd, numpy as np, warnings, itertools
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuit
from sklearn.model_selection import RepeatedKFold
from numpy.polynomial import legendre as L
sp_ = lambda n: list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=7).split(np.zeros(n)))

# ---- var1: backward elimination by |t| and OMP at degree 5
tr = pd.read_csv("data/IMT2024044_train_var1.csv")
X = tr.drop(columns="y").values; y = tr.y.values; sp = sp_(len(y))
P = PolynomialFeatures(5, include_bias=True).fit_transform(X)  # col 0 is bias
def backward(Pa, ya, thr):
    s = np.arange(Pa.shape[1])
    while True:
        A = Pa[:, s]; beta, *_ = np.linalg.lstsq(A, ya, rcond=None)
        r = ya - A @ beta; dof = len(ya) - len(s); s2 = r @ r / dof
        cov = s2 * np.linalg.pinv(A.T @ A); t = np.abs(beta) / np.sqrt(np.diag(cov))
        t[0] = 99  # keep bias
        w = t.argmin()
        if t[w] >= thr: return s, beta
        s = np.delete(s, w)
for thr in (2.0, 2.5, 3.0, 3.5, 4.0):
    e = []; n = []
    for a_, b_ in sp:
        s, beta = backward(P[a_], y[a_], thr); n.append(len(s))
        e.append(np.mean((P[b_][:, s] @ beta - y[b_]) ** 2))
    print(f"var1 backward thr={thr}: mse={np.mean(e):.4f} nnz~{np.mean(n):.0f}", flush=True)
for k in (50, 60, 70, 80, 90):
    e = []
    for a_, b_ in sp:
        m = OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(P[a_][:, 1:], y[a_])
        e.append(np.mean((m.predict(P[b_][:, 1:]) - y[b_]) ** 2))
    print(f"var1 OMP k={k}: mse={np.mean(e):.4f}", flush=True)

# ---- var2: legendre tensor basis ridge vs monomial, same degree
tr = pd.read_csv("data/IMT2024044_train_var2.csv")
X = tr.drop(columns="y").values; y = tr.y.values; sp = sp_(len(y))
def leg(X, deg):
    V = [L.legvander(X[:, j], deg) for j in range(3)]
    cols = [V[0][:, i] * V[1][:, j] * V[2][:, k] for i in range(deg + 1) for j in range(deg + 1) for k in range(deg + 1) if i + j + k <= deg]
    return np.column_stack(cols)
for deg in (10, 11, 12):
    P = leg(X, deg); out = {}
    for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1):
        out[a] = np.mean([np.mean((Ridge(alpha=a).fit(P[a_], y[a_]).predict(P[b_]) - y[b_]) ** 2) for a_, b_ in sp])
    b = min(out, key=out.get); print(f"var2 legendre deg {deg}: best a={b} mse={out[b]:.4f}", flush=True)
