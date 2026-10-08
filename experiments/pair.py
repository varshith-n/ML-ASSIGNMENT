import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
tr = pd.read_csv("data/IMT2024044_train_var2.csv")
X = tr.drop(columns="y").values; y = tr.y.values
sp = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=11).split(X))
cfg = {10: [0.05, 0.1, 0.2], 11: [0.1, 0.2, 0.3], 12: [0.1, 0.2, 0.3, 0.5], 13: [0.2, 0.3, 0.5]}
res = {}
for deg, al in cfg.items():
    P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    for a in al:
        res[(deg, a)] = np.array([np.mean((Ridge(alpha=a).fit(P[i], y[i]).predict(P[j]) - y[j]) ** 2) for i, j in sp])
best = min(res, key=lambda k: res[k].mean())
for k, v in sorted(res.items()):
    d = v - res[best]
    print(f"deg {k[0]} a={k[1]:<4} mse={v.mean():.4f}  diff vs best={d.mean():+.4f} (se {d.std()/np.sqrt(len(d)):.4f})")
print("best", best)
