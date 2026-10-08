import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures

ROLL = "IMT2024044"
DEGREES = [10, 11, 12, 13]
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]


def search(X, y, reps=10):
    splits = list(RepeatedKFold(n_splits=5, n_repeats=reps, random_state=11).split(X))
    out = {}
    for d in DEGREES:
        P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
        for a in ALPHAS:
            e = [np.mean((Ridge(alpha=a).fit(P[i], y[i]).predict(P[j]) - y[j]) ** 2) for i, j in splits]
            out[(d, a)] = np.mean(e)
    best = min(out, key=out.get)
    return best, out


def fit(X, y, d, a):
    pf = PolynomialFeatures(d, include_bias=False)
    return pf, Ridge(alpha=a).fit(pf.fit_transform(X), y)


tr = pd.read_csv(f"data/{ROLL}_train_var2.csv")
te = pd.read_csv(f"data/{ROLL}_test_var2.csv")
X, y = tr.drop(columns="y").values, tr.y.values

# held-out 20% check, selection repeated on the remaining 80%
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=0)
bh, _ = search(Xa, ya, reps=3)
pf, m = fit(Xa, ya, *bh)
ph = m.predict(pf.transform(Xb))
hold = dict(degree=bh[0], alpha=bh[1], mse=mean_squared_error(yb, ph), r2=r2_score(yb, ph))
print("holdout:", hold)

(d, a), grid = search(X, y)
pf, m = fit(X, y, d, a)
ptr = m.predict(pf.transform(X))
print("chosen degree", d, "alpha", a, "cv mse", grid[(d, a)])
print("train mse", mean_squared_error(y, ptr), "r2", r2_score(y, ptr))

pred = m.predict(pf.transform(te.values))
pd.DataFrame({"y": pred}).to_csv(f"predictions/{ROLL}_pred_var2.csv", index=False)

pd.DataFrame([dict(degree=k[0], alpha=k[1], cv_mse=v) for k, v in grid.items()]).to_csv("results/var2_grid.csv", index=False)
json.dump(dict(degree=d, alpha=a, n_terms=int(m.coef_.size), cv_mse=float(grid[(d, a)]), holdout=hold,
               train_mse=float(mean_squared_error(y, ptr)), train_r2=float(r2_score(y, ptr)),
               pred_mean=float(pred.mean()), pred_std=float(pred.std())),
          open("results/var2_summary.json", "w"), indent=1)
np.save("results/var2_holdout.npy", np.c_[yb, ph])
