import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, lasso_path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024044"
DEG = 5
ALPHAS = np.logspace(-0.8, -2.2, 30)  # descending, lasso_path wants that


def fit(Xtr, ytr, alpha):
    # lasso picks the terms, plain least squares gives the final coefficients
    pf = PolynomialFeatures(DEG, include_bias=False)
    P = pf.fit_transform(Xtr)
    sc = StandardScaler().fit(P)
    Z = sc.transform(P)
    mu = ytr.mean()
    _, co, _ = lasso_path(Z, ytr - mu, alphas=[alpha], max_iter=100000)
    s = np.flatnonzero(co[:, 0])
    lr = LinearRegression().fit(Z[:, s], ytr)
    return pf, sc, s, lr


def predict(model, Xnew):
    pf, sc, s, lr = model
    return lr.predict(sc.transform(pf.transform(Xnew))[:, s])


def pick_alpha(X, y):
    P = PolynomialFeatures(DEG, include_bias=False).fit_transform(X)
    err = np.zeros(len(ALPHAS))
    nnz = np.zeros(len(ALPHAS))
    splits = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=7).split(X))
    for a_, b_ in splits:
        # scaler fit on the training fold only
        sc = StandardScaler().fit(P[a_])
        Za, Zb = sc.transform(P[a_]), sc.transform(P[b_])
        mu = y[a_].mean()
        _, co, _ = lasso_path(Za, y[a_] - mu, alphas=ALPHAS, max_iter=100000)
        for k in range(len(ALPHAS)):
            s = np.flatnonzero(co[:, k])
            nnz[k] += len(s)
            if len(s) == 0:
                err[k] += np.mean((mu - y[b_]) ** 2)
                continue
            lr = LinearRegression().fit(Za[:, s], y[a_])
            err[k] += np.mean((lr.predict(Zb[:, s]) - y[b_]) ** 2)
    err /= len(splits)
    nnz /= len(splits)
    return ALPHAS[err.argmin()], err, nnz


tr = pd.read_csv(f"data/{ROLL}_train_var1.csv")
te = pd.read_csv(f"data/{ROLL}_test_var1.csv")
X, y = tr.drop(columns="y").values, tr.y.values

# sanity check on a held-out 20% (whole selection redone on the other 80%)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=0)
a_h, _, _ = pick_alpha(Xa, ya)
m_h = fit(Xa, ya, a_h)
ph = predict(m_h, Xb)
hold = dict(mse=mean_squared_error(yb, ph), r2=r2_score(yb, ph), terms=int(len(m_h[2])))
print("holdout:", hold)

# final model on all 1000 rows
alpha, err, nnz = pick_alpha(X, y)
model = fit(X, y, alpha)
ptr = predict(model, X)
print("alpha", alpha, "terms kept", len(model[2]), "cv mse", err.min())
print("train mse", mean_squared_error(y, ptr), "r2", r2_score(y, ptr))

pred = predict(model, te.values)
pd.DataFrame({"y": pred}).to_csv(f"predictions/{ROLL}_pred_var1.csv", index=False)

pd.DataFrame({"alpha": ALPHAS, "cv_mse": err, "terms": nnz}).to_csv("results/var1_alpha_path.csv", index=False)
names = model[0].get_feature_names_out([f"x{i}" for i in range(1, 7)])[model[2]]
json.dump(dict(degree=DEG, alpha=float(alpha), terms=int(len(model[2])), cv_mse=float(err.min()),
               holdout=hold, train_mse=float(mean_squared_error(y, ptr)), train_r2=float(r2_score(y, ptr)),
               pred_mean=float(pred.mean()), pred_std=float(pred.std())),
          open("results/var1_summary.json", "w"), indent=1)
np.save("results/var1_holdout.npy", np.c_[yb, ph])
