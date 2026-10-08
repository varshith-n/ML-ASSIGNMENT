import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLL = "IMT2024044"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
C = {"ols": "#8a8a8a", "ridge": "#2a6f97", "lasso": "#d1495b"}

s1 = pd.read_csv("results/sweep_var1.csv")
s2 = pd.read_csv("results/sweep_var2.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for a, s, t in ((ax[0], s1, "Problem 1 (6 features)"), (ax[1], s2, "Problem 2 (3 features)")):
    a.plot(s.degree, s.ols, "o--", color=C["ols"], ms=3, label="least squares")
    a.plot(s.degree, s.ridge, "o-", color=C["ridge"], ms=3, label="ridge")
    if s.relaxed_lasso.notna().any():
        a.plot(s.degree, s.relaxed_lasso, "o-", color=C["lasso"], ms=3, label="lasso + refit")
    a.set_yscale("log"); a.set_xlabel("polynomial degree"); a.set_ylabel("5-fold CV MSE"); a.set_title(t, fontsize=9)
    a.set_ylim(0.15, 60)
ax[0].legend(frameon=False, fontsize=7.5)
ax[0].axvline(5, color=C["lasso"], lw=0.8, alpha=0.5); ax[1].axvline(12, color=C["ridge"], lw=0.8, alpha=0.5)
plt.tight_layout(); plt.savefig("results/fig_degree.png"); plt.close()

# shift + holdout fits
tr = pd.read_csv(f"data/{ROLL}_train_var1.csv"); te = pd.read_csv(f"data/{ROLL}_test_var1.csv")
kt = (tr.drop(columns="y").abs() == 1).sum(axis=1); ke = (te.abs() == 1).sum(axis=1)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
ks = np.arange(7)
ax[0].bar(ks - 0.2, [(kt == k).mean() for k in ks], 0.4, color="#8a8a8a", label="train")
ax[0].bar(ks + 0.2, [(ke == k).mean() for k in ks], 0.4, color=C["ridge"], label="test")
ax[0].set_xlabel("inputs sitting at +-1 in a row"); ax[0].set_ylabel("share of rows"); ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title("Problem 1 input shift", fontsize=9)
for a, f, t in ((ax[1], "var1", "Problem 1 hold-out"), (ax[2], "var2", "Problem 2 hold-out")):
    h = np.load(f"results/{f}_holdout.npy")
    a.scatter(h[:, 0], h[:, 1], s=6, alpha=0.6, color=C["ridge"], edgecolor="none")
    lo, hi = h.min(), h.max(); a.plot([lo, hi], [lo, hi], color="#444", lw=0.8)
    a.set_xlabel("actual y"); a.set_ylabel("predicted y"); a.set_title(t, fontsize=9)
plt.tight_layout(); plt.savefig("results/fig_diag.png"); plt.close()
