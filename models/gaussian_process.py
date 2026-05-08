from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, Matern, WhiteKernel, ConstantKernel as C
)

train_df = pd.read_parquet(FEAT_DIR / "train.parquet")
val_df   = pd.read_parquet(FEAT_DIR / "val.parquet")
test_df  = pd.read_parquet(FEAT_DIR / "test.parquet")

with open(FEAT_DIR / "feature_config.json") as f:
    feat_cfg = json.load(f)

TARGET  = feat_cfg["TARGET"]
groups  = sorted(train_df["group_id"].unique())

LAMBDA_VALUES    = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
HOLDING_COST     = 0.008
COVERAGE_TARGET  = 0.80

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

kernel = (
    C(1.0, (1e-3, 1e3)) * RBF(length_scale=24.0, length_scale_bounds=(6.0, 120.0))
    + C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=12.0,
                                             length_scale_bounds=(0.1, 10.0),
                                             periodicity_bounds=(11.0, 13.0))
    + C(1.0, (1e-3, 1e3)) * Matern(length_scale=2.0, length_scale_bounds=(0.5, 12.0), nu=0.5)
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
)

print("=" * 65)
print("  CELL 16 — GAUSSIAN PROCESS COMPOSITE KERNEL")
print("  K = C*RBF + C*Periodic(12) + C*Matern(0.5) + White")
print(f"  Groups : {len(groups)}")
print(f"  Lambda sweep : {LAMBDA_VALUES}")
print("=" * 65)

gp_results   = []
gp_forecasts = {}
lambda_revenues = {lam: [] for lam in LAMBDA_VALUES}

for group in groups:
    tr = train_df[train_df["group_id"]==group].sort_values("month")
    vl = val_df[val_df["group_id"]==group].sort_values("month")
    te = test_df[test_df["group_id"]==group].sort_values("month")

    if len(tr) < 12 or len(te) < 3:
        continue

    y_tr = tr[TARGET].values.astype(float)
    y_vl = vl[TARGET].values.astype(float)
    y_te = te[TARGET].values.astype(float)

    scaler_y = MinMaxScaler()
    y_tr_s   = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_vl_s   = scaler_y.transform(y_vl.reshape(-1,1)).ravel()
    y_te_s   = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    n_tr = len(y_tr_s)
    n_vl = len(y_vl_s)
    n_te = len(y_te_s)

    t_tr = np.arange(n_tr).reshape(-1, 1).astype(float)
    t_vl = np.arange(n_tr, n_tr + n_vl).reshape(-1, 1).astype(float)
    t_te = np.arange(n_tr + n_vl, n_tr + n_vl + n_te).reshape(-1, 1).astype(float)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=SEED,
    )
    gp.fit(t_tr, y_tr_s)

    mu_te_s, sigma_te_s = gp.predict(t_te, return_std=True)

    mu_te    = scaler_y.inverse_transform(mu_te_s.reshape(-1,1)).ravel()
    sigma_te = sigma_te_s * (scaler_y.data_max_ - scaler_y.data_min_)

    lower_80 = mu_te - 1.282 * sigma_te
    upper_80 = mu_te + 1.282 * sigma_te
    lower_95 = mu_te - 1.960 * sigma_te
    upper_95 = mu_te + 1.960 * sigma_te

    cov_80 = np.mean((y_te >= lower_80) & (y_te <= upper_80))
    cov_95 = np.mean((y_te >= lower_95) & (y_te <= upper_95))
    te_mape = mape(y_te, mu_te)

    learned = gp.kernel_
    print(f"  {group}")
    print(f"    MAPE={te_mape:.2f}%  cov80={cov_80:.2f}  cov95={cov_95:.2f}")
    print(f"    Kernel: {learned}")

    gp_results.append({
        "group"      : group,
        "GP_MAPE_%"  : round(te_mape, 3),
        "GP_cov_80"  : round(cov_80, 4),
        "GP_cov_95"  : round(cov_95, 4),
        "sigma_mean" : round(float(sigma_te.mean()), 2),
    })

    gp_forecasts[group] = {
        "dates"    : te["month"].values,
        "actual"   : y_te,
        "mu"       : mu_te,
        "sigma"    : sigma_te,
        "lower_80" : lower_80,
        "upper_80" : upper_80,
        "lower_95" : lower_95,
        "upper_95" : upper_95,
    }

    for lam in LAMBDA_VALUES:
        QUANTITY   = 100
        holdings   = 0
        revenue    = 0

        for t in range(n_te):
            p_now = y_te[t]

            if t < n_te - 1:
                threshold = (mu_te[t+1] - lam * sigma_te[t+1]) / (1 + HOLDING_COST)
            else:
                threshold = 0

            if p_now >= threshold:
                months_held  = holdings + 1
                penalty      = (1 - HOLDING_COST) ** holdings
                revenue     += p_now * QUANTITY * months_held * penalty
                holdings     = 0
            else:
                holdings += 1

        if holdings > 0:
            penalty  = (1 - HOLDING_COST) ** holdings
            revenue += y_te[-1] * QUANTITY * holdings * penalty

        lambda_revenues[lam].append(revenue)

df_gp = pd.DataFrame(gp_results)
df_gp.to_csv(OUT_DIR / "gp_results.csv", index=False)

print(f"\n{'='*65}")
print(f"  GP RESULTS SUMMARY")
print(f"{'='*65}")
print(df_gp.to_string(index=False))
print(f"\n  Mean GP MAPE    : {df_gp['GP_MAPE_%'].mean():.3f}%")
print(f"  Mean cov @ 80%  : {df_gp['GP_cov_80'].mean():.4f}  (nominal 0.80)")
print(f"  Mean cov @ 95%  : {df_gp['GP_cov_95'].mean():.4f}  (nominal 0.95)")

print(f"\n{'='*65}")
print(f"  RISK AVERSION SWEEP  (λ = 0 risk-neutral → 3 very risk-averse)")
print(f"  Revenue per 100 quintals, averaged across groups")
print(f"{'='*65}")

lam_mean_rev = {lam: np.mean(lambda_revenues[lam]) for lam in LAMBDA_VALUES}
best_lam     = min(lam_mean_rev, key=lambda l: -lam_mean_rev[l])

print(f"\n  {'λ':>6}  {'Mean Revenue (₹)':>20}  {'Note'}")
print(f"  {'─'*50}")
for lam in LAMBDA_VALUES:
    note = " ← OPTIMAL" if lam == best_lam else ""
    print(f"  {lam:>6.1f}  ₹{lam_mean_rev[lam]:>19,.0f}  {note}")

df_lam = pd.DataFrame([
    {"lambda": lam, "mean_revenue": lam_mean_rev[lam]}
    for lam in LAMBDA_VALUES
])
df_lam.to_csv(OUT_DIR / "lambda_sweep.csv", index=False)


fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Gaussian Process Composite Kernel — Test Set Posterior",
             fontsize=13, color="white", y=1.01)

for ax, group in zip(axes, groups[:2]):
    if group not in gp_forecasts:
        continue
    fc = gp_forecasts[group]
    gp_row = df_gp[df_gp["group"]==group]

    ax.fill_between(fc["dates"], fc["lower_95"]/100, fc["upper_95"]/100,
                    alpha=0.15, color=IOD_COL, label="95% GP interval")
    ax.fill_between(fc["dates"], fc["lower_80"]/100, fc["upper_80"]/100,
                    alpha=0.30, color=IOD_COL, label="80% GP interval")
    ax.plot(fc["dates"], fc["actual"]/100,
            color=VANILLA_COL, linewidth=2.0, label="Actual")
    ax.plot(fc["dates"], fc["mu"]/100,
            color=ARECANUT_COL, linewidth=1.6, linestyle="--",
            label="GP posterior mean")
    ax.fill_between(fc["dates"],
                    (fc["mu"] - fc["sigma"])/100,
                    (fc["mu"] + fc["sigma"])/100,
                    alpha=0.12, color=SHOCK_COL, label="±1σ")

    cov80 = gp_row["GP_cov_80"].values[0] if len(gp_row) else 0
    gp_m  = gp_row["GP_MAPE_%"].values[0] if len(gp_row) else 0
    ax.set_title(f"{group}  |  GP MAPE={gp_m:.2f}%  cov80={cov80*100:.1f}%",
                 fontsize=10, color="white", pad=8, loc="left")
    ax.set_ylabel("₹ per kg", fontsize=9)
    ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333")
    ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot21_gp_posterior.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0f0f0f")

revs  = [lam_mean_rev[l] for l in LAMBDA_VALUES]
colors_bar = [VANILLA_COL if l == best_lam else "#4a7fa5" for l in LAMBDA_VALUES]
ax.bar([str(l) for l in LAMBDA_VALUES], revs,
       color=colors_bar, alpha=0.85, edgecolor="none")
ax.axhline(np.mean(lambda_revenues[0.0]), color=SHOCK_COL,
           linewidth=1.2, linestyle="--", label="λ=0 (risk-neutral) revenue")
ax.set_xlabel("Risk aversion parameter λ", fontsize=10)
ax.set_ylabel("Mean revenue ₹ / 100 quintals", fontsize=10)
ax.set_title("Optimal λ Sweep — GP-Informed Selling Strategy\n"
             "(gold bar = revenue-maximising risk aversion level)",
             fontsize=12, color="white", pad=10, loc="left")
ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333")
ax.grid(axis="y")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig(OUT_DIR / "plot22_lambda_sweep.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()

print(f"\n  ✓ plot21_gp_posterior.png saved")
print(f"  ✓ plot22_lambda_sweep.png saved")
print(f"  ✓ gp_results.csv saved")
print(f"  ✓ lambda_sweep.csv saved")

aci_cov   = pd.read_csv(OUT_DIR / "aci_results.csv")["empirical_coverage"].mean()
gp_cov_80 = df_gp["GP_cov_80"].mean()

print(f"""
{'='*65}
  THREE-WAY UNCERTAINTY COMPARISON

  Method            Coverage @ 80%    Gap
  ─────────────────────────────────────────
  Static conformal  48.3%             31.7pp
  ACI               {aci_cov*100:.1f}%             {abs(aci_cov-0.80)*100:.2f}pp
  GP posterior      {gp_cov_80*100:.1f}%             {abs(gp_cov_80-0.80)*100:.2f}pp

  Optimal λ         : {best_lam}
  GP revenue at λ*  : ₹{lam_mean_rev[best_lam]:,.0f}
  GP revenue at λ=0 : ₹{lam_mean_rev[0.0]:,.0f}

  Sell condition (GP):
    p_t ≥ μ_GP(t+1) - {best_lam} × σ_GP(t+1)  /  (1 + 0.008)
{'='*65}
""")
