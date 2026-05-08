# ============================================================
# CELL 10 — SARIMA / SARIMAX BASELINE
#
# Why this cell exists:
#   Every ML model in this project will be judged against this.
#   SARIMA uses only the price history itself — no climate data,
#   no export signals, no macro features. It is the minimum
#   sensible forecast. If your LightGBM cannot beat this, your
#   features are not adding value.
#
# What we fit:
#   SARIMA(p,d,q)(P,D,Q,12) — one model per group
#   Walk-forward validation on the val set (no peeking at future)
#
# Output:
#   - RMSE, MAE, MAPE per group on validation set
#   - A results table you will copy into your paper
#   - A forecast plot for the best and worst performing group
# ============================================================

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_parquet(FEAT_DIR / "train.parquet")
val_df   = pd.read_parquet(FEAT_DIR / "val.parquet")
test_df  = pd.read_parquet(FEAT_DIR / "test.parquet")

# ── Metric helpers ────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask   = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluation_row(group, y_true, y_pred, model_name="SARIMA"):
    return {
        "model"    : model_name,
        "group"    : group,
        "RMSE"     : round(rmse(y_true, y_pred), 2),
        "MAE"      : round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE (%)" : round(mape(y_true, y_pred), 3),
        "R²"       : round(r2_score(y_true, y_pred), 4),
        "n_obs"    : len(y_true),
    }


print("=" * 65)
print("  SARIMA BASELINE — WALK-FORWARD VALIDATION")
print("  Fitting one model per district × commodity group")
print("=" * 65)

# SARIMA order — informed by Cell 5 ACF/PACF results
# AR(1) dominant (price_lag_1m ρ=0.95), seasonal at lag 12
# For real data: tune these per group using auto_arima or grid search
SARIMA_ORDER         = (1, 1, 1)      # (p, d, q) — non-seasonal
SARIMA_SEASONAL      = (1, 0, 1, 12)  # (P, D, Q, s) — annual seasonality

results_sarima = []
forecasts_dict = {}

groups = sorted(train_df["group_id"].unique())
print(f"\n  Groups to fit: {len(groups)}\n")

for group in groups:
    print(f"  Fitting: {group} ...", end=" ")

    # Pull training and validation series for this group
    train_series = (
        train_df[train_df["group_id"] == group]
        .sort_values("month")
        .set_index("month")["price_modal"]
        .dropna()
    )
    val_series = (
        val_df[val_df["group_id"] == group]
        .sort_values("month")
        .set_index("month")["price_modal"]
        .dropna()
    )

    if len(train_series) < 24 or len(val_series) < 6:
        print(f"⚠ skipped (insufficient data: train={len(train_series)}, val={len(val_series)})")
        continue

    # ── Walk-forward forecast ─────────────────────────────────
    # Instead of fitting once and forecasting the whole val set,
    # we re-fit the model at each step using all data up to that
    # point. This is how you would actually use it in practice.
    history    = train_series.copy()
    preds      = []
    actuals    = []

    for step in range(len(val_series)):
        try:
            model = SARIMAX(
                history,
                order=SARIMA_ORDER,
                seasonal_order=SARIMA_SEASONAL,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)
            yhat   = fitted.forecast(steps=1).iloc[0]
            preds.append(max(yhat, 0))     # prices cannot be negative
            actuals.append(val_series.iloc[step])
            # Add actual observation to history before next step
            history = pd.concat([
                history,
                val_series.iloc[[step]]
            ])
        except Exception:
            # If SARIMA fails on this step, use last known price
            preds.append(history.iloc[-1])
            actuals.append(val_series.iloc[step])

    preds   = np.array(preds)
    actuals = np.array(actuals)

    results_sarima.append(evaluation_row(group, actuals, preds, "SARIMA"))
    forecasts_dict[group] = {
        "dates"  : val_series.index,
        "actual" : actuals,
        "pred"   : preds,
    }

    mape_val = mape(actuals, preds)
    print(f"MAPE = {mape_val:.2f}%")


# ── Results table ─────────────────────────────────────────────
df_results = pd.DataFrame(results_sarima)

print(f"\n{'='*65}")
print(f"  SARIMA BASELINE RESULTS — VALIDATION SET")
print(f"{'='*65}")
print(df_results[[
    "group", "RMSE", "MAE", "MAPE (%)", "R²", "n_obs"
]].to_string(index=False))

print(f"\n  {'─'*50}")
print(f"  SUMMARY STATISTICS")
print(f"  {'─'*50}")
print(f"  Mean MAPE  : {df_results['MAPE (%)'].mean():.3f}%")
print(f"  Best group : {df_results.loc[df_results['MAPE (%)'].idxmin(), 'group']} "
      f"({df_results['MAPE (%)'].min():.3f}%)")
print(f"  Worst group: {df_results.loc[df_results['MAPE (%)'].idxmax(), 'group']} "
      f"({df_results['MAPE (%)'].max():.3f}%)")
print(f"  Mean R²    : {df_results['R²'].mean():.4f}")

# Save results
df_results.to_csv(OUT_DIR / "sarima_val_results.csv", index=False)
print(f"\n  ✓ Results saved → sarima_val_results.csv")


# ── Forecast plot ─────────────────────────────────────────────
# Plot best and worst group side by side
best_group  = df_results.loc[df_results["MAPE (%)"].idxmin(), "group"]
worst_group = df_results.loc[df_results["MAPE (%)"].idxmax(), "group"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("SARIMA Walk-Forward Forecast — Validation Set",
             fontsize=13, color="white", y=1.01)

for ax, group, label in zip(axes,
                              [best_group, worst_group],
                              ["Best performing group", "Worst performing group"]):
    if group not in forecasts_dict:
        continue
    fc   = forecasts_dict[group]
    mape_val = df_results.loc[df_results["group"]==group, "MAPE (%)"].values[0]

    # Training tail for context
    train_tail = (
        train_df[train_df["group_id"] == group]
        .sort_values("month")
        .tail(24)
        .set_index("month")["price_modal"]
    )
    ax.plot(train_tail.index, train_tail.values / 100,
            color="#555", linewidth=1.2, label="Training (last 24m)")
    ax.plot(fc["dates"], fc["actual"] / 100,
            color=VANILLA_COL, linewidth=2.0, label="Actual")
    ax.plot(fc["dates"], fc["pred"] / 100,
            color=SHOCK_COL, linewidth=1.8,
            linestyle="--", label="SARIMA forecast")

    ax.set_title(f"{label}\n{group}  |  MAPE = {mape_val:.2f}%",
                 fontsize=10, color="white", pad=8, loc="left")
    ax.set_ylabel("₹ per kg", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333")
    ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot13_sarima_forecast.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✓ Plot 13 — SARIMA forecast saved")

# ── Naive baseline (sanity check) ────────────────────────────
# A model that just predicts "tomorrow = today" (random walk).
# If SARIMA cannot beat this, something is wrong.
print(f"\n{'='*65}")
print(f"  NAIVE BASELINE (random walk) — for comparison")
print(f"  Predicts: price_t = price_(t-1)")
print(f"{'='*65}")

naive_results = []
for group in groups:
    val_series = (
        val_df[val_df["group_id"] == group]
        .sort_values("month")["price_modal"]
        .dropna()
        .values
    )
    train_last = (
        train_df[train_df["group_id"] == group]
        .sort_values("month")["price_modal"]
        .dropna()
        .values
    )
    if len(val_series) < 2:
        continue
    naive_pred = np.concatenate([[train_last[-1]], val_series[:-1]])
    naive_results.append(evaluation_row(group, val_series, naive_pred, "Naive"))

df_naive = pd.DataFrame(naive_results)

print(f"\n  Naive mean MAPE  : {df_naive['MAPE (%)'].mean():.3f}%")
print(f"  SARIMA mean MAPE : {df_results['MAPE (%)'].mean():.3f}%")

improvement = df_naive["MAPE (%)"].mean() - df_results["MAPE (%)"].mean()
if improvement > 0:
    print(f"\n  ✓ SARIMA beats naive by {improvement:.3f} percentage points")
    print(f"    This confirms the model is learning meaningful patterns")
else:
    print(f"\n  ⚠ SARIMA does not beat naive — consider re-tuning SARIMA order")
    print(f"    Try auto_arima from pmdarima for data-driven order selection")

print(f"""
{'='*65}
  CELL 10 COMPLETE — BASELINE ESTABLISHED

  These SARIMA numbers are your paper's Table 1.
  Every model from Cell 11 onwards must beat:
    MAPE  < {df_results['MAPE (%)'].mean():.2f}%
    RMSE  < {df_results['RMSE'].mean():,.0f}
    R²    > {df_results['R²'].mean():.4f}

  Next: Cell 11 — LightGBM with all 26 features
  Expected improvement: significant, because LightGBM
  uses climate, export, and macro signals that SARIMA ignores.
{'='*65}
""")
