# ============================================================
# CELL 6 — CROSS-CORRELATION & LAG STRUCTURE ANALYSIS
# Formally proves which lag of world vanilla price / IOD / INR
# most strongly predicts Karnataka mandi prices.
# Also runs Granger causality tests.
# ============================================================

df = pd.read_parquet(PROC_DIR / "panel_monthly.parquet")

# Aggregate vanilla to single national series (median across districts)
van_nat = (df[df["commodity"] == "vanilla"]
           .groupby("month")[["price_modal",
                               "vanilla_world_price_usd_kg",
                               "iod_index",
                               "inr_usd_rate"]]
           .median()
           .sort_index()
           .dropna(subset=["price_modal"]))

# ── 1. Cross-correlation function (CCF) ──────────────────────
def ccf_values(x: pd.Series, y: pd.Series, max_lag: int = 18) -> pd.DataFrame:
    """
    Compute cross-correlation of x with y at lags 0..max_lag.
    Lag k means: how well does x_{t-k} predict y_t?
    """
    x_s = (x - x.mean()) / x.std()
    y_s = (y - y.mean()) / y.std()
    rows = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            aligned_x, aligned_y = x_s.values, y_s.values
        else:
            aligned_x = x_s.values[lag:]
            aligned_y = y_s.values[:-lag] if lag > 0 else y_s.values
            min_len   = min(len(aligned_x), len(aligned_y))
            aligned_x = aligned_x[:min_len]
            aligned_y = aligned_y[:min_len]
        corr = np.corrcoef(aligned_x, aligned_y)[0, 1]
        rows.append({"lag_months": lag, "ccf": corr})
    return pd.DataFrame(rows)


# Compute CCF for three exogenous predictors
predictors = {
    "World Vanilla Price (USD/kg)" : "vanilla_world_price_usd_kg",
    "IOD Index"                    : "iod_index",
    "INR/USD Exchange Rate"        : "inr_usd_rate",
}

ccf_results = {}
for label, col in predictors.items():
    if col in van_nat.columns:
        s = van_nat[[col, "price_modal"]].dropna()
        ccf_results[label] = ccf_values(s[col], s["price_modal"], max_lag=18)

# ── Plot CCF ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Cross-Correlation Function — Predictor → Karnataka Vanilla Price\n"
             "(lag = months by which predictor LEADS the mandi price)",
             fontsize=12, color="white", y=1.03)

colors_ccf = [VANILLA_COL, IOD_COL, ARECANUT_COL]

for ax, (label, ccf_df), color in zip(axes, ccf_results.items(), colors_ccf):
    ci = 1.96 / np.sqrt(len(van_nat))

    bars = ax.bar(ccf_df["lag_months"], ccf_df["ccf"],
                  color=[color if abs(v) > ci else "#333"
                         for v in ccf_df["ccf"]],
                  alpha=0.85, width=0.7, edgecolor="none")

    ax.axhline( ci, color="#888", linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(-ci, color="#888", linestyle="--", linewidth=0.8)
    ax.axhline( 0,  color="#555", linewidth=0.5)

    # Mark the peak lag
    peak_row  = ccf_df.loc[ccf_df["ccf"].abs().idxmax()]
    peak_lag  = int(peak_row["lag_months"])
    peak_corr = peak_row["ccf"]
    ax.axvline(peak_lag, color=color, linewidth=1.5, linestyle=":",
               label=f"Peak lag = {peak_lag}m  (ρ={peak_corr:.3f})")
    ax.annotate(f"  lag {peak_lag}\n  ρ={peak_corr:.3f}",
                xy=(peak_lag, peak_corr),
                xytext=(peak_lag + 1.5, peak_corr * 0.85),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.set_xlabel("Lag (months)", fontsize=9)
    ax.set_ylabel("Cross-correlation", fontsize=9)
    ax.set_title(label, fontsize=10, color="white", pad=8, loc="left")
    ax.set_xticks(range(0, 19, 2))
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333")
    ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot10_cross_correlation.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✓ Plot 10 — Cross-correlation saved")


# ── 2. Granger causality tests ───────────────────────────────
print(f"\n{'='*65}")
print("  GRANGER CAUSALITY TESTS")
print("  H0: predictor does NOT Granger-cause vanilla mandi price")
print("  Reject H0 (p < 0.05) → predictor has predictive power")
print(f"{'='*65}")

max_lag_granger = 6
granger_rows    = []

for label, col in predictors.items():
    if col not in van_nat.columns:
        continue
    pair = van_nat[["price_modal", col]].dropna()
    if len(pair) < 24:
        continue
    try:
        gc_result = grangercausalitytests(
            pair[["price_modal", col]], maxlag=max_lag_granger, verbose=False
        )
        for lag in range(1, max_lag_granger + 1):
            f_stat = gc_result[lag][0]["ssr_ftest"][0]
            p_val  = gc_result[lag][0]["ssr_ftest"][1]
            granger_rows.append({
                "predictor" : label,
                "lag"       : lag,
                "F-stat"    : round(f_stat, 4),
                "p-value"   : round(p_val,  4),
                "significant": "✓" if p_val < 0.05 else "✗",
            })
    except Exception as e:
        print(f"  ⚠ Granger test failed for {label}: {e}")

df_granger = pd.DataFrame(granger_rows)
if not df_granger.empty:
    print(df_granger.to_string(index=False))
    print(f"\n  Best predictive lags (p < 0.05):")
    sig = df_granger[df_granger["p-value"] < 0.05]
    for pred, grp in sig.groupby("predictor"):
        lags_sig = grp["lag"].tolist()
        print(f"    {pred}: lags {lags_sig}")

print(f"{'='*65}")
print("✓ Cell 6 complete — cross-correlation + Granger causality done")
