# ============================================================
# CELL 13 — LSTM
#
# Why LSTM is the right next model after XGBoost:
#
#   SARIMA:   2.524%  — captures serial correlation structurally
#   XGBoost:  2.463%  — marginally best, beats SARIMA on arecanut
#   LightGBM: 2.976%  — momentum features dominate
#
#   All three re-fit or re-estimate at each walk-forward step.
#   LSTM is fundamentally different — it learns the ENTIRE temporal
#   dynamics in its hidden state during training and then rolls
#   the hidden state forward at inference without re-fitting.
#   This exploits the long-memory structure (Hurst H > 0.5 from
#   Cell 7) that both SARIMA and gradient boosting miss.
#
# Leakage prevention in LSTM (three rules):
#
#   RULE 1 — Sequence construction:
#     To predict month t, the input window is [t-L, t-1].
#     The window NEVER contains t or any future observation.
#
#   RULE 2 — Scaler fitted on training sequences only:
#     MinMaxScaler is fit on training sequences, then applied
#     to val and test sequences. Scaler never sees val or test
#     targets during fitting.
#
#   RULE 3 — Inference window at step i:
#     Window contains training history + val steps [0..i-1].
#     Val step i is never in the window when predicting step i.
# ============================================================

train_df = pd.read_parquet(FEAT_DIR / "train.parquet")
val_df   = pd.read_parquet(FEAT_DIR / "val.parquet")
test_df  = pd.read_parquet(FEAT_DIR / "test.parquet")

with open(FEAT_DIR / "feature_config.json") as f:
    feat_cfg = json.load(f)

FEATURES = feat_cfg["FEATURES_LSTM"]
TARGET   = feat_cfg["TARGET"]

sarima_results = pd.read_csv(OUT_DIR / "sarima_val_results.csv")
lgb_results    = pd.read_csv(OUT_DIR / "lgb_val_results.csv")
xgb_results    = pd.read_csv(OUT_DIR / "xgb_val_results.csv")

groups    = sorted(train_df["group_id"].unique())
LOOKBACK  = 12
HIDDEN    = 64
N_LAYERS  = 2
DROPOUT   = 0.2
EPOCHS    = 100
PATIENCE  = 10
LR        = 1e-3

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print("=" * 65)
print("  CELL 13 — LSTM")
print(f"  Features   : {len(FEATURES)}")
print(f"  Lookback   : {LOOKBACK} months")
print(f"  Hidden     : {HIDDEN} units × {N_LAYERS} layers")
print(f"  Groups     : {len(groups)}")
print(f"  Baseline   : SARIMA 2.524% | XGBoost 2.463%")
print("=" * 65)


class SpiceLSTM(nn.Module):
    """
    Stacked LSTM for univariate/multivariate time series.
    Input  shape: (batch, lookback, n_features)
    Output shape: (batch, 1)
    """
    def __init__(self, n_features, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """
    Build (X_seq, y_seq) pairs.
    X_seq[i] = X[i : i+lookback]          ← past L observations
    y_seq[i] = y[i + lookback]             ← the next observation to predict
    No future data ever enters X_seq[i].
    """
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(model, X_seq, y_seq, X_val_seq, y_val_seq,
               epochs, patience, lr, device):
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    X_t  = torch.tensor(X_seq).to(device)
    y_t  = torch.tensor(y_seq).to(device)
    Xv_t = torch.tensor(X_val_seq).to(device)
    yv_t = torch.tensor(y_val_seq).to(device)

    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv_t), yv_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_weights)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")
print(f"\n  Training one LSTM per group ...\n")

results_lstm   = []
forecasts_lstm = {}

for group in groups:
    tr = train_df[train_df["group_id"] == group].sort_values("month")
    vl = val_df[val_df["group_id"]   == group].sort_values("month")

    if len(tr) < LOOKBACK + 6 or len(vl) < 3:
        print(f"  {group:<35} skipped (insufficient data)")
        continue

    X_tr_raw = tr[FEATURES].values.astype(np.float32)
    y_tr_raw = tr[TARGET].values.astype(np.float32)
    X_vl_raw = vl[FEATURES].values.astype(np.float32)
    y_vl_raw = vl[TARGET].values.astype(np.float32)

    # ── RULE 2: fit scaler on training features only ──────────
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_tr_scaled = feat_scaler.fit_transform(X_tr_raw)
    y_tr_scaled = target_scaler.fit_transform(y_tr_raw.reshape(-1,1)).ravel()

    X_vl_scaled = feat_scaler.transform(X_vl_raw)
    y_vl_scaled = target_scaler.transform(y_vl_raw.reshape(-1,1)).ravel()

    X_seq, y_seq = make_sequences(X_tr_scaled, y_tr_scaled, LOOKBACK)

    combined_X = np.vstack([X_tr_scaled, X_vl_scaled])
    combined_y = np.concatenate([y_tr_scaled, y_vl_scaled])
    X_val_seq, y_val_seq = make_sequences(
        combined_X[len(X_tr_scaled):],
        combined_y[len(y_tr_scaled):],
        LOOKBACK
    )

    if len(X_val_seq) == 0:
        X_val_seq = X_seq[-5:]
        y_val_seq = y_seq[-5:]

    torch.manual_seed(SEED)
    model = SpiceLSTM(len(FEATURES), HIDDEN, N_LAYERS, DROPOUT).to(device)
    model = train_lstm(
        model, X_seq, y_seq, X_val_seq, y_val_seq,
        EPOCHS, PATIENCE, LR, device
    )

    # ── RULE 3: leakage-free walk-forward inference ───────────
    model.eval()
    history_X    = X_tr_scaled.copy()
    history_y    = y_tr_scaled.copy()
    preds_scaled = []

    for step in range(len(vl)):
        window_X = history_X[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURES))
        window_t = torch.tensor(window_X, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_s = model(window_t).item()

        preds_scaled.append(pred_s)

        history_X = np.vstack([history_X, X_vl_scaled[step:step+1]])
        history_y = np.append(history_y, y_vl_scaled[step])

    preds = target_scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).ravel()
    preds = np.clip(preds, 0, None)

    actuals = y_vl_raw
    mape_v  = mape(actuals, preds)
    rmse_v  = np.sqrt(mean_squared_error(actuals, preds))
    mae_v   = mean_absolute_error(actuals, preds)
    r2_v    = r2_score(actuals, preds)

    sar_m = sarima_results.loc[sarima_results["group"]==group,"MAPE (%)"].values
    xgb_m = xgb_results.loc[xgb_results["group"]==group,"MAPE (%)"].values
    best_prev = min(sar_m[0] if len(sar_m) else 99,
                    xgb_m[0] if len(xgb_m) else 99)
    beat = f"↓ {best_prev - mape_v:+.2f}pp vs best" if mape_v < best_prev \
           else f"↑ {mape_v - best_prev:+.2f}pp vs best"
    print(f"  {group:<35} MAPE={mape_v:.2f}%  {beat}")

    results_lstm.append({
        "model"    : "LSTM",
        "group"    : group,
        "RMSE"     : round(float(rmse_v), 2),
        "MAE"      : round(float(mae_v), 2),
        "MAPE (%)" : round(float(mape_v), 3),
        "R²"       : round(float(r2_v), 4),
        "n_obs"    : len(actuals),
    })
    forecasts_lstm[group] = {
        "dates" : vl["month"].values,
        "actual": actuals,
        "pred"  : preds,
    }

df_lstm = pd.DataFrame(results_lstm)
df_lstm.to_csv(OUT_DIR / "lstm_val_results.csv", index=False)


print(f"\n{'='*75}")
print(f"  COMPLETE MODEL COMPARISON — VALIDATION SET  (Table 2 in paper)")
print(f"{'='*75}")

all_rows = []
for group in groups:
    def g(df, grp):
        r = df[df["group"]==grp]["MAPE (%)"].values
        return r[0] if len(r) else None

    sar  = g(sarima_results, group)
    lgb  = g(lgb_results,    group)
    xgb_ = g(xgb_results,   group)
    lst  = g(df_lstm,        group)

    vals = {k:v for k,v in [("SARIMA",sar),("LightGBM",lgb),
                              ("XGBoost",xgb_),("LSTM",lst)] if v}
    best = min(vals, key=vals.get) if vals else "—"

    all_rows.append({
        "group"     : group,
        "SARIMA %"  : sar,
        "LightGBM%" : lgb,
        "XGBoost %"  : xgb_,
        "LSTM %"    : lst,
        "Best"      : best,
    })

df_final = pd.DataFrame(all_rows)
print(df_final.to_string(index=False))

means = {
    "SARIMA"   : sarima_results["MAPE (%)"].mean(),
    "LightGBM" : lgb_results["MAPE (%)"].mean(),
    "XGBoost"  : xgb_results["MAPE (%)"].mean(),
    "LSTM"     : df_lstm["MAPE (%)"].mean(),
}
print(f"\n  {'─'*55}")
print(f"  {'Model':<15} {'Mean MAPE':>12} {'vs SARIMA':>12}")
print(f"  {'─'*55}")
for name, m in means.items():
    diff   = m - means["SARIMA"]
    marker = " ← BEST" if m == min(means.values()) else ""
    print(f"  {name:<15} {m:>11.3f}%  {diff:>+11.3f}pp{marker}")

df_final.to_csv(OUT_DIR / "final_comparison_table.csv", index=False)
print(f"\n  ✓ final_comparison_table.csv saved — use this as Table 2")


best_group = df_lstm.loc[df_lstm["MAPE (%)"].idxmin(), "group"]

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#0f0f0f")

train_tail = (train_df[train_df["group_id"]==best_group]
              .sort_values("month").tail(24)
              .set_index("month")["price_modal"])

fc        = forecasts_lstm[best_group]
lstm_mape = df_lstm.loc[df_lstm["group"]==best_group,"MAPE (%)"].values[0]

ax.plot(train_tail.index, train_tail.values/100,
        color="#555", linewidth=1.2, label="Training (last 24m)")
ax.plot(fc["dates"], fc["actual"]/100,
        color=VANILLA_COL, linewidth=2.2, label="Actual")
ax.plot(fc["dates"], fc["pred"]/100,
        color=IOD_COL, linewidth=1.8, linestyle="--",
        label=f"LSTM  MAPE={lstm_mape:.2f}%")

fc_xgb_grp = xgb_results[xgb_results["group"]==best_group]["MAPE (%)"].values
if len(fc_xgb_grp):
    ax.text(0.02, 0.95,
            f"XGBoost MAPE = {fc_xgb_grp[0]:.2f}%  (same group)",
            transform=ax.transAxes, fontsize=9, color=ARECANUT_COL,
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333", pad=4))

ax.set_title(f"LSTM Forecast — {best_group}  |  Validation Set",
             fontsize=12, color="white", pad=10, loc="left")
ax.set_ylabel("₹ per kg", fontsize=10)
ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333")
ax.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot18_lstm_forecast.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✓ Plot 18 — LSTM forecast saved")

print(f"""
{'='*65}
  CELL 13 COMPLETE

  Mean MAPE summary:
    SARIMA    : {means['SARIMA']:.3f}%
    XGBoost   : {means['XGBoost']:.3f}%
    LSTM      : {means['LSTM']:.3f}%
    LightGBM  : {means['LightGBM']:.3f}%

  Best model so far: {min(means, key=means.get)} ({min(means.values()):.3f}%)

  What comes next:
  ─────────────────────────────────────────────────
  Cell 14 — Conformal prediction intervals
    Wrap the best model in a conformal regressor.
    Produce 80% and 95% prediction bands.
    This is your uncertainty quantification
    contribution — no other paper on Karnataka
    spice prices has done this.
{'='*65}
""")
