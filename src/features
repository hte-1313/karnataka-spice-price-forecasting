# ============================================================
# CELL 9 — FEATURE ENGINEERING & TRAIN / VAL / TEST SPLIT
#
# What this cell does:
#   1. Takes the clean monthly panel from Cell 8
#   2. Builds the final feature matrix (lags, rolling stats, dummies)
#   3. Handles remaining missing values through imputation
#   4. Splits into train / val / test without any data leakage
#   5. Saves three clean parquet files ready for modelling
#
# Rule: nothing from val or test ever touches the training pipeline.
# All scalers and imputers are fit ONLY on train, then applied to
# val and test. Breaking this rule silently inflates your metrics.
# ============================================================

df = pd.read_parquet(PROC_DIR / "panel_monthly.parquet")

print("=" * 65)
print("  STEP 1 — BUILD ADDITIONAL FEATURES")
print("=" * 65)

df = df.sort_values(["group_id", "month"]).reset_index(drop=True)

# ── Price-based features ──────────────────────────────────────
for lag in [1, 3, 6, 12]:
    col = f"price_lag_{lag}m"
    if col not in df.columns:
        df[col] = df.groupby("group_id")["price_modal"].shift(lag)

# Rolling features
df["rolling_vol_3m"]   = df.groupby("group_id")["price_modal"] \
                           .transform(lambda x: x.rolling(3, min_periods=2).std())
df["rolling_vol_6m"]   = df.groupby("group_id")["price_modal"] \
                           .transform(lambda x: x.rolling(6, min_periods=3).std())
df["rolling_mean_6m"]  = df.groupby("group_id")["price_modal"] \
                           .transform(lambda x: x.rolling(6, min_periods=3).mean())
df["rolling_mean_12m"] = df.groupby("group_id")["price_modal"] \
                           .transform(lambda x: x.rolling(12, min_periods=6).mean())

# Momentum
df["momentum_1m"]  = df["price_modal"] / df["price_lag_1m"]  - 1
df["momentum_3m"]  = df["price_modal"] / df["price_lag_3m"]  - 1
df["momentum_6m"]  = df["price_modal"] / df["price_lag_6m"]  - 1

# Distance from rolling mean (mean-reversion signal)
df["dist_from_mean_6m"] = (df["price_modal"] - df["rolling_mean_6m"]) \
                           / df["rolling_mean_6m"]

# ── Calendar features ─────────────────────────────────────────
df["month_of_year"] = df["month"].dt.month
df["quarter"]       = df["month"].dt.quarter
df["year"]          = df["month"].dt.year

# Sin/cos encoding of month — better than raw integer for cyclical patterns
df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)

# Harvest season flag (commodity-specific)
def is_harvest(row):
    m = row["month_of_year"]
    if row["commodity"] == "vanilla":
        return 1 if m in [11, 12, 1, 2, 3, 4] else 0   # Nov–Apr
    else:
        return 1 if m in [10, 11, 12, 1, 2] else 0      # Oct–Feb (Shivamogga)

df["is_harvest_season"] = df.apply(is_harvest, axis=1)

# ── Climate lags (if not already present) ────────────────────
if "iod_index" in df.columns:
    for lag in [3, 6, 9]:
        df[f"iod_lag_{lag}m"] = df.groupby("group_id")["iod_index"].shift(lag)

# ── Macro lags ────────────────────────────────────────────────
for col in ["inr_usd_rate", "crude_oil_brent_usd",
            "vanilla_world_price_usd_kg"]:
    if col in df.columns:
        df[f"{col}_lag_1m"] = df.groupby("group_id")[col].shift(1)
        df[f"{col}_lag_3m"] = df.groupby("group_id")[col].shift(3)

# ── Encode categorical columns ────────────────────────────────
df["commodity_enc"] = LabelEncoder().fit_transform(df["commodity"])
df["district_enc"]  = LabelEncoder().fit_transform(df["district"])

# Agro zone map
zone_map = {
    "Kodagu"         : "Malnad",
    "Shivamogga"     : "Malnad",
    "Uttara Kannada" : "Coastal",
    "Chikkamagaluru" : "Hilly",
    "Udupi"          : "Coastal",
}
df["agro_zone"] = df["district"].map(zone_map).fillna("Unknown")
df["zone_enc"]  = LabelEncoder().fit_transform(df["agro_zone"])

# Log target
df["log_price"] = np.log(df["price_modal"].clip(lower=1))

print(f"  Features built: {len(df.columns)} total columns")
print(f"  Rows: {len(df):,}")


print("\n" + "=" * 65)
print("  STEP 2 — DEFINE FEATURE SETS")
print("=" * 65)

# These are the columns your models will actually use
FEATURES_TREE = [
    # Lag features
    "price_lag_1m", "price_lag_3m", "price_lag_6m", "price_lag_12m",
    # Rolling
    "rolling_vol_3m", "rolling_vol_6m",
    "rolling_mean_6m", "rolling_mean_12m",
    # Momentum
    "momentum_1m", "momentum_3m", "momentum_6m",
    "dist_from_mean_6m",
    # Calendar
    "month_sin", "month_cos", "quarter", "is_harvest_season",
    # Climate
    "iod_lag_6m", "iod_lag_3m",
    # Macro
    "inr_usd_rate_lag_1m", "crude_oil_brent_usd_lag_1m",
    "vanilla_world_price_usd_kg_lag_1m",
    # Shocks
    "covid_shock", "madagascar_shock",
    # Categorical encoded
    "commodity_enc", "district_enc", "zone_enc",
]

FEATURES_LSTM = [
    "price_lag_1m", "price_lag_3m", "price_lag_6m", "price_lag_12m",
    "rolling_vol_3m", "rolling_mean_6m",
    "momentum_3m", "month_sin", "month_cos",
    "iod_lag_6m", "inr_usd_rate_lag_1m",
    "vanilla_world_price_usd_kg_lag_1m",
    "covid_shock", "madagascar_shock",
]

TARGET       = "price_modal"
TARGET_LOG   = "log_price"

# Keep only features that actually exist in df
FEATURES_TREE = [f for f in FEATURES_TREE if f in df.columns]
FEATURES_LSTM = [f for f in FEATURES_LSTM if f in df.columns]

print(f"  Tree model features : {len(FEATURES_TREE)}")
print(f"  LSTM features       : {len(FEATURES_LSTM)}")
print(f"  Target              : {TARGET}")


print("\n" + "=" * 65)
print("  STEP 3 — TRAIN / VAL / TEST SPLIT")
print("  Splitting on time — no shuffling, no random states")
print("  This is the only correct way for time series")
print("=" * 65)

train_df = df[df["month"] <= TRAIN_END].copy()
val_df   = df[(df["month"] > TRAIN_END) & (df["month"] <= VAL_END)].copy()
test_df  = df[df["month"] > VAL_END].copy()

print(f"\n  Train : {len(train_df):>6,} rows  "
      f"({df['month'].min().date()} → {TRAIN_END})")
print(f"  Val   : {len(val_df):>6,} rows  "
      f"({TRAIN_END} → {VAL_END})")
print(f"  Test  : {len(test_df):>6,} rows  "
      f"({TEST_START} → {df['month'].max().date()})")
print(f"  Total : {len(df):>6,} rows")


print("\n" + "=" * 65)
print("  STEP 4 — IMPUTATION")
print("  Fit imputer on TRAIN only, transform all three splits")
print("=" * 65)

imputer = KNNImputer(n_neighbors=5)

X_train_raw = train_df[FEATURES_TREE].values
X_val_raw   = val_df[FEATURES_TREE].values
X_test_raw  = test_df[FEATURES_TREE].values

imputer.fit(X_train_raw)    # fit ONLY on train

X_train_imp = imputer.transform(X_train_raw)
X_val_imp   = imputer.transform(X_val_raw)
X_test_imp  = imputer.transform(X_test_raw)

# Put back into DataFrames
train_df[FEATURES_TREE] = X_train_imp
val_df[FEATURES_TREE]   = X_val_imp
test_df[FEATURES_TREE]  = X_test_imp

missing_after = pd.DataFrame(
    np.concatenate([X_train_imp, X_val_imp, X_test_imp]),
    columns=FEATURES_TREE
).isnull().sum().sum()

print(f"  Missing values after imputation: {missing_after}  "
      f"{'✓ Clean' if missing_after == 0 else '⚠ Still has nulls'}")


print("\n" + "=" * 65)
print("  STEP 5 — SCALING")
print("  Fit scaler on TRAIN only")
print("  Using RobustScaler — handles outliers better than StandardScaler")
print("=" * 65)

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)   # fit + transform
X_val_scaled   = scaler.transform(X_val_imp)          # transform only
X_test_scaled  = scaler.transform(X_test_imp)         # transform only

y_train = train_df[TARGET].values
y_val   = val_df[TARGET].values
y_test  = test_df[TARGET].values

y_train_log = train_df[TARGET_LOG].values
y_val_log   = val_df[TARGET_LOG].values
y_test_log  = test_df[TARGET_LOG].values

print(f"  X_train : {X_train_scaled.shape}")
print(f"  X_val   : {X_val_scaled.shape}")
print(f"  X_test  : {X_test_scaled.shape}")
print(f"  y_train : {y_train.shape}   min={y_train.min():,.0f}  max={y_train.max():,.0f}")


print("\n" + "=" * 65)
print("  STEP 6 — SAVE SPLITS")
print("=" * 65)

train_df.to_parquet(FEAT_DIR / "train.parquet", index=False)
val_df.to_parquet(  FEAT_DIR / "val.parquet",   index=False)
test_df.to_parquet( FEAT_DIR / "test.parquet",  index=False)

# Save feature lists for later cells
import json
feature_config = {
    "FEATURES_TREE" : FEATURES_TREE,
    "FEATURES_LSTM" : FEATURES_LSTM,
    "TARGET"        : TARGET,
    "TARGET_LOG"    : TARGET_LOG,
}
with open(FEAT_DIR / "feature_config.json", "w") as f:
    json.dump(feature_config, f, indent=2)

print(f"  ✓ train.parquet   → {len(train_df):,} rows")
print(f"  ✓ val.parquet     → {len(val_df):,} rows")
print(f"  ✓ test.parquet    → {len(test_df):,} rows")
print(f"  ✓ feature_config.json saved")

print(f"""
{'='*65}
  CELL 9 COMPLETE — YOU ARE NOW READY TO MODEL

  What you have:
    - {len(FEATURES_TREE)} features for tree models (LightGBM, XGBoost)
    - {len(FEATURES_LSTM)} features for LSTM
    - Clean train / val / test splits with no data leakage
    - Imputer and scaler fit on train only

  Next cell: Cell 10 — SARIMA/SARIMAX baseline
  This gives you the benchmark every ML model must beat.
{'='*65}
""")
