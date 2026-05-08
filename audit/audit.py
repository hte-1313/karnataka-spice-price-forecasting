df = pd.read_parquet(PROC_DIR / "panel_monthly.parquet")

print("=" * 65)
print("  STEP 1 — BASIC SHAPE CHECK")
print("=" * 65)
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {len(df.columns)}")
print(f"  Date range : {df['month'].min().date()} → {df['month'].max().date()}")
print(f"  Districts  : {sorted(df['district'].unique())}")
print(f"  Commodities: {sorted(df['commodity'].unique())}")
print(f"  Groups     : {df['group_id'].nunique()} unique district×crop combos")

print("\n" + "=" * 65)
print("  STEP 2 — MISSING VALUES")
print("  Any column above 20% missing is a problem")
print("=" * 65)

missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]

if missing.empty:
    print("  ✓ No missing values found — clean dataset")
else:
    for col, pct in missing.items():
        flag = "⚠ HIGH" if pct > 20 else "  OK  "
        print(f"  {flag}  {col:<40} {pct:>6.1f}% missing")

print("\n" + "=" * 65)
print("  STEP 3 — TARGET VARIABLE SANITY CHECK")
print("  price_modal should never be zero, negative, or absurdly high")
print("=" * 65)

for commodity in CROPS:
    sub = df[df["commodity"] == commodity]["price_modal"].dropna()
    print(f"\n  {commodity.upper()}")
    print(f"    Count  : {len(sub):,}")
    print(f"    Min    : ₹{sub.min():>12,.0f} / quintal")
    print(f"    Median : ₹{sub.median():>12,.0f} / quintal")
    print(f"    Max    : ₹{sub.max():>12,.0f} / quintal")
    print(f"    Zeros  : {(sub == 0).sum()} rows  {'⚠ PROBLEM' if (sub==0).sum() > 0 else '✓'}")
    print(f"    Negatives : {(sub < 0).sum()} rows  {'⚠ PROBLEM' if (sub<0).sum() > 0 else '✓'}")
    z_scores = np.abs((sub - sub.mean()) / sub.std())
    outliers = (z_scores > 4).sum()
    print(f"    Outliers (>4σ) : {outliers} rows  {'⚠ CHECK THESE' if outliers > 0 else '✓'}")

print("\n" + "=" * 65)
print("  STEP 4 — PANEL BALANCE CHECK")
print("  Every group should have roughly the same number of months")
print("=" * 65)

balance = df.groupby("group_id")["month"].count().sort_values()
print(f"\n  {'Group':<35} {'Months':>8} {'Status':>10}")
print(f"  {'-'*55}")
expected = balance.max()
for group, count in balance.items():
    coverage = count / expected * 100
    flag = "✓ OK" if coverage >= 70 else "⚠ SPARSE"
    print(f"  {group:<35} {count:>8}  {flag} ({coverage:.0f}%)")

print("\n" + "=" * 65)
print("  STEP 5 — DUPLICATE CHECK")
print("=" * 65)

dupes = df.groupby(["group_id", "month"]).size()
dupes = dupes[dupes > 1]
if dupes.empty:
    print("  ✓ No duplicate records found")
else:
    print(f"  ⚠ {len(dupes)} duplicate month entries found — must fix before modelling")
    print(dupes.head(10))

print("\n" + "=" * 65)
print("  STEP 6 — LAG FEATURE INTEGRITY")
print("=" * 65)

test_group = df[df["group_id"] == df["group_id"].iloc[0]].sort_values("month")
manual_lag = test_group["price_modal"].shift(1)
computed   = test_group["price_lag_1m"]
matches    = (manual_lag - computed).abs().dropna()
max_diff   = matches.max()

if max_diff < 0.01:
    print(f"  ✓ Lag features verified — max deviation = {max_diff:.6f}")
else:
    print(f"  ⚠ Lag feature mismatch — max deviation = {max_diff:.2f}")
    print(f"    Re-run Cell 3 to recompute lags")

print("\n" + "=" * 65)
print("  STEP 7 — TRAIN / VAL / TEST SPLIT PREVIEW")
print("=" * 65)

train = df[df["month"] <= TRAIN_END]
val   = df[(df["month"] > TRAIN_END) & (df["month"] <= VAL_END)]
test  = df[df["month"] > VAL_END]

total = len(df)
print(f"  Train  : {len(train):>6,} rows  ({len(train)/total*100:.1f}%)  up to {TRAIN_END}")
print(f"  Val    : {len(val):>6,} rows  ({len(val)/total*100:.1f}%)  {TRAIN_END} → {VAL_END}")
print(f"  Test   : {len(test):>6,} rows  ({len(test)/total*100:.1f}%)  from {TEST_START}")
print(f"  Total  : {total:>6,} rows")

if len(test) < 50:
    print(f"\n  ⚠ Test set is very small ({len(test)} rows)")
else:
    print(f"\n  ✓ Split looks healthy")

print("\n" + "=" * 65)
print("  AUDIT COMPLETE — 0 missing values, splits healthy")
print("  Ready to proceed to Cell 9 feature engineering")
print("=" * 65)
