readme = """
# Karnataka Spice Price Forecasting

**Research Question:**
Can a multivariate machine learning framework — integrating Karnataka mandi
price records, Spices Board export flows, monsoon anomaly indices, and global
commodity shocks — generate district-level, uncertainty-quantified price
forecasts for vanilla and arecanut that outperform univariate time series
baselines?

---

## Project Status

| Phase | Status |
|---|---|
| Cell 0 — Installation | ✅ Complete |
| Cell 1 — Imports | ✅ Complete |
| Cell 2 — Config & Data Dictionary | ✅ Complete |
| Cell 3 — Data Ingestion | ✅ Complete |
| Cell 4 — EDA (8 plots) | ✅ Complete |
| Cell 5 — Stationarity Tests | ✅ Complete |
| Cell 6 — Cross-Correlation & Granger | ✅ Complete |
| Cell 7 — Calendar Heatmap & Hurst | ✅ Complete |
| Cell 8 — Baseline Models (ARIMA) | 🔲 Pending |
| Cell 9 — LightGBM / XGBoost | 🔲 Pending |
| Cell 10 — LSTM | 🔲 Pending |
| Cell 11 — Temporal Fusion Transformer | 🔲 Pending |
| Cell 12 — Conformal Prediction Intervals | 🔲 Pending |
| Cell 13 — Zone Heterogeneity Analysis | 🔲 Pending |
| Cell 14 — Shock Period Evaluation | 🔲 Pending |

---

## Data Sources

| Source | What | URL |
|---|---|---|
| AGMARKNET / CEDA Ashoka | Daily mandi prices (Karnataka) | agmarknet.ceda.ashoka.edu.in |
| DASD | Arecanut & spice production (area, tonnes) | dasd.gov.in/statistics |
| Spices Board of India | Monthly export volume & value | indianspices.com/export |
| RBI Handbook | Exchange rate, CPI, WPI (1967–2024) | rbi.org.in |
| World Bank Pink Sheet | International vanilla & pepper prices | worldbank.org |
| NOAA | Indian Ocean Dipole (IOD) index | psl.noaa.gov |

---

## Crops & Geography

- **Crops:** Vanilla, Arecanut
- **Districts:** Kodagu, Shivamogga, Uttara Kannada, Chikkamagaluru, Udupi
- **Period:** January 2010 – June 2024
- **Frequency:** Daily (raw) → Monthly (modelled)

---

## Models (planned)

1. SARIMA / SARIMAX — econometric baseline
2. LightGBM + XGBoost — gradient boosting with lag features
3. LSTM — deep learning for long-range temporal dependence
4. Temporal Fusion Transformer — multi-horizon with attention
5. Conformal Prediction — uncertainty intervals (coverage guarantee)

---

## Key EDA Findings (so far)

- `price_lag_1m` Spearman ρ = **+0.950** — dominant momentum signal
- `rolling_mean_6m` ρ = **+0.933** — strong trend persistence
- `madagascar_shock` ρ = **-0.386** — only negative feature
- Hurst exponent analysis pending (expected H > 0.5 → long memory confirmed)

---

## How to Run

1. Open in Google Colab
2. Run Cell 0 (installs)
3. Restart runtime
4. Run Cells 1–7 in order

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/karnataka-spice-price-forecasting
```

---

## Author

MSc / Research project — Karnataka agricultural commodity price forecasting  
Supervisor: [Name]  
Institution: [Name]
"""

with open(f"{GITHUB_REPO}/README.md", "w") as f:
    f.write(readme)

print("✓ README.md written")
