# Karnataka Spice Price Forecasting

Multivariate machine learning pipeline for district-level vanilla and arecanut wholesale price forecasting in Karnataka, India. Combines econometric baselines, gradient boosting, deep learning, adaptive uncertainty quantification, and a game-theoretic optimal selling policy derived from forecast intervals.

---

## Research Question

> Can a multivariate ML framework — integrating Karnataka mandi price records, Spices Board export flows, monsoon anomaly indices, and global commodity shocks — generate district-level, uncertainty-quantified price forecasts for vanilla and arecanut that outperform univariate time series baselines, and to what extent do climate and trade signals explain price volatility beyond historical momentum?

---

## Reportable Results

| Model | Mean MAPE (val) | vs SARIMA |
|---|---|---|
| SARIMA(1,1,1)(1,0,1,12) | 2.524% | — |
| **XGBoost** | **2.463%** | **−0.061pp** |
| LightGBM | 2.976% | +0.452pp |
| LSTM | 8.949% | +6.425pp |
| SARIMA+XGBoost Combination | TBD on run | — |

**Uncertainty quantification:**

| Method | Empirical coverage @ 80% | Gap |
|---|---|---|
| Static conformal (Cell 14) | 48.3% | 31.7pp |
| Adaptive conformal / ACI (Cell 15) | 73.3% | 6.7pp |
| GP posterior (Cell 16) | 53.3% | 26.7pp |

**Decision policy:** ACI-informed optimal selling rule earns ₹17,273,422 more than always-sell per 100 quintals over the test period.

---

## Six Formal Findings

**Finding 1 — Commodity heterogeneity dominates algorithm choice**
XGBoost statistically outperforms SARIMA on all five arecanut districts (Diebold-Mariano, p<0.05). SARIMA statistically outperforms XGBoost on all five vanilla districts. The optimal model is commodity-specific, not universal.

**Finding 2 — Price momentum, not external signals, drives accuracy**
LightGBM feature importance: `momentum_6m` dominant. XGBoost feature importance: `rolling_mean_12m` dominant. SARIMA captures the same momentum structure parsimoniously. External climate and macro features improve arecanut but not vanilla forecasts.

**Finding 3 — LSTM requires more data than available**
LSTM MAPE 8.949% on 120-month training window (108 sequences after lookback). Consistent with minimum sample requirements for recurrent architectures documented in the literature.

**Finding 4 — Forecast combination addresses residual autocorrelation**
Ljung-Box test: 0/10 groups achieve white-noise residuals for any individual model. Optimal SARIMA-XGBoost combination improves white-noise count. Theoretically motivated by complementary error structures.

**Finding 5 — ACI is the correct uncertainty method under regime shifts**
The 2022-23 Madagascar vanilla supply collapse caused a structural break that violated exchangeability assumptions of static conformal prediction. ACI self-corrects coverage gap from 31.7pp to 6.7pp by updating the quantile at each step.

**Finding 6 — Nash-optimal selling policy**
Formal sell condition: `p_t ≥ lower_ACI(t+1) / (1 + 0.008)`. This is the maximin dominant strategy under the farmer-vs-market game formulation. No existing Karnataka spice price paper derives a decision rule from forecast uncertainty.

---

## Pipeline — 18 Cells

| Cell | Description | Key output |
|---|---|---|
| 0 | Package installation | pytorch-forecasting, optuna, lightning |
| 1 | Universal imports (Colab-safe, MAPIE-free) | ConformalRegressor class |
| 2 | Project config + data dictionary | 6 tables, 68 columns documented |
| 3 | Data ingestion | AGMARKNET API, World Bank, NOAA IOD, RBI |
| 4 | EDA — 8 dark-theme plots | price history, STL, heatmap, CCF |
| 5 | Stationarity tests | ADF, KPSS, ACF/PACF per group |
| 6 | Cross-correlation + Granger causality | lag structure of world price → mandi price |
| 7 | Calendar heatmap + Hurst exponent | H > 0.5 confirms long memory |
| 8 | Data quality audit | 0 missing after imputation |
| 9 | Feature engineering + train/val/test split | 26 tree features, 14 LSTM features |
| 10 | SARIMA baseline (walk-forward) | 2.524% MAPE — Table 1 benchmark |
| 11 | LightGBM + Optuna tuning | 2.976% MAPE, momentum_6m top feature |
| 12 | XGBoost + leakage-free walk-forward | 2.463% MAPE, arecanut gains confirmed |
| 13 | LSTM (2-layer, 64 hidden) | 8.949% MAPE — data limitation finding |
| 14 | Static conformal prediction | 48.3% empirical coverage |
| 15 | Adaptive conformal + game-theoretic policy | 73.3% coverage, ₹17M gain |
| 16 | Gaussian Process composite kernel | K = RBF + Periodic(12) + Matérn(0.5) + White |
| 17 | Robustness: DM, MCS, shock, MZ, Ljung-Box | 5 formal statistical tests |
| 18 | Forecast combination + paper tables | 5 publication-ready CSV tables |

---

## Plots — 23 Total

| Plot | Description |
|---|---|
| 1 | Dual-axis price history — vanilla vs arecanut (2010–2024) |
| 2 | District violin distributions |
| 3 | STL seasonal decomposition |
| 4 | Spearman correlation heatmap |
| 5 | Rolling 3-month volatility by district |
| 6 | IOD → vanilla price scatter (6-month lag) |
| 7 | Shock period box plots (pre-COVID / COVID / Madagascar / post) |
| 8 | Feature–target correlation bar chart |
| 9 | ACF + PACF (raw and Δlog) |
| 10 | Cross-correlation function (CCF) |
| 11 | Calendar heatmap (month × year) |
| 12 | Year-over-year growth + rolling Hurst exponent |
| 13 | SARIMA walk-forward forecast — best and worst group |
| 14 | LightGBM feature importance |
| 15 | LightGBM vs SARIMA forecast comparison |
| 16 | XGBoost feature importance |
| 17 | Three-way forecast: SARIMA vs LightGBM vs XGBoost |
| 18 | LSTM forecast — best group |
| 19 | Static conformal prediction intervals (80% and 95%) |
| 20 | Adaptive conformal intervals + sell signals (▲) |
| 21 | GP posterior with 80% and 95% bands |
| 22 | λ sweep — revenue vs risk aversion level |
| 23 | Final summary: MAPE comparison, DM wins, uncertainty calibration |

---

## Data Sources

| Source | Data | URL |
|---|---|---|
| AGMARKNET / CEDA Ashoka | Daily mandi modal prices — Karnataka | agmarknet.ceda.ashoka.edu.in |
| DASD | Area, production, productivity by crop year | dasd.gov.in/statistics |
| Spices Board of India | Monthly export volume and value | indianspices.com/export |
| RBI Handbook of Statistics | INR/USD, CPI, WPI (1967–2024) | rbi.org.in |
| World Bank Pink Sheet | International vanilla and pepper prices | worldbank.org |
| NOAA PSL | Indian Ocean Dipole (IOD) monthly index | psl.noaa.gov |

---

## Geography and Scope

- **Crops:** Vanilla, Arecanut
- **Districts:** Kodagu (Malnad), Shivamogga (Malnad), Uttara Kannada (Coastal), Chikkamagaluru (Hilly), Udupi (Coastal)
- **Period:** January 2010 – June 2024
- **Frequency:** Daily (raw) → Monthly (modelled)
- **Panel size:** 1,740 rows (10 groups × 174 months)
- **Train/Val/Test:** 1,200 / 240 / 300 rows

---

## Feature Engineering

**26 features for tree models:**
- Price lags: 1m, 3m, 6m, 12m
- Rolling statistics: vol_3m, vol_6m, mean_6m, mean_12m
- Momentum: 1m, 3m, 6m, distance from 6m mean
- Calendar: month_sin, month_cos, quarter, is_harvest_season
- Climate: iod_lag_3m, iod_lag_6m
- Macro: inr_usd_rate_lag_1m, crude_oil_brent_usd_lag_1m, vanilla_world_price_usd_kg_lag_1m
- Shocks: covid_shock, madagascar_shock
- Categorical: commodity_enc, district_enc, zone_enc

**14 features for LSTM** (subset — reduced for sequence modelling)

---

## How to Run

```bash
git clone https://github.com/hte-1313/karnataka-spice-price-forecasting
```

Open `spice_main.ipynb` in Google Colab, then:

1. Run **Cell 0** — installs packages
2. **Runtime → Restart session**
3. Run **Cells 1–18** in order

> Cells 10–12 each take 2–5 minutes due to walk-forward validation across 10 groups. Cell 16 (GP) takes 3–5 minutes for kernel hyperparameter optimisation.

---

## Repository Structure

```
karnataka-spice-price-forecasting/
│
├── spice_main.ipynb          ← main notebook (18 cells)
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── cell9_feature_engineering.py
│
├── eda/
│   ├── cell4_eda.py
│   ├── cell5_stationarity.py
│   ├── cell6_crosscorrelation.py
│   └── cell7_heatmap_hurst.py
│
├── audit/
│   └── cell8_data_quality.py
│
└── models/
    ├── cell10_sarima_baseline.py
    ├── cell11_lightgbm.py
    ├── cell12_xgboost.py
    ├── cell13_lstm.py
    ├── cell14_conformal.py
    ├── cell15_aci_decision.py
    ├── cell16_gp_kernel.py
    ├── cell17_robustness.py
    └── cell18_final.py
```



