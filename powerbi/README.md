# Power BI dashboard: Credit Risk & Early Warning

All tables are exported to `powerbi/data/` by the pipeline (`python run_all.py`).
Open Power BI Desktop → **Get data → Text/CSV** (or **Folder** and pick `powerbi/data`) and load every file.

## Data model

| Table | Grain | Relationship |
|---|---|---|
| `fact_applications` | one row per scored application | `grade` → `risk_grades[grade]` (many-to-one) |
| `risk_grades` | one row per grade A–E | dimension |
| `cutoff_strategy` | one row per candidate cut-off | standalone (what-if table) |
| `model_comparison` | one row per model | standalone |
| `delinquency_trend` | one row per month | standalone |
| `roll_rates` | from-bucket × to-bucket | standalone (matrix visual) |
| `vintage_curves` | cohort × months on book | standalone |
| `ews_backtest` | one row per RAG status | standalone |
| `ews_watchlist` | one row per active account | standalone |
| `monitoring_monthly` | one row per production month | dimension on `month` |
| `monitoring_csi` | month × feature | `month` → `monitoring_monthly[month]` |
| `monitoring_alerts` | one row per alert | `month` → `monitoring_monthly[month]` |
| `fraud_daily` | one row per day | dimension on `day` |
| `fraud_alerts` | alerts + missed fraud | `day` → `fraud_daily[day]` |
| `fraud_rules`, `fraud_thresholds` | small lookup tables | standalone |

Set data types after loading: rates and PDs as **Percentage**; `AMT_CREDIT`, `expected_loss`, `recommended_limit` as **Currency**.
Sort `risk_grades[grade]` A→E and the DPD buckets in `roll_rates` with a sort-order column: Current=1, 1-30=2, 31-60=3, 61-90=4, 90+=5.

## DAX measures

Create a blank table called `_Measures` and add these measures to it.

```DAX
// ---------- Portfolio & decisioning (fact_applications)
Applications = COUNTROWS ( fact_applications )

Approved = CALCULATE ( [Applications], fact_applications[approved_at_cutoff] = 1 )

Approval Rate = DIVIDE ( [Approved], [Applications] )

Default Rate = AVERAGE ( fact_applications[TARGET] )

Default Rate (Approved) =
    CALCULATE ( [Default Rate], fact_applications[approved_at_cutoff] = 1 )

Defaults Avoided =
    CALCULATE ( SUM ( fact_applications[TARGET] ), fact_applications[approved_at_cutoff] = 0 )

Avg PD = AVERAGE ( fact_applications[pd_scorecard] )

Calibration Ratio = DIVIDE ( [Default Rate], [Avg PD] )   // 1.0 = perfectly calibrated

Exposure (EAD) = SUM ( fact_applications[AMT_CREDIT] )

Expected Loss = SUM ( fact_applications[expected_loss] )   // PD x LGD x EAD

Expected Loss Rate = DIVIDE ( [Expected Loss], [Exposure (EAD)] )

Thin-file Share =
    DIVIDE ( CALCULATE ( [Applications], fact_applications[THIN_FILE] = 1 ), [Applications] )

Avg Score = AVERAGE ( fact_applications[score] )

// ---------- Delinquency
Latest PAR30 =
    VAR m = MAX ( delinquency_trend[MONTHS_BALANCE] )
    RETURN CALCULATE ( MAX ( delinquency_trend[PAR30] ), delinquency_trend[MONTHS_BALANCE] = m )

Latest PAR90 =
    VAR m = MAX ( delinquency_trend[MONTHS_BALANCE] )
    RETURN CALCULATE ( MAX ( delinquency_trend[PAR90] ), delinquency_trend[MONTHS_BALANCE] = m )

Cure Rate 1-30 =
    CALCULATE ( SUM ( roll_rates[roll_rate] ),
        roll_rates[from_bucket] = "1-30", roll_rates[to_bucket] = "Current" )

// ---------- Early warning
Red Accounts = CALCULATE ( COUNTROWS ( ews_watchlist ), ews_watchlist[RAG] = "Red" )

Amber Accounts = CALCULATE ( COUNTROWS ( ews_watchlist ), ews_watchlist[RAG] = "Amber" )

Red Hit Rate (backtest) =
    CALCULATE ( MAX ( ews_backtest[hit_rate] ), ews_backtest[RAG] = "Red" )

// ---------- Monitoring
Latest Month = MAX ( monitoring_monthly[month] )

Latest Score PSI =
    CALCULATE ( MAX ( monitoring_monthly[score_psi] ),
        monitoring_monthly[month] = [Latest Month] )

Latest Status =
    CALCULATE ( SELECTEDVALUE ( monitoring_monthly[overall_status] ),
        monitoring_monthly[month] = [Latest Month] )

Red Alerts = CALCULATE ( COUNTROWS ( monitoring_alerts ), monitoring_alerts[status] = "Red" )

Status Colour =   // use as field value for conditional formatting
    SWITCH ( SELECTEDVALUE ( monitoring_monthly[overall_status] ),
        "Red", "#D55E00", "Amber", "#E69F00", "#009E73" )

CSI Colour =
    SWITCH ( SELECTEDVALUE ( monitoring_csi[csi_status] ),
        "Red", "#D55E00", "Amber", "#E69F00", "#009E73" )

// ---------- Fraud operations
Fraud Alerts = SUM ( fraud_daily[alerts] )

Alerts per Day = AVERAGE ( fraud_daily[alerts] )

Fraud Precision = DIVIDE ( SUM ( fraud_daily[caught] ), SUM ( fraud_daily[alerts] ) )

Fraud Recall = DIVIDE ( SUM ( fraud_daily[caught] ), SUM ( fraud_daily[fraud_cases] ) )

Fraud Value Caught =
    CALCULATE ( SUM ( fraud_alerts[amount] ), fraud_alerts[outcome] = "True positive" )

Fraud Value Missed =
    CALCULATE ( SUM ( fraud_alerts[amount] ), fraud_alerts[outcome] = "Missed fraud" )
```

## Report pages

**1. Portfolio overview** (for leadership)
- KPI cards: Applications, Approval Rate, Default Rate (Approved), Expected Loss Rate, Latest PAR30, Thin-file Share
- Clustered bar: Applications and Default Rate by `risk_grades[grade]`
- Table: grade, decision, pricing tier, limit multiple, predicted PD vs actual DR (from `risk_grades`)
- Slicers: `split`, `NAME_INCOME_TYPE`, `THIN_FILE`

**2. Credit model performance** (for risk)
- Table: `model_comparison` (AUC, Gini, KS, Brier, HL p-value), champion highlighted
- Line: approval rate vs bad rate approved across `cutoff_strategy[cutoff_score]`, plus a card for the recommended cut-off
- Column: `risk_grades` predicted PD vs actual DR (calibration by grade)
- Image visuals: `reports/figures/shap_beeswarm.png` and `roc_comparison.png`
- Table: `fact_applications` top 50 declines with `reason_codes` (adverse-action reasons)

**3. Delinquency & early warning** (for collections)
- Line: PAR30 and PAR90 over `MONTHS_BALANCE`
- Matrix: `roll_rates` (rows from_bucket, columns to_bucket, values roll_rate, background colour scale)
- Line: `vintage_curves` ever30_rate by MOB, legend COHORT
- Column: `ews_backtest` hit_rate by RAG, coloured Red/Amber/Green
- Table: `ews_watchlist` filtered to RAG = Red/Amber: account, behaviour_pd_6m, signals_triggered, recommended_action

**4. Model monitoring** (monthly health report)
- KPI cards: Latest Status (coloured by Status Colour), Latest Score PSI, Red Alerts
- Column: score_psi by month, conditional colour by Status Colour; constant lines at 0.10 and 0.25
- Line: gini and ks by month (matured months only) with validation Gini as a constant line
- Line: mean_pd vs actual_default_rate by month (calibration)
- Matrix heatmap: `monitoring_csi` feature × month, values csi, background colour by CSI Colour
- Table: `monitoring_alerts`

**5. Fraud operations**
- KPI cards: Alerts per Day, Fraud Precision, Fraud Recall, Fraud Value Caught
- Line: alerts vs fraud_cases vs caught by day
- Line: `fraud_thresholds` precision and recall vs alerts_per_day (threshold tuning)
- Table: `fraud_rules` with hard_block flag
- Table: `fraud_alerts` filtered to `outcome` = True positive / False positive

Footer on every page: *Portfolio prototype on public data (Home Credit, PaySim). Not a production lending system.*
