-- Behavioural repayment features in Snowflake SQL.
-- Mirrors src/credit/features.py::instalment_features so the feature logic can run
-- inside the warehouse (feature pipeline) instead of in pandas.
--
-- Load the Kaggle CSVs into RAW.INSTALLMENTS_PAYMENTS and RAW.APPLICATION_TRAIN first
-- (Snowsight: Data > Add data > Load files, or COPY INTO from a stage).

CREATE OR REPLACE TABLE FEATURES.INSTALMENT_BEHAVIOUR AS
WITH per_instalment AS (
    -- one row per instalment (an instalment can be paid in several parts)
    SELECT
        SK_ID_PREV,
        SK_ID_CURR,
        NUM_INSTALMENT_NUMBER,
        MIN(DAYS_INSTALMENT)     AS due_day,
        MAX(DAYS_ENTRY_PAYMENT)  AS paid_day,
        MAX(AMT_INSTALMENT)      AS amt_due,
        SUM(AMT_PAYMENT)         AS amt_paid
    FROM RAW.INSTALLMENTS_PAYMENTS
    WHERE DAYS_INSTALMENT < 0                       -- due before the application date
    GROUP BY 1, 2, 3
),
scored AS (
    SELECT
        *,
        -- payments made after the application were NOT known at decision time
        CASE WHEN paid_day IS NULL OR paid_day >= 0 THEN -due_day
             ELSE GREATEST(paid_day - due_day, 0) END                  AS dpd,
        LEAST(GREATEST(amt_paid / NULLIF(amt_due, 0), 0), 2)            AS pay_ratio,
        due_day >= -365                                                 AS is_recent
    FROM per_instalment
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                            AS inst_count,
    COUNT(DISTINCT SK_ID_PREV)                          AS inst_prev_loans,
    AVG(dpd)                                            AS inst_dpd_mean,
    MAX(dpd)                                            AS inst_dpd_max,
    AVG(IFF(dpd > 0, 1, 0))                             AS inst_late_share,
    AVG(IFF(dpd > 30, 1, 0))                            AS inst_late30_share,
    AVG(pay_ratio)                                      AS inst_pay_ratio_mean,
    AVG(IFF(pay_ratio < 0.95, 1, 0))                    AS inst_underpaid_share,
    AVG(IFF(is_recent, IFF(dpd > 0, 1, 0), NULL))       AS inst_late_share_12m,
    MAX(IFF(is_recent, dpd, NULL))                      AS inst_dpd_max_12m
FROM scored
GROUP BY SK_ID_CURR;


-- Monthly roll-rate matrix from POS_CASH_balance (same logic as src/credit/dpd_ews.py)
CREATE OR REPLACE VIEW FEATURES.ROLL_RATES AS
WITH b AS (
    SELECT
        SK_ID_PREV, MONTHS_BALANCE,
        CASE WHEN SK_DPD = 0 THEN 'Current'
             WHEN SK_DPD <= 30 THEN '1-30'
             WHEN SK_DPD <= 60 THEN '31-60'
             WHEN SK_DPD <= 90 THEN '61-90'
             ELSE '90+' END AS bucket
    FROM RAW.POS_CASH_BALANCE
    WHERE NAME_CONTRACT_STATUS = 'Active'
),
t AS (
    SELECT
        bucket AS from_bucket,
        LEAD(bucket)         OVER (PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE) AS to_bucket,
        LEAD(MONTHS_BALANCE) OVER (PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE) - MONTHS_BALANCE AS gap
    FROM b
)
SELECT
    from_bucket, to_bucket,
    COUNT(*) AS transitions,
    COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY from_bucket) AS roll_rate
FROM t
WHERE gap = 1
GROUP BY 1, 2;
