"""Feature engineering for application scoring.

Three feature families, mirroring what a lender would have at decision time:
  1. Application data      - affordability ratios, tenure, stability
  2. Behavioural history   - repayment behaviour on the applicant's PREVIOUS loans
                             (instalments + monthly POS balances): DPD, lateness, underpayment
  3. Bureau data           - other lenders' exposure and arrears; absence = thin file

Leakage control: only history dated BEFORE the application (day/month < 0) is used,
and payments made after the application date are treated as unpaid at decision time.
"""
import numpy as np
import pandas as pd

from src.common.config import DATA_DIR, EXCLUDED_FEATURES

APP_COLS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "OCCUPATION_TYPE",
    "REGION_RATING_CLIENT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

CATEGORICAL = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
               "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]


def load_application(path=None) -> pd.DataFrame:
    path = path or DATA_DIR / "application_train.csv"
    df = pd.read_csv(path, usecols=APP_COLS)
    return df


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    # 365243 is Home Credit's placeholder for "not employed / pensioner"
    df["EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"].replace(365243, np.nan)) / 365.25
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]      # debt-service ratio
    df["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_MISSING_CNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1)
    df["YEARS_ID_PUBLISH"] = -df["DAYS_ID_PUBLISH"] / 365.25
    df["YEARS_PHONE_CHANGE"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"])
    return df


def instalment_features(path=None) -> pd.DataFrame:
    """Repayment behaviour on previous loans, one row per applicant."""
    path = path or DATA_DIR / "installments_payments.csv"
    cols = ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    dtypes = {"SK_ID_PREV": "int32", "SK_ID_CURR": "int32", "NUM_INSTALMENT_NUMBER": "int16",
              **{c: "float32" for c in cols[3:]}}
    ins = pd.read_csv(path, usecols=cols, dtype=dtypes)          # compact dtypes: 13.6M rows on a laptop
    ins = ins[ins["DAYS_INSTALMENT"] < 0]                       # due before application date
    # A single instalment can be paid in several parts: collapse to one row per instalment
    g = ins.groupby(["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER"], sort=False)
    ins = g.agg(DUE=("DAYS_INSTALMENT", "min"), PAID_ON=("DAYS_ENTRY_PAYMENT", "max"),
                AMT_DUE=("AMT_INSTALMENT", "max"), AMT_PAID=("AMT_PAYMENT", "sum")).reset_index()
    # Anything paid after the application date was not yet paid when we decided
    unpaid = ins["PAID_ON"].isna() | (ins["PAID_ON"] >= 0)
    ins["DPD"] = np.where(unpaid, -ins["DUE"], (ins["PAID_ON"] - ins["DUE"]).clip(lower=0))
    ins["LATE"] = (ins["DPD"] > 0).astype("int8")
    ins["LATE_30"] = (ins["DPD"] > 30).astype("int8")
    ins["PAY_RATIO"] = (ins["AMT_PAID"] / ins["AMT_DUE"].replace(0, np.nan)).clip(0, 2)
    ins["UNDERPAID"] = (ins["PAY_RATIO"] < 0.95).astype("int8")
    ins["RECENT"] = ins["DUE"] >= -365

    agg = ins.groupby("SK_ID_CURR").agg(
        INST_COUNT=("DPD", "size"),
        INST_PREV_LOANS=("SK_ID_PREV", "nunique"),
        INST_DPD_MEAN=("DPD", "mean"),
        INST_DPD_MAX=("DPD", "max"),
        INST_LATE_SHARE=("LATE", "mean"),
        INST_LATE30_SHARE=("LATE_30", "mean"),
        INST_PAY_RATIO_MEAN=("PAY_RATIO", "mean"),
        INST_UNDERPAID_SHARE=("UNDERPAID", "mean"),
    )
    recent = ins[ins["RECENT"]].groupby("SK_ID_CURR").agg(
        INST_LATE_SHARE_12M=("LATE", "mean"), INST_DPD_MAX_12M=("DPD", "max"))
    return agg.join(recent, how="left")


def pos_features(path=None) -> pd.DataFrame:
    path = path or DATA_DIR / "POS_CASH_balance.csv"
    pos = pd.read_csv(path, usecols=["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"],
                      dtype={"SK_ID_CURR": "int32", "MONTHS_BALANCE": "int16", "SK_DPD": "float32"})
    pos["DPD_POS"] = (pos["SK_DPD"] > 0).astype("int8")
    agg = pos.groupby("SK_ID_CURR").agg(POS_MONTHS=("SK_DPD", "size"),
                                        POS_DPD_MAX=("SK_DPD", "max"),
                                        POS_DPD_MONTH_SHARE=("DPD_POS", "mean"))
    rec = pos[pos["MONTHS_BALANCE"] >= -12].groupby("SK_ID_CURR")["SK_DPD"].max().rename("POS_DPD_MAX_12M")
    return agg.join(rec, how="left")


def bureau_features(path=None) -> pd.DataFrame | None:
    path = path or DATA_DIR / "bureau.csv"
    if not path.exists():
        return None
    b = pd.read_csv(path, usecols=["SK_ID_CURR", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE",
                                   "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"])
    b["ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    agg = b.groupby("SK_ID_CURR").agg(
        BUR_COUNT=("ACTIVE", "size"), BUR_ACTIVE_COUNT=("ACTIVE", "sum"),
        BUR_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"), BUR_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUR_OVERDUE_ANY=("OVERDUE", "max"), BUR_OVERDUE_AMT=("AMT_CREDIT_SUM_OVERDUE", "sum"))
    agg["BUR_DEBT_RATIO"] = agg["BUR_DEBT_SUM"] / agg["BUR_CREDIT_SUM"].replace(0, np.nan)
    return agg.drop(columns=["BUR_CREDIT_SUM"])


def build_dataset(verbose=True) -> pd.DataFrame:
    app = application_features(load_application())
    parts = [instalment_features(), pos_features(), bureau_features()]
    df = app.set_index("SK_ID_CURR")
    for p in parts:
        if p is not None:
            df = df.join(p, how="left")
    # No history row = no previous loans with us (count zero, behaviour unknown)
    for c in ["INST_COUNT", "INST_PREV_LOANS", "POS_MONTHS", "BUR_COUNT", "BUR_ACTIVE_COUNT"]:
        if c in df:
            df[c] = df[c].fillna(0)
    df["THIN_FILE"] = (df.get("BUR_COUNT", 0) == 0).astype(int)
    df["NEW_TO_LENDER"] = (df["INST_COUNT"] == 0).astype(int)
    if "BUR_COUNT" in df:
        df["BUR_DEBT_TO_INCOME"] = df["BUR_DEBT_SUM"] / df["AMT_INCOME_TOTAL"]
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Missing").astype(str)
    if verbose:
        print(f"[features] {df.shape[0]:,} applicants x {df.shape[1]} columns from {DATA_DIR}")
    return df.reset_index()


def model_feature_list(df: pd.DataFrame) -> list[str]:
    drop = {"SK_ID_CURR", "TARGET", "THIN_FILE", "BUR_DEBT_SUM", *EXCLUDED_FEATURES}
    return [c for c in df.columns if c not in drop]
