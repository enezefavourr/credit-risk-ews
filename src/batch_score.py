"""Batch scoring: score a CSV of applications overnight, the way a lending platform would
refresh limits and offers for its whole customer base.

Usage:
  python -m src.batch_score --input data/sample/application_train.csv --output scored.csv
The input needs the Home Credit application columns; history features are built from the
instalment / POS / bureau files in DATA_DIR when they exist.
"""
import argparse
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.common.config import ARTIFACTS, LGD
from src.credit.features import (application_features, instalment_features, pos_features,
                                 bureau_features, CATEGORICAL, APP_COLS)
from src.credit.preprocess import assign_grade, GRADES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    b = joblib.load(ARTIFACTS / "credit_models.joblib")
    sc, prep, xgbm, meta = b["scorecard"], b["prep"], b["xgb"], b["meta"]

    raw = pd.read_csv(args.input, usecols=lambda c: c in APP_COLS)
    if "TARGET" not in raw:
        raw["TARGET"] = pd.NA
    df = application_features(raw).set_index("SK_ID_CURR")
    for part in (instalment_features(), pos_features(), bureau_features()):
        if part is not None:
            df = df.join(part, how="left")
    for c in ["INST_COUNT", "INST_PREV_LOANS", "POS_MONTHS", "BUR_COUNT", "BUR_ACTIVE_COUNT"]:
        if c in df:
            df[c] = df[c].fillna(0)
    df["NEW_TO_LENDER"] = (df["INST_COUNT"] == 0).astype(int)
    if "BUR_DEBT_SUM" in df:
        df["BUR_DEBT_TO_INCOME"] = df["BUR_DEBT_SUM"] / df["AMT_INCOME_TOTAL"]
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Missing").astype(str)
    df = df.reset_index()
    X = df[meta["all_features"]]

    pol = pd.DataFrame(GRADES, columns=["pd_upper", "grade", "decision", "pricing_tier", "limit_x"])
    out = df[["SK_ID_CURR"]].copy()
    out["score"] = sc.score(X)
    out["pd_12m"] = sc.predict_proba(X)
    out["grade"] = assign_grade(out["pd_12m"])
    out = out.merge(pol[["grade", "decision", "pricing_tier", "limit_x"]], on="grade", how="left")
    out["recommended_limit"] = (df["AMT_INCOME_TOTAL"] / 12 * out["limit_x"]).round(-2)
    out["expected_loss"] = out["pd_12m"] * LGD * df["AMT_CREDIT"]
    out["challenger_pd_xgboost"] = xgbm.predict_proba(prep.transform(X))[:, 1]
    out["reason_codes"] = ["; ".join(r) for r in sc.reason_codes(X)]
    out["model_version"] = meta["model_version"]
    out["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.drop(columns="limit_x").to_csv(args.output, index=False)
    print(f"Scored {len(out):,} applications -> {args.output}")
    print(out["decision"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
