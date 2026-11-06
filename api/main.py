"""Credit decisioning and fraud-check API (portfolio prototype).

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs

Endpoints
  GET  /health             liveness + model versions
  GET  /model-card         validation metrics, features, policy, retraining triggers
  POST /score              one application -> score, PD, grade, decision, limit, reason codes
  POST /score/batch        many applications
  POST /fraud/score        one transaction -> fraud score, rules hit, alert yes/no

Every decision is appended to logs/scoring_log.jsonl so production monitoring can
compute PSI on real traffic and join outcomes later.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.config import ARTIFACTS, ROOT, LGD
from src.credit.features import application_features, CATEGORICAL
from src.credit.preprocess import assign_grade, grade_policy

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Credit Risk & Fraud Scoring API",
              description="Portfolio prototype on public data (Home Credit, PaySim). Not a production lending system.",
              version="1.0.0")

_credit = joblib.load(ARTIFACTS / "credit_models.joblib")
SC, PREP, XGB, META = _credit["scorecard"], _credit["prep"], _credit["xgb"], _credit["meta"]
_fraud_path = ARTIFACTS / "fraud_models.joblib"
FRAUD = joblib.load(_fraud_path) if _fraud_path.exists() else None
_explainer = None


def explainer():
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(XGB)
    return _explainer


# ---------------------------------------------------------------- schemas
class Application(BaseModel):
    """Raw application fields (Home Credit names) + optional pre-computed history features,
    as they would come from a feature store. Missing history = new / thin-file customer."""
    SK_ID_CURR: int = Field(..., examples=[100002])
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    FLAG_OWN_CAR: Literal["Y", "N"] = "N"
    FLAG_OWN_REALTY: Literal["Y", "N"] = "Y"
    CNT_CHILDREN: int = Field(0, ge=0)
    AMT_INCOME_TOTAL: float = Field(..., gt=0, examples=[202500])
    AMT_CREDIT: float = Field(..., gt=0, examples=[406597.5])
    AMT_ANNUITY: float = Field(..., gt=0, examples=[24700.5])
    AMT_GOODS_PRICE: Optional[float] = Field(None, examples=[351000])
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    DAYS_BIRTH: int = Field(..., lt=0, examples=[-9461], description="Days before application (negative)")
    DAYS_EMPLOYED: int = Field(..., examples=[-637], description="Negative days; 365243 = not employed")
    DAYS_ID_PUBLISH: int = Field(-2000, le=0)
    DAYS_LAST_PHONE_CHANGE: float = Field(-1000, le=0)
    OCCUPATION_TYPE: Optional[str] = None
    REGION_RATING_CLIENT: int = Field(2, ge=1, le=3)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    # ---- behavioural (previous loans) and bureau features, optional
    INST_COUNT: float = 0
    INST_PREV_LOANS: float = 0
    INST_DPD_MEAN: Optional[float] = None
    INST_DPD_MAX: Optional[float] = None
    INST_LATE_SHARE: Optional[float] = None
    INST_LATE30_SHARE: Optional[float] = None
    INST_PAY_RATIO_MEAN: Optional[float] = None
    INST_UNDERPAID_SHARE: Optional[float] = None
    INST_LATE_SHARE_12M: Optional[float] = None
    INST_DPD_MAX_12M: Optional[float] = None
    POS_MONTHS: float = 0
    POS_DPD_MAX: Optional[float] = None
    POS_DPD_MONTH_SHARE: Optional[float] = None
    POS_DPD_MAX_12M: Optional[float] = None
    BUR_COUNT: float = 0
    BUR_ACTIVE_COUNT: float = 0
    BUR_DEBT_SUM: Optional[float] = None
    BUR_OVERDUE_ANY: Optional[float] = None
    BUR_OVERDUE_AMT: Optional[float] = None
    BUR_DEBT_RATIO: Optional[float] = None


class Decision(BaseModel):
    request_id: str
    applicant_id: int
    model_version: str
    score: int
    pd_12m: float
    risk_grade: str
    decision: str
    pricing_tier: str
    recommended_limit: float
    expected_loss: float
    thin_file: bool
    reason_codes: list[str]
    challenger_pd_xgboost: float
    challenger_top_drivers: list[str]
    scored_at: str


class Transaction(BaseModel):
    step: int = Field(..., ge=1, description="Hour index")
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(0, ge=0)
    newbalanceDest: float = Field(0, ge=0)
    # network context from the transaction history store
    dest_prior_txns: int = 0
    dest_prior_senders: int = 0
    dest_prior_amount: float = 0
    orig_received_before: bool = False


# ---------------------------------------------------------------- helpers
def to_frame(apps: list[Application]) -> pd.DataFrame:
    raw = pd.DataFrame([a.model_dump() for a in apps])
    if raw["AMT_GOODS_PRICE"].isna().any():
        raw["AMT_GOODS_PRICE"] = raw["AMT_GOODS_PRICE"].fillna(raw["AMT_CREDIT"])
    df = application_features(raw)
    df["NEW_TO_LENDER"] = (df["INST_COUNT"] == 0).astype(int)
    df["THIN_FILE"] = (df["BUR_COUNT"] == 0).astype(int)
    df["BUR_DEBT_TO_INCOME"] = df["BUR_DEBT_SUM"] / df["AMT_INCOME_TOTAL"]
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Missing").astype(str)
    for c in META["all_features"]:
        if c not in df:
            df[c] = np.nan
    num = [c for c in META["all_features"] if c not in CATEGORICAL]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


def shap_drivers(Xt: pd.DataFrame, top=3) -> list[list[str]]:
    sv = explainer().shap_values(Xt)
    out = []
    for row in np.atleast_2d(sv):
        idx = np.argsort(-row)[:top]
        out.append([f"{Xt.columns[i]} (+{row[i]:.2f} log-odds)" for i in idx if row[i] > 0])
    return out


def log(records: list[dict]):
    with open(LOG_DIR / "scoring_log.jsonl", "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")


def score_frame(df: pd.DataFrame) -> list[Decision]:
    feats = META["all_features"]
    X = df[feats]
    score = SC.score(X)
    pd_ = SC.predict_proba(X)
    grades = assign_grade(pd_)
    reasons = SC.reason_codes(X)
    Xt = PREP.transform(X)
    pd_x = XGB.predict_proba(Xt)[:, 1]
    drivers = shap_drivers(Xt)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for i in range(len(df)):
        pol = grade_policy(grades[i])
        limit = 0.0 if pol["decision"] == "Decline" else round(
            df["AMT_INCOME_TOTAL"].iloc[i] / 12 * pol["limit_x_monthly_income"], -2)
        out.append(Decision(
            request_id=str(uuid.uuid4()), applicant_id=int(df["SK_ID_CURR"].iloc[i]),
            model_version=META["model_version"], score=int(score[i]), pd_12m=round(float(pd_[i]), 5),
            risk_grade=grades[i], decision=pol["decision"], pricing_tier=pol["pricing_tier"],
            recommended_limit=limit, expected_loss=round(float(pd_[i] * LGD * df["AMT_CREDIT"].iloc[i]), 2),
            thin_file=bool(df["THIN_FILE"].iloc[i]), reason_codes=reasons[i],
            challenger_pd_xgboost=round(float(pd_x[i]), 5), challenger_top_drivers=drivers[i], scored_at=now))
    log([d.model_dump() for d in out])
    return out


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "credit_model": META["model_version"], "trained_at": META["trained_at"],
            "fraud_model_loaded": FRAUD is not None}


@app.get("/model-card")
def model_card():
    return META


@app.post("/score", response_model=Decision)
def score(application: Application):
    return score_frame(to_frame([application]))[0]


@app.post("/score/batch", response_model=list[Decision])
def score_batch(applications: list[Application]):
    if not applications:
        raise HTTPException(400, "Empty batch")
    if len(applications) > 1000:
        raise HTTPException(413, "Max 1000 applications per call; use src/batch_score.py for bulk runs")
    return score_frame(to_frame(applications))


@app.post("/fraud/score")
def fraud_score(t: Transaction):
    if FRAUD is None:
        raise HTTPException(503, "Fraud model not trained. Run: python -m src.fraud.train_fraud")
    f = {
        "is_transfer": int(t.type == "TRANSFER"), "amount_log": np.log1p(t.amount),
        "hour": (t.step - 1) % 24, "night": int(((t.step - 1) % 24) <= 5),
        "orig_balance_log": np.log1p(t.oldbalanceOrg),
        "amount_to_balance": min(t.amount / t.oldbalanceOrg, 10) if t.oldbalanceOrg > 0 else 10,
        "drains_account": int(t.newbalanceOrig == 0 and t.oldbalanceOrg > 0 and t.amount >= 0.99 * t.oldbalanceOrg),
        "orig_balance_error": t.oldbalanceOrg - t.amount - t.newbalanceOrig,
        "dest_balance_error": t.oldbalanceDest + t.amount - t.newbalanceDest,
        "dest_zero_balances": int(t.oldbalanceDest == 0 and t.newbalanceDest == 0),
        "dest_prior_txns": t.dest_prior_txns, "dest_prior_amount_log": np.log1p(t.dest_prior_amount),
        "dest_prior_senders": t.dest_prior_senders, "orig_received_before": int(t.orig_received_before),
    }
    X = pd.DataFrame([f])[FRAUD["features"]]
    p = float(FRAUD["xgb"].predict_proba(X)[:, 1][0])
    anomaly = float(-FRAUD["iso"].score_samples(X[FRAUD["iso_cols"]])[0])
    rules = []
    if f["drains_account"] and f["is_transfer"]:
        rules.append("R1_drain_transfer")
    if t.amount > 200_000 and f["is_transfer"]:
        rules.append("R2_large_transfer")
    if f["dest_zero_balances"]:
        rules.append("R3_dest_balance_unchanged")
    if t.dest_prior_senders >= 5:
        rules.append("R4_mule_fan_in")
    if f["night"] and f["drains_account"]:
        rules.append("R5_night_drain")
    hard = [r for r in rules if r in FRAUD.get("hard_rules", [])]
    alert = bool(hard) or p >= FRAUD["threshold"]
    return {"fraud_probability": round(p, 5), "anomaly_score": round(anomaly, 4), "rules_hit": rules,
            "alert": alert, "action": "Hold and review" if alert else "Allow",
            "reason": ("hard rule: " + ", ".join(hard)) if hard else
                      ("model score above threshold" if alert else "below threshold")}
