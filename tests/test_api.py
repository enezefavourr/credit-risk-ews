"""Run:  pytest -q   (after training the models)"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

GOOD = {
    "SK_ID_CURR": 900001, "AMT_INCOME_TOTAL": 270000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 18000,
    "AMT_GOODS_PRICE": 450000, "DAYS_BIRTH": -16000, "DAYS_EMPLOYED": -3500,
    "NAME_EDUCATION_TYPE": "Higher education", "EXT_SOURCE_1": 0.75, "EXT_SOURCE_2": 0.72,
    "EXT_SOURCE_3": 0.70, "INST_COUNT": 24, "INST_PREV_LOANS": 2, "INST_DPD_MEAN": 0,
    "INST_DPD_MAX": 0, "INST_LATE_SHARE": 0, "INST_PAY_RATIO_MEAN": 1.0, "BUR_COUNT": 4,
    "BUR_ACTIVE_COUNT": 1, "BUR_DEBT_SUM": 50000, "BUR_OVERDUE_ANY": 0, "BUR_OVERDUE_AMT": 0,
}
RISKY = {
    "SK_ID_CURR": 900002, "AMT_INCOME_TOTAL": 90000, "AMT_CREDIT": 600000, "AMT_ANNUITY": 30000,
    "DAYS_BIRTH": -8200, "DAYS_EMPLOYED": -200, "REGION_RATING_CLIENT": 3,
    "EXT_SOURCE_2": 0.08, "EXT_SOURCE_3": 0.10, "INST_COUNT": 12, "INST_PREV_LOANS": 1,
    "INST_DPD_MEAN": 25, "INST_DPD_MAX": 95, "INST_LATE_SHARE": 0.6, "INST_UNDERPAID_SHARE": 0.4,
    "INST_PAY_RATIO_MEAN": 0.7,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_good_beats_risky():
    g = client.post("/score", json=GOOD).json()
    b = client.post("/score", json=RISKY).json()
    assert g["score"] > b["score"]
    assert g["pd_12m"] < b["pd_12m"]
    assert b["reason_codes"], "risky applicant must get reason codes"


def test_thin_file_is_scored():
    thin = {k: v for k, v in GOOD.items() if not k.startswith(("INST_", "BUR_"))}
    r = client.post("/score", json=thin)
    assert r.status_code == 200 and r.json()["thin_file"] is True


def test_validation_rejects_bad_input():
    assert client.post("/score", json={**GOOD, "AMT_INCOME_TOTAL": -5}).status_code == 422


def test_batch():
    r = client.post("/score/batch", json=[GOOD, RISKY])
    assert r.status_code == 200 and len(r.json()) == 2


def test_fraud_drain_is_flagged():
    t = {"step": 3, "type": "TRANSFER", "amount": 181000, "oldbalanceOrg": 181000,
         "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0, "dest_prior_senders": 6}
    r = client.post("/fraud/score", json=t).json()
    assert r["alert"] is True


def test_fraud_normal_is_allowed():
    t = {"step": 14, "type": "CASH_OUT", "amount": 3000, "oldbalanceOrg": 50000,
         "newbalanceOrig": 47000, "oldbalanceDest": 120000, "newbalanceDest": 123000}
    r = client.post("/fraud/score", json=t).json()
    assert r["alert"] is False
