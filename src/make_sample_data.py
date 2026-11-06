"""Generate small synthetic files with the SAME column names and conventions as
the Kaggle Home Credit Default Risk and PaySim datasets.

Purpose: lets anyone clone the repo and run the full pipeline in a few minutes
without downloading ~3 GB of data. The numbers it produces are NOT meaningful;
all reported results in the README come from the real datasets.

Usage:  python -m src.make_sample_data
"""
import numpy as np
import pandas as pd

from src.common.config import ROOT, RANDOM_STATE

OUT = ROOT / "data" / "sample"
rng = np.random.default_rng(RANDOM_STATE)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------------
# Home Credit: application_train, bureau, installments_payments, POS_CASH_balance
# --------------------------------------------------------------------------
def make_home_credit(n=20000):
    ids = np.arange(100002, 100002 + n)
    z = rng.normal(size=n)                      # latent credit risk

    def ext(missing):
        v = sigmoid(-1.1 * z + rng.normal(0, 0.9, n))
        v[rng.random(n) < missing] = np.nan
        return v

    age_years = np.clip(rng.normal(43, 12, n) + 2.5 * -z, 21, 69)
    income_type = rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"],
                             n, p=[0.52, 0.23, 0.18, 0.07])
    days_employed = -np.abs(rng.gamma(2, 900, n)) * (1 + 0.15 * -z).clip(0.3)
    days_employed = np.where(income_type == "Pensioner", 365243, days_employed)
    income = np.round(np.exp(rng.normal(11.9, 0.5, n) - 0.08 * z), -2)
    credit = np.round(income * rng.uniform(1, 6, n), -2)
    term_months = rng.choice([12, 18, 24, 36, 48, 60], n)
    annuity = np.round(credit / term_months * 1.25, 1)
    region = rng.choice([1, 2, 3], n, p=[0.1, 0.74, 0.16])
    occupation = rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
                             "High skill tech staff", "Accountants", None], n,
                            p=[0.18, 0.1, 0.09, 0.07, 0.06, 0.04, 0.03, 0.43])
    has_bureau = rng.random(n) < 0.86

    logit = (-3.05 + 0.95 * z + 0.25 * (region - 2) + 0.35 * (age_years < 30)
             + 0.25 * (income_type == "Working") + 0.2 * ~has_bureau
             + 0.15 * (credit / income > 4))
    target = (rng.random(n) < sigmoid(logit)).astype(int)

    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.66, 0.34]),
        "FLAG_OWN_CAR": rng.choice(["N", "Y"], n, p=[0.66, 0.34]),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.69, 0.31]),
        "CNT_CHILDREN": rng.poisson(0.4, n),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": np.round(credit * rng.uniform(0.8, 1.0, n), -2),
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education",
                                           "Incomplete higher", "Lower secondary"], n,
                                          p=[0.71, 0.24, 0.035, 0.015]),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage",
                                          "Separated", "Widow"], n,
                                         p=[0.64, 0.15, 0.1, 0.06, 0.05]),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Municipal apartment",
                                         "Rented apartment"], n, p=[0.89, 0.05, 0.04, 0.02]),
        "DAYS_BIRTH": -(age_years * 365.25).astype(int),
        "DAYS_EMPLOYED": days_employed.astype(int),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
        "DAYS_LAST_PHONE_CHANGE": -rng.gamma(1.5, 650, n).astype(int) * (1 - 0.1 * z).clip(0.2),
        "OCCUPATION_TYPE": occupation,
        "REGION_RATING_CLIENT": region,
        "EXT_SOURCE_1": ext(0.56),
        "EXT_SOURCE_2": ext(0.002),
        "EXT_SOURCE_3": ext(0.20),
    })
    app["DAYS_LAST_PHONE_CHANGE"] = app["DAYS_LAST_PHONE_CHANGE"].round()

    # ---- bureau: other lenders' credit records (missing = thin file) ----
    b_ids = np.repeat(ids[has_bureau], rng.poisson(4, has_bureau.sum()) + 1)
    zb = pd.Series(z, index=ids).loc[b_ids].to_numpy()
    nb = len(b_ids)
    overdue = rng.random(nb) < sigmoid(-3.2 + 0.9 * zb)
    amt = np.round(np.exp(rng.normal(12, 1, nb)), -2)
    bureau = pd.DataFrame({
        "SK_ID_CURR": b_ids,
        "SK_ID_BUREAU": np.arange(5000000, 5000000 + nb),
        "CREDIT_ACTIVE": rng.choice(["Closed", "Active"], nb, p=[0.63, 0.37]),
        "DAYS_CREDIT": -rng.integers(1, 2900, nb),
        "CREDIT_DAY_OVERDUE": np.where(overdue, rng.integers(1, 120, nb), 0),
        "AMT_CREDIT_SUM": amt,
        "AMT_CREDIT_SUM_DEBT": np.round(amt * rng.uniform(0, 0.8, nb) * (1 + 0.1 * zb).clip(0), -2),
        "AMT_CREDIT_SUM_OVERDUE": np.where(overdue, np.round(amt * rng.uniform(0.01, 0.2, nb), -2), 0.0),
    })

    # ---- previous loans: monthly POS balance (Markov DPD chain) + instalments ----
    n_prev = rng.poisson(1.6, n) * (rng.random(n) < 0.85)
    states = np.array([0, 15, 45, 75, 120])     # representative DPD per bucket
    pos_rows, inst_rows = [], []
    sk_prev = 1000000
    for cid, zi, k in zip(ids, z, n_prev):
        for _ in range(k):
            sk_prev += 1
            T = int(rng.choice([6, 10, 12, 18, 24]))
            end = -int(rng.integers(1, 72))     # months before application the loan ended/snapshot
            start = end - T + 1
            p_bad = sigmoid(-3.3 + 1.1 * zi)
            s = 0
            for m_idx, mb in enumerate(range(start, end + 1)):
                if s == 0:
                    s = 1 if rng.random() < p_bad else 0
                else:
                    r = rng.random()
                    cure = 0.55 - 0.15 * zi
                    if r < cure:
                        s = 0
                    elif r < cure + 0.25:
                        s = s                       # stay
                    else:
                        s = min(s + 1, 4)
                dpd = 0 if s == 0 else int(states[s] + rng.integers(-10, 10))
                pos_rows.append((sk_prev, cid, mb, T, T - m_idx - 1,
                                 "Active" if mb < end else "Completed", max(dpd, 0), max(dpd, 0)))
                day_inst = mb * 30 + int(rng.integers(-5, 5))
                inst_amt = round(float(rng.uniform(3000, 25000)), 2)
                pay_late = 0 if s == 0 else max(dpd, 1)
                early = int(rng.integers(0, 8)) if s == 0 else 0
                paid_frac = 1.0 if s == 0 else float(rng.choice([1.0, 0.6, 0.3], p=[0.6, 0.25, 0.15]))
                inst_rows.append((sk_prev, cid, 1, m_idx + 1, float(day_inst),
                                  float(day_inst + pay_late - early), inst_amt,
                                  round(inst_amt * paid_frac, 2)))

    pos = pd.DataFrame(pos_rows, columns=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE",
                                          "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
                                          "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"])
    inst = pd.DataFrame(inst_rows, columns=["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION",
                                            "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
                                            "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"])
    return app, bureau, inst, pos


# --------------------------------------------------------------------------
# PaySim-like mobile money transactions
# --------------------------------------------------------------------------
def make_paysim(n=200000, fraud_rate=0.004):
    steps = rng.integers(1, 744, n)
    types = rng.choice(["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"], n,
                       p=[0.35, 0.34, 0.22, 0.08, 0.01])
    old_org = np.round(np.exp(rng.normal(10, 1.6, n)), 2)
    old_org[rng.random(n) < 0.3] = 0.0
    frac = rng.beta(1.2, 4, n)
    amount = np.round(np.where(old_org > 0, old_org * frac, np.exp(rng.normal(9, 1.3, n))), 2)
    amount = np.where(types == "PAYMENT", np.round(np.exp(rng.normal(8.5, 1, n)), 2), amount)
    # occasional legitimate full-balance transfers (keeps rules imperfect)
    full = (types == "TRANSFER") & (rng.random(n) < 0.06) & (old_org > 0)
    amount = np.where(full, old_org, amount)
    new_org = np.where(types == "CASH_IN", old_org + amount, np.clip(old_org - amount, 0, None))
    names_orig = np.array([f"C{v}" for v in rng.integers(10**8, 10**10, n)])
    dest_pool = np.array([f"C{v}" for v in rng.integers(10**8, 10**10, 40000)])
    names_dest = np.where(types == "PAYMENT",
                          np.array([f"M{v}" for v in rng.integers(10**8, 10**10, n)]),
                          rng.choice(dest_pool, n))
    old_dest = np.where(types == "PAYMENT", 0.0, np.round(np.exp(rng.normal(11, 1.5, n)), 2))
    new_dest = np.where(types == "PAYMENT", 0.0, old_dest + amount)
    noise = rng.random(n) < 0.25                       # PaySim has many inconsistent dest balances
    new_dest = np.where(noise & (types != "PAYMENT"), old_dest, new_dest)

    df = pd.DataFrame({"step": steps, "type": types, "amount": amount, "nameOrig": names_orig,
                       "oldbalanceOrg": old_org, "newbalanceOrig": new_org, "nameDest": names_dest,
                       "oldbalanceDest": old_dest, "newbalanceDest": new_dest})
    df["isFraud"] = 0

    # Fraud pattern: account takeover -> TRANSFER drains account -> CASH_OUT via mule
    n_f = int(n * fraud_rate / 2)
    mules = np.array([f"C{v}" for v in rng.integers(10**8, 10**10, max(n_f // 6, 5))])
    rows = []
    for _ in range(n_f):
        st = int(rng.choice(np.arange(1, 744), p=None))
        if rng.random() < 0.6:                          # skew towards night hours
            st = (st // 24) * 24 + int(rng.integers(0, 6)) + 1
        bal = round(float(np.exp(rng.normal(12, 1.2))), 2)
        amt = bal if rng.random() < 0.85 else round(bal * rng.uniform(0.5, 0.95), 2)
        victim = f"C{rng.integers(10**8, 10**10)}"
        mule = rng.choice(mules)
        zero_dest = rng.random() < 0.6
        rows.append((st, "TRANSFER", amt, victim, bal, round(bal - amt, 2), mule,
                     0.0 if zero_dest else 5000.0, 0.0 if zero_dest else 5000.0 + amt, 1))
        cbal = round(float(amt + rng.uniform(0, 3000)), 2)
        rows.append((min(st + int(rng.integers(0, 2)), 743), "CASH_OUT", amt, mule, cbal,
                     round(cbal - amt, 2), rng.choice(dest_pool), 0.0, 0.0 if zero_dest else amt, 1))
    fraud = pd.DataFrame(rows, columns=df.columns)
    df = pd.concat([df, fraud], ignore_index=True).sort_values("step", kind="stable")
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > 200000) & (df["isFraud"] == 1)
                            & (rng.random(len(df)) < 0.01)).astype(int)
    return df.reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app, bureau, inst, pos = make_home_credit()
    app.to_csv(OUT / "application_train.csv", index=False)
    bureau.to_csv(OUT / "bureau.csv", index=False)
    inst.to_csv(OUT / "installments_payments.csv", index=False)
    pos.to_csv(OUT / "POS_CASH_balance.csv", index=False)
    ps = make_paysim()
    ps.to_csv(OUT / "paysim.csv", index=False)
    print(f"application_train {app.shape}, default rate {app.TARGET.mean():.3f}")
    print(f"bureau {bureau.shape}, installments {inst.shape}, POS_CASH {pos.shape}")
    print(f"paysim {ps.shape}, fraud rate {ps.isFraud.mean():.4f}")


if __name__ == "__main__":
    main()
