"""Transaction fraud and anomaly detection on PaySim (simulated mobile-money data).

Three layers, as an ops-ready hybrid:
  1. Rules            - transparent, explainable checks (account drained, balance mismatch, ...)
  2. Isolation Forest - unsupervised anomaly score; needs no fraud labels (cold start)
  3. XGBoost          - supervised score once confirmed-fraud labels exist
  Hybrid alert        = any hard rule  OR  ML score above a threshold chosen for the ops alert budget

Graph / network features capture mule behaviour: accounts receiving money from many senders
(fan-in) and accounts that receive and immediately cash out (pass-through).

Time-based split: train on the first ~2/3 of hours, test on the rest (no future leakage).

Honesty note: PaySim's balance fields make fraud unusually easy to separate; near-perfect
scores here are a dataset artifact. The transferable part is the method: rules + ML,
leakage-safe features, and tuning thresholds to an alert budget.

Usage:  python -m src.fraud.train_fraud
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.common.config import DATA_DIR, ARTIFACTS, REPORTS, FIGURES, POWERBI, RANDOM_STATE
from src.common.plotting import style, COLORS

# Review capacity of the fraud ops team: staffed for 1.2x the average daily fraud count seen in the
# training period (on PaySim ~250 frauds/day -> ~300 reviews/day). The threshold table shows
# what other capacities would buy.
CAPACITY_MULTIPLE = 1.2
# Balance-reconciliation fields are a known PaySim artifact (fraud leaves balances inconsistent in a
# way real ledgers rarely expose). A second model without them shows how much the score relies on it.
ARTIFACT_FEATURES = ["orig_balance_error", "dest_balance_error", "dest_zero_balances"]
SPLIT_STEP = 500                   # hours 1-500 train, 501+ test


def load():
    p = DATA_DIR / "paysim.csv"
    if not p.exists():                                 # Kaggle file name
        p = next(DATA_DIR.glob("PS_*log*.csv"))
    df = pd.read_csv(p, dtype={"step": "int16", "type": "category", "isFraud": "int8", "isFlaggedFraud": "int8"})
    # Fraud only occurs in TRANSFER and CASH_OUT in PaySim; scoring the rest wastes alerts
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    return df.sort_values("step", kind="stable").reset_index(drop=True)


def features(df):
    f = pd.DataFrame(index=df.index)
    f["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    f["amount_log"] = np.log1p(df["amount"])
    f["hour"] = (df["step"] - 1) % 24
    f["night"] = f["hour"].between(0, 5).astype(int)
    f["orig_balance_log"] = np.log1p(df["oldbalanceOrg"])
    f["amount_to_balance"] = (df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)).clip(0, 10).fillna(10)
    f["drains_account"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
                           & (df["amount"] >= 0.99 * df["oldbalanceOrg"])).astype(int)
    f["orig_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    f["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    f["dest_zero_balances"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)

    # --- network features, computed only from PAST transactions (leakage-safe)
    g = df.groupby("nameDest", sort=False)
    f["dest_prior_txns"] = g.cumcount()
    f["dest_prior_amount_log"] = np.log1p(g["amount"].cumsum() - df["amount"])
    # fan-in: distinct senders seen so far by this destination
    first_pair = ~df.duplicated(["nameDest", "nameOrig"])
    f["dest_prior_senders"] = first_pair.groupby(df["nameDest"]).cumsum() - first_pair.astype(int)
    # pass-through: this sender previously RECEIVED money (receive -> send on = mule pattern)
    first_recv_step = df.groupby("nameDest")["step"].min()
    f["orig_received_before"] = (df["nameOrig"].map(first_recv_step) < df["step"]).astype(int)
    return f


RULES = {
    "R1_drain_transfer": lambda f, d: (f["drains_account"] == 1) & (f["is_transfer"] == 1),
    "R2_large_transfer": lambda f, d: (d["amount"] > 200_000) & (f["is_transfer"] == 1),
    "R3_dest_balance_unchanged": lambda f, d: (f["dest_zero_balances"] == 1) & (d["amount"] > 0),
    "R4_mule_fan_in": lambda f, d: f["dest_prior_senders"] >= 5,
    "R5_night_drain": lambda f, d: (f["night"] == 1) & (f["drains_account"] == 1),
}


def apply_rules(f, d):
    r = pd.DataFrame({k: fn(f, d).astype(int) for k, fn in RULES.items()}, index=f.index)
    return r


def rule_performance(r, y):
    rows = []
    for c in r.columns:
        fl = r[c] == 1
        rows.append({"rule": c, "alerts": int(fl.sum()), "precision": y[fl].mean() if fl.any() else np.nan,
                     "recall": y[fl].sum() / y.sum()})
    return pd.DataFrame(rows)


def threshold_table(score, y, days):
    """Precision/recall/alert volume at a range of thresholds, to pick one ops can sustain."""
    per_day = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500])
    q = np.clip(1 - per_day * days / len(score), 0, 1)
    qs = np.unique(np.quantile(score, q))
    rows = []
    for t in qs:
        a = score >= t
        rows.append({"threshold": t, "alerts": int(a.sum()), "alerts_per_day": a.sum() / days,
                     "precision": y[a].mean() if a.any() else np.nan, "recall": y[a].sum() / y.sum(),
                     "false_positives_per_day": (a & (y == 0)).sum() / days})
    return pd.DataFrame(rows)


def pick_threshold(tbl, budget):
    ok = tbl[tbl["alerts_per_day"] <= budget]
    return (ok.sort_values("recall", ascending=False).iloc[0] if not ok.empty else tbl.iloc[-1])


def main():
    style()
    df = load()
    f = features(df)
    y = df["isFraud"].to_numpy()
    tr, te = (df["step"] <= SPLIT_STEP).to_numpy(), (df["step"] > SPLIT_STEP).to_numpy()
    test_days = df.loc[te, "step"].nunique() / 24
    print(f"[fraud] {len(df):,} TRANSFER/CASH_OUT txns, fraud rate {y.mean():.4%}; test covers {test_days:.1f} days")

    # 1. rules
    r = apply_rules(f, df)
    rules_tr = rule_performance(r[tr], y[tr])
    rules_te = rule_performance(r[te], y[te])
    # Hard-block rules are picked on TRAIN: only precise rules (>=50%) earn an automatic alert.
    # Noisy rules still contribute, as inputs to the ML model.
    hard_rules = rules_tr.loc[rules_tr["precision"] >= 0.5, "rule"].tolist()
    hard_rule = (r[hard_rules].max(axis=1).to_numpy() if hard_rules else np.zeros(len(r), int))
    rules_te["hard_block"] = rules_te["rule"].isin(hard_rules)
    rules_te["train_precision"] = rules_tr["precision"].to_numpy()
    print(f"[rules] hard blocks chosen on train: {hard_rules}")

    # 2. Isolation Forest (unsupervised: fit on train, ignores labels)
    iso_cols = ["amount_log", "amount_to_balance", "orig_balance_log", "night", "drains_account",
                "dest_prior_senders", "orig_received_before", "is_transfer"]
    iso = IsolationForest(n_estimators=200, max_samples=min(256 * 40, tr.sum()), contamination="auto",
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(f.loc[tr, iso_cols])
    iso_score = -iso.score_samples(f[iso_cols])            # higher = more anomalous

    # 3. XGBoost (supervised)
    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)  # ranking task, so re-weighting is fine here
    xgbm = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                             tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
    xgbm.fit(f[tr], y[tr])
    xgb_score = xgbm.predict_proba(f)[:, 1]

    beh_cols = [c for c in f.columns if c not in ARTIFACT_FEATURES]
    xgb_beh = xgb.XGBClassifier(**xgbm.get_params()).fit(f.loc[tr, beh_cols], y[tr])
    beh_score = xgb_beh.predict_proba(f[beh_cols])[:, 1]

    # comparison on the out-of-time test period
    comp = []
    for name, s in [("Isolation Forest (no labels)", iso_score), ("XGBoost (supervised)", xgb_score),
                    ("XGBoost without balance-error fields", beh_score)]:
        comp.append({"model": name, "ROC_AUC": roc_auc_score(y[te], s[te]),
                     "PR_AUC": average_precision_score(y[te], s[te])})
    comp = pd.DataFrame(comp)
    print(comp.round(4).to_string(index=False))

    # Threshold tuned to the ops alert budget on the TRAIN period (rules take their share first),
    # then evaluated on the test period.
    tr_days = df.loc[tr, "step"].nunique() / 24
    budget = int(np.ceil(CAPACITY_MULTIPLE * y[tr].sum() / tr_days))
    print(f"[ops] review capacity set to {budget} alerts/day")
    budget_for_ml = budget - hard_rule[tr].sum() / tr_days
    thr_tr = threshold_table(np.where(hard_rule[tr] == 1, -1, xgb_score[tr]), y[tr], tr_days)
    chosen = pick_threshold(thr_tr, max(budget_for_ml, 1))
    thr_tbl = threshold_table(xgb_score[te], y[te], test_days)
    iso_tbl = threshold_table(iso_score[te], y[te], test_days)
    beh_tbl = threshold_table(beh_score[te], y[te], test_days)

    hybrid = ((hard_rule == 1) | (xgb_score >= chosen["threshold"])).astype(int)
    h = hybrid[te] == 1
    hybrid_perf = {"alerts_per_day": h.sum() / test_days, "precision": y[te][h].mean(),
                   "recall": y[te][h].sum() / y[te].sum(),
                   "fraud_amount_caught_share": df.loc[te, "amount"][h & (y[te] == 1)].sum()
                   / df.loc[te, "amount"][y[te] == 1].sum(),
                   "xgb_threshold": chosen["threshold"], "alert_budget_per_day": budget}
    print(f"[hybrid] {hybrid_perf['alerts_per_day']:.1f} alerts/day, precision {hybrid_perf['precision']:.1%}, "
          f"recall {hybrid_perf['recall']:.1%}, value caught {hybrid_perf['fraud_amount_caught_share']:.1%}")

    # ---- figures
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, s), col in zip([("Isolation Forest (no labels)", iso_score), ("XGBoost (supervised)", xgb_score),
                               ("XGBoost w/o balance-error fields", beh_score)], COLORS):
        p, rcl, _ = precision_recall_curve(y[te], s[te])
        ax.plot(rcl, p, lw=2, color=col, label=f"{name}  PR-AUC {average_precision_score(y[te], s[te]):.3f}")
    ax.scatter([hybrid_perf["recall"]], [hybrid_perf["precision"]], s=70, color=COLORS[3], zorder=5,
               label="Hybrid at alert budget")
    ax.set(xlabel="Recall (share of fraud caught)", ylabel="Precision (share of alerts that are fraud)",
           title="Fraud detection on unseen later period")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "fraud_pr_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(thr_tbl["alerts_per_day"], thr_tbl["recall"] * 100, "o-", color=COLORS[0], lw=2, label="Recall")
    ax.plot(thr_tbl["alerts_per_day"], thr_tbl["precision"] * 100, "o-", color=COLORS[1], lw=2, label="Precision")
    ax.axvline(budget, color="#999", ls="--", lw=1)
    ax.text(budget, 5, f" ops budget\n {budget}/day", fontsize=8)
    ax.plot(beh_tbl["alerts_per_day"], beh_tbl["recall"] * 100, "o--", color=COLORS[2], lw=1.5,
            label="Recall w/o balance-error fields")
    ax.set_xscale("log")
    ax.set(xlabel="Alerts per day (log scale)", ylabel="%", title="Threshold tuning: alert volume vs catch rate")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "fraud_threshold_tuning.png", dpi=150); plt.close(fig)

    imp = pd.Series(xgbm.feature_importances_, index=f.columns).sort_values(ascending=False)

    # ---- outputs
    comp.to_csv(REPORTS / "fraud_model_comparison.csv", index=False)
    rules_te.to_csv(REPORTS / "fraud_rules.csv", index=False)
    thr_tbl.to_csv(REPORTS / "fraud_threshold_xgb.csv", index=False)
    iso_tbl.to_csv(REPORTS / "fraud_threshold_isoforest.csv", index=False)
    beh_tbl.to_csv(REPORTS / "fraud_threshold_xgb_no_balance_fields.csv", index=False)
    pd.Series(hybrid_perf).to_csv(REPORTS / "fraud_hybrid_performance.csv", header=["value"])
    imp.to_csv(REPORTS / "fraud_feature_importance.csv", header=["importance"])
    joblib.dump({"xgb": xgbm, "iso": iso, "iso_cols": iso_cols, "threshold": float(chosen["threshold"]),
                 "features": list(f.columns), "rules": list(RULES), "hard_rules": hard_rules}, ARTIFACTS / "fraud_models.joblib")

    alerts = df.loc[te, ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]].copy()
    alerts["day"] = (alerts["step"] - 1) // 24 + 1
    alerts["xgb_score"] = xgb_score[te]
    alerts["anomaly_score"] = iso_score[te]
    alerts["alert"] = hybrid[te]
    alerts["outcome"] = np.select([(alerts.alert == 1) & (alerts.isFraud == 1), (alerts.alert == 1),
                                   alerts.isFraud == 1], ["True positive", "False positive", "Missed fraud"], "Clear")
    flagged = alerts[alerts["outcome"] != "Clear"].copy()
    # rule labels only for the rows we export (row-wise text on ~1M test rows would be slow)
    flagged["rules_hit"] = r.loc[flagged.index].apply(lambda row: ", ".join(k for k, v in row.items() if v), axis=1)
    flagged.to_csv(POWERBI / "fraud_alerts.csv", index=False)
    daily = alerts.groupby("day").agg(transactions=("alert", "size"), alerts=("alert", "sum"),
                                      fraud_cases=("isFraud", "sum"),
                                      caught=("outcome", lambda s: (s == "True positive").sum())).reset_index()
    daily.to_csv(POWERBI / "fraud_daily.csv", index=False)
    thr_tbl.to_csv(POWERBI / "fraud_thresholds.csv", index=False)
    rules_te.to_csv(POWERBI / "fraud_rules.csv", index=False)


if __name__ == "__main__":
    main()
