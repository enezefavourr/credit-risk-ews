"""Delinquency analytics and Early Warning System (EWS).

Uses month-by-month loan balances (POS_CASH_balance) and instalment payments.

  1. Portfolio delinquency trend  - share of active loans per DPD bucket each month (PAR30/PAR90)
  2. Roll-rate matrix             - probability of moving between DPD buckets month to month
  3. Vintage curves               - cumulative ever-30+ DPD by months on book, per origination cohort
  4. Behavioural score + EWS      - predicts which currently-performing accounts will hit 60+ DPD
                                    in the next 6 months; trained on five pooled snapshots and tested
                                    OUT-OF-TIME on a later snapshot; outputs Red/Amber/Green.

Usage:  python -m src.credit.dpd_ews
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from src.common import metrics as M
from src.common.config import DATA_DIR, REPORTS, FIGURES, POWERBI, DPD_BUCKETS
from src.common.plotting import style, COLORS, RAG

BUCKET_ORDER = [b[2] for b in DPD_BUCKETS]
HORIZON = 6            # months ahead for the EWS outcome
BAD_DPD = 60           # outcome = reaching 60+ DPD


def bucket(dpd):
    dpd = np.asarray(dpd)
    out = np.full(len(dpd), "Current", dtype=object)
    for lo, hi, name in DPD_BUCKETS[1:]:
        out[(dpd > lo) & (dpd <= hi)] = name
    return out


def load_pos():
    pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv",
                      usecols=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT",
                               "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD"],
                      dtype={"SK_ID_PREV": "int32", "SK_ID_CURR": "int32", "MONTHS_BALANCE": "int16",
                             "SK_DPD": "float32", "CNT_INSTALMENT": "float32",
                             "CNT_INSTALMENT_FUTURE": "float32", "NAME_CONTRACT_STATUS": "category"})
    pos = pos[pos["NAME_CONTRACT_STATUS"] == "Active"].copy()
    pos["SK_DPD"] = pos["SK_DPD"].fillna(0)
    pos["BUCKET"] = bucket(pos["SK_DPD"].to_numpy())
    return pos.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"]).reset_index(drop=True)


# ------------------------------------------------------------------ 1. trend
def delinquency_trend(pos, months=36):
    p = pos[pos["MONTHS_BALANCE"] >= -months]
    t = (p.groupby(["MONTHS_BALANCE", "BUCKET"]).size().unstack(fill_value=0)
         .reindex(columns=BUCKET_ORDER, fill_value=0))
    share = t.div(t.sum(axis=1), axis=0)
    share["PAR30"] = share[["31-60", "61-90", "90+"]].sum(axis=1)
    share["PAR90"] = share["90+"]
    share["active_loans"] = t.sum(axis=1)
    return share.reset_index()


# ------------------------------------------------------------------ 2. roll rates
def roll_rates(pos):
    nxt = pos.groupby("SK_ID_PREV")[["MONTHS_BALANCE", "BUCKET"]].shift(-1)
    ok = (nxt["MONTHS_BALANCE"] - pos["MONTHS_BALANCE"]) == 1        # consecutive months only
    t = pd.crosstab(pos.loc[ok, "BUCKET"], nxt.loc[ok, "BUCKET"])
    t = t.reindex(index=BUCKET_ORDER, columns=BUCKET_ORDER, fill_value=0)
    return t.div(t.sum(axis=1), axis=0), t


# ------------------------------------------------------------------ 3. vintages
def vintage_curves(pos, max_mob=24):
    p = pos[["SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD"]].copy()
    first = p.groupby("SK_ID_PREV")["MONTHS_BALANCE"].transform("min")
    p["MOB"] = p["MONTHS_BALANCE"] - first + 1
    # Home Credit has no calendar dates, so cohorts are relative: years before the application
    yrs = np.minimum((-first - 1) // 12 + 1, 5)
    p["COHORT"] = np.where(yrs >= 5, "5+ yrs ago", yrs.astype(str) + " yr ago")
    p["EVER30"] = (p["SK_DPD"] > 30).astype("int8")
    p["EVER30"] = p.groupby("SK_ID_PREV")["EVER30"].cummax()
    p = p[p["MOB"] <= max_mob]
    v = p.groupby(["COHORT", "MOB"]).agg(loans=("EVER30", "size"), ever30_rate=("EVER30", "mean")).reset_index()
    return v[v["loans"] >= 30]


# ------------------------------------------------------------------ 4. EWS
def load_instalments():
    ins = pd.read_csv(DATA_DIR / "installments_payments.csv",
                      usecols=["SK_ID_PREV", "DAYS_INSTALMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
                      dtype={"SK_ID_PREV": "int32", "DAYS_INSTALMENT": "float32",
                             "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"})
    ins["MONTH"] = np.floor(ins["DAYS_INSTALMENT"] / 30).astype("int16")
    ins["UNDERPAID"] = (ins["AMT_PAYMENT"].fillna(0) < 0.95 * ins["AMT_INSTALMENT"]).astype("int8")
    return ins[["SK_ID_PREV", "MONTH", "UNDERPAID"]]


def snapshot_features(pos, ins, s):
    """Signals as known at month s, for loans active and not yet 60+ DPD at s."""
    hist = pos[(pos["MONTHS_BALANCE"] <= s) & (pos["MONTHS_BALANCE"] > s - 12)]
    now = hist[hist["MONTHS_BALANCE"] == s].set_index("SK_ID_PREV")
    now = now[now["SK_DPD"] < BAD_DPD]
    h = hist[hist["SK_ID_PREV"].isin(now.index)]
    last6 = h[h["MONTHS_BALANCE"] > s - 6]
    last3 = h[h["MONTHS_BALANCE"] > s - 3]
    f = pd.DataFrame(index=now.index)
    f["SK_ID_CURR"] = now["SK_ID_CURR"]
    f["dpd_now"] = now["SK_DPD"]
    f["max_dpd_6m"] = last6.groupby("SK_ID_PREV")["SK_DPD"].max()
    f["late_months_6m"] = last6.assign(l=last6["SK_DPD"] > 0).groupby("SK_ID_PREV")["l"].sum()
    f["late_months_3m"] = last3.assign(l=last3["SK_DPD"] > 0).groupby("SK_ID_PREV")["l"].sum()
    dpd_3ago = h[h["MONTHS_BALANCE"] == s - 3].set_index("SK_ID_PREV")["SK_DPD"]
    f["dpd_trend_3m"] = f["dpd_now"] - dpd_3ago.reindex(f.index).fillna(0)
    f["months_on_book"] = h.groupby("SK_ID_PREV")["MONTHS_BALANCE"].size()
    cnt = now["CNT_INSTALMENT"].replace(0, np.nan)
    f["share_term_remaining"] = (now["CNT_INSTALMENT_FUTURE"] / cnt).clip(0, 1).fillna(0.5)
    iw = ins[(ins["MONTH"] <= s) & (ins["MONTH"] > s - 3) & ins["SK_ID_PREV"].isin(f.index)]
    f["underpaid_3m"] = iw.groupby("SK_ID_PREV")["UNDERPAID"].sum()
    f = f.fillna({"max_dpd_6m": 0, "late_months_6m": 0, "late_months_3m": 0, "underpaid_3m": 0})

    # Transparent rule-based signals (what an ops team can read at a glance)
    f["sig_currently_late"] = (f["dpd_now"] > 0).astype(int)
    f["sig_worsening"] = (f["dpd_trend_3m"] > 0).astype(int)
    f["sig_repeat_late"] = (f["late_months_6m"] >= 2).astype(int)
    f["sig_underpaying"] = (f["underpaid_3m"] >= 1).astype(int)
    f["sig_30plus_recent"] = (f["max_dpd_6m"] > 30).astype(int)
    return f


def snapshot_outcome(pos, ids, s):
    fut = pos[(pos["MONTHS_BALANCE"] > s) & (pos["MONTHS_BALANCE"] <= s + HORIZON) & pos["SK_ID_PREV"].isin(ids)]
    bad = fut.groupby("SK_ID_PREV")["SK_DPD"].max() >= BAD_DPD
    seen = fut.groupby("SK_ID_PREV").size() > 0
    return bad.reindex(ids).astype(float).where(seen.reindex(ids, fill_value=False))


EWS_FEATURES = ["dpd_now", "max_dpd_6m", "late_months_6m", "late_months_3m", "dpd_trend_3m",
                "months_on_book", "share_term_remaining", "underpaid_3m"]
SIGNALS = ["sig_currently_late", "sig_worsening", "sig_repeat_late", "sig_underpaying", "sig_30plus_recent"]


def rag(prob, red_cut, amber_cut):
    return np.where(prob >= red_cut, "Red", np.where(prob >= amber_cut, "Amber", "Green"))


TRAIN_SNAPSHOTS = [-25, -22, -19, -16, -13]   # pooled: 60+ DPD roll is rare, one snapshot is too few cases
TEST_SNAPSHOT = -7                             # outcome window -6..-1, after every training window ends


def build_ews(pos, ins):
    parts = []
    for s in TRAIN_SNAPSHOTS:
        f = snapshot_features(pos, ins, s)
        f["y"] = snapshot_outcome(pos, f.index, s)
        parts.append(f.dropna(subset=["y"]))
    tr = pd.concat(parts)
    te = snapshot_features(pos, ins, TEST_SNAPSHOT)
    te["y"] = snapshot_outcome(pos, te.index, TEST_SNAPSHOT)
    te = te.dropna(subset=["y"])
    s_train, s_test = f"{TRAIN_SNAPSHOTS[0]}..{TRAIN_SNAPSHOTS[-1]} (pooled)", TEST_SNAPSHOT
    print(f"[ews] train {len(tr):,} account-snapshots ({int(tr.y.sum()):,} went 60+), "
          f"test {len(te):,} ({int(te.y.sum()):,} went 60+)")

    # Gradient boosting: DPD signals are heavy-tailed and interact (e.g. late AND worsening),
    # which a linear model handles poorly. Monotonic constraints keep it explainable:
    # more DPD / more late months / more underpayment can never LOWER the risk.
    mono = {"dpd_now": 1, "max_dpd_6m": 1, "late_months_6m": 1, "late_months_3m": 1,
            "dpd_trend_3m": 1, "underpaid_3m": 1, "months_on_book": 0, "share_term_remaining": 0}
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0,
                                           monotonic_cst=[mono[c] for c in EWS_FEATURES],
                                           early_stopping=True, validation_fraction=0.15,
                                           random_state=42)
    model.fit(tr[EWS_FEATURES], tr["y"])
    p_tr = model.predict_proba(tr[EWS_FEATURES])[:, 1]
    te["p"] = model.predict_proba(te[EWS_FEATURES])[:, 1]

    # RAG cut-offs set on the TRAIN snapshot: Red = riskiest 5%, Amber = next 15%
    red_cut, amber_cut = np.quantile(p_tr, 0.95), np.quantile(p_tr, 0.80)
    te["RAG"] = rag(te["p"], red_cut, amber_cut)

    perf = {"train_snapshot": s_train, "test_snapshot": s_test, "horizon_months": HORIZON,
            "n_test": len(te), "test_bad_rate": te["y"].mean(),
            "Gini_out_of_time": M.gini(te["y"], te["p"]), "KS_out_of_time": M.ks(te["y"], te["p"])}
    rag_tbl = te.groupby("RAG").agg(accounts=("y", "size"), went_60plus=("y", "sum"), hit_rate=("y", "mean"))
    rag_tbl["share_of_accounts"] = rag_tbl["accounts"] / rag_tbl["accounts"].sum()
    rag_tbl["share_of_bads_caught"] = rag_tbl["went_60plus"] / rag_tbl["went_60plus"].sum()
    rag_tbl = rag_tbl.reindex(["Red", "Amber", "Green"]).reset_index()

    sig_tbl = pd.DataFrame([{
        "signal": s, "accounts_flagged": int(te[s].sum()),
        "hit_rate_flagged": te.loc[te[s] == 1, "y"].mean(),
        "hit_rate_not_flagged": te.loc[te[s] == 0, "y"].mean(),
        "share_of_bads_caught": te.loc[te[s] == 1, "y"].sum() / te["y"].sum()} for s in SIGNALS])
    sig_tbl["lift"] = sig_tbl["hit_rate_flagged"] / perf["test_bad_rate"]

    sample = te.sample(min(len(te), 60000), random_state=42)
    pi = permutation_importance(model, sample[EWS_FEATURES], sample["y"], scoring="roc_auc",
                                n_repeats=3, random_state=42)
    coef = pd.Series(pi.importances_mean, index=EWS_FEATURES).sort_values(ascending=False)
    return model, (red_cut, amber_cut), perf, rag_tbl, sig_tbl, coef


def current_watchlist(pos, ins, model, cuts):
    s = int(pos["MONTHS_BALANCE"].max())
    cur = snapshot_features(pos, ins, s)
    cur["behaviour_pd_6m"] = model.predict_proba(cur[EWS_FEATURES])[:, 1]
    cur["RAG"] = rag(cur["behaviour_pd_6m"], *cuts)
    cur["signals_triggered"] = cur[SIGNALS].apply(
        lambda r: ", ".join(k.replace("sig_", "").replace("_", " ") for k, v in r.items() if v), axis=1)
    cur["recommended_action"] = cur["RAG"].map({
        "Red": "Call within 48h; freeze limit increase; offer restructure",
        "Amber": "Payment reminder + SMS nudge; review at next cycle",
        "Green": "No action"})
    cur["snapshot_month"] = s
    return cur.reset_index().sort_values("behaviour_pd_6m", ascending=False)


# ------------------------------------------------------------------ plots
def plot_trend(tr):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(tr["MONTHS_BALANCE"], tr["PAR30"] * 100, color=COLORS[3], lw=2, label="PAR30 (share of loans 31+ DPD)")
    ax.plot(tr["MONTHS_BALANCE"], tr["PAR90"] * 100, color=COLORS[0], lw=2, label="PAR90 (share 90+ DPD)")
    ax.set(xlabel="Month relative to application (0 = application)", ylabel="% of active loans",
           title="Portfolio-at-risk over time")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "par_trend.png", dpi=150); plt.close(fig)


def plot_roll(rr):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(rr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(rr.columns)), rr.columns)
    ax.set_yticks(range(len(rr.index)), rr.index)
    for i in range(rr.shape[0]):
        for j in range(rr.shape[1]):
            v = rr.values[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else "#222")
    ax.set(xlabel="Bucket next month", ylabel="Bucket this month", title="Monthly roll-rate matrix")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIGURES / "roll_rate_matrix.png", dpi=150); plt.close(fig)


def plot_vintage(v):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (c, g), col in zip(v.groupby("COHORT"), COLORS):
        ax.plot(g["MOB"], g["ever30_rate"] * 100, lw=2, color=col, label=c)
    ax.set(xlabel="Months on book", ylabel="Cumulative % ever 30+ DPD",
           title="Vintage curves by origination cohort")
    ax.legend(title="Originated", frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "vintage_curves.png", dpi=150); plt.close(fig)


def plot_rag(rag_tbl):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rag_tbl["RAG"], rag_tbl["hit_rate"] * 100, color=[RAG[r] for r in rag_tbl["RAG"]])
    for i, r in rag_tbl.iterrows():
        ax.text(i, r["hit_rate"] * 100, f"{r['hit_rate']:.1%}\n{r['share_of_bads_caught']:.0%} of bads",
                ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="% reaching 60+ DPD within 6 months",
           title="Early warning: out-of-time hit rate by RAG status")
    ax.margins(y=0.25)
    fig.tight_layout(); fig.savefig(FIGURES / "ews_rag_hit_rate.png", dpi=150); plt.close(fig)


def main():
    style()
    pos = load_pos()
    ins = load_instalments()
    print(f"[dpd] {len(pos):,} active loan-months, {pos.SK_ID_PREV.nunique():,} loans")

    tr = delinquency_trend(pos)
    rr, rr_counts = roll_rates(pos)
    v = vintage_curves(pos)
    print("[dpd] roll-rate matrix:\n" + (rr * 100).round(1).to_string())

    model, cuts, perf, rag_tbl, sig_tbl, coef = build_ews(pos, ins)
    print(f"[ews] out-of-time Gini {perf['Gini_out_of_time']:.3f}, KS {perf['KS_out_of_time']:.3f}")
    print(rag_tbl.round(3).to_string(index=False))
    watch = current_watchlist(pos, ins, model, cuts)
    print(f"[ews] current watchlist: {(watch.RAG == 'Red').sum():,} Red, {(watch.RAG == 'Amber').sum():,} Amber")

    plot_trend(tr); plot_roll(rr); plot_vintage(v); plot_rag(rag_tbl)

    tr.to_csv(REPORTS / "delinquency_trend.csv", index=False)
    rr.to_csv(REPORTS / "roll_rates.csv")
    v.to_csv(REPORTS / "vintage_curves.csv", index=False)
    rag_tbl.to_csv(REPORTS / "ews_rag_backtest.csv", index=False)
    sig_tbl.to_csv(REPORTS / "ews_signal_backtest.csv", index=False)
    pd.Series(perf).to_csv(REPORTS / "ews_performance.csv", header=["value"])
    coef.to_csv(REPORTS / "ews_feature_importance.csv", header=["auc_drop_when_shuffled"])

    # Power BI (long formats are easier to visualise)
    tr.to_csv(POWERBI / "delinquency_trend.csv", index=False)
    rr.reset_index(names="from_bucket").melt(id_vars="from_bucket", var_name="to_bucket", value_name="roll_rate") \
      .to_csv(POWERBI / "roll_rates.csv", index=False)
    v.to_csv(POWERBI / "vintage_curves.csv", index=False)
    rag_tbl.to_csv(POWERBI / "ews_backtest.csv", index=False)
    watch.to_csv(POWERBI / "ews_watchlist.csv", index=False)


if __name__ == "__main__":
    main()
