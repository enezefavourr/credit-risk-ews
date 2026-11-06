"""Production monitoring for the application PD model.

Simulation (clearly a simulation): the 20% held-out 'production stream' is replayed as
12 monthly batches. Months 1-6 are business as usual. From month 7 a gradual applicant mix
shift is injected (more applicants with weak external scores and younger profiles), and in
month 10 a data-quality incident (EXT_SOURCE_3 feed drops out for most applicants).

Outcome lag: default flags only mature after ~3 months, so discrimination (Gini/KS) and
calibration are reported for months 1-9 only; stability (PSI/CSI, missing rates) is
available immediately for every month. This is how real monitoring works.

Thresholds / retraining triggers (also stored in the model card)
  Score PSI             < 0.10 Green | 0.10-0.25 Amber | > 0.25 Red
  Feature CSI           same bands
  Gini drop vs validation > 0.05 absolute       -> Red
  Calibration ratio (actual / predicted DR) outside 0.80-1.25 -> Amber
  Missing-rate jump     > 10 percentage points vs training -> Red (data pipeline issue)

Usage:  python -m src.monitoring.monitor
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import metrics as M
from src.common.config import ARTIFACTS, REPORTS, FIGURES, POWERBI, RANDOM_STATE
from src.common.plotting import style, COLORS, RAG

N_MONTHS = 12
OUTCOME_LAG = 3
rng = np.random.default_rng(RANDOM_STATE)


def make_batches(prod: pd.DataFrame):
    size = max(len(prod) // 6, 200)
    risk = prod["EXT_SOURCE_MEAN"].rank(pct=True).fillna(0.5).to_numpy()   # 0 = weakest
    young = (prod["AGE_YEARS"] < 30).to_numpy()
    batches = []
    for m in range(1, N_MONTHS + 1):
        shift = 0.0 if m <= 6 else min((m - 6) * 0.45, 2.5)       # gradual, then stronger
        w = np.exp(-shift * risk) * np.where(young, 1 + shift * 0.4, 1.0)
        idx = rng.choice(len(prod), size=size, replace=True, p=w / w.sum())
        b = prod.iloc[idx].copy()
        if m == 10:                                                 # feed outage
            drop = rng.random(len(b)) < 0.7
            b.loc[drop, "EXT_SOURCE_3"] = np.nan
            b["EXT_SOURCE_MEAN"] = b[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
            b["EXT_SOURCE_MISSING_CNT"] = b[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1)
        b["month"] = m
        batches.append(b)
    return batches


def status_worst(*s):
    order = {"Green": 0, "Amber": 1, "Red": 2}
    return max(s, key=lambda x: order[x])


def main():
    style()
    bundle = joblib.load(ARTIFACTS / "credit_models.joblib")
    splits = joblib.load(ARTIFACTS / "credit_splits.joblib")
    sc, prep, xgbm, meta = bundle["scorecard"], bundle["prep"], bundle["xgb"], bundle["meta"]
    train, prod, feats = splits["train"], splits["prod"], splits["features"]
    cutoff = meta["recommended_cutoff"]
    val_gini = meta["validation"]["Scorecard (LR)"]["Gini"]
    trig = meta["retraining_triggers"]

    ref_score = sc.score(train[feats])
    score_edges = M.psi_edges(ref_score)
    ref_xgb = xgbm.predict_proba(prep.transform(train[feats]))[:, 1]
    xgb_edges = M.psi_edges(ref_xgb)
    mon_feats = sc.features
    ref_missing = train[feats].isna().mean()          # data-quality check on ALL model inputs
    hist_y, hist_p = {}, {}

    rows, csi_rows, alerts = [], [], []
    for b in make_batches(prod):
        m = int(b["month"].iloc[0])
        X = b[feats]
        score = sc.score(X)
        pd_ = sc.predict_proba(X)
        y = b["TARGET"].to_numpy()
        matured = m <= N_MONTHS - OUTCOME_LAG
        r = {"month": m, "applications": len(b), "approval_rate": (score >= cutoff).mean(),
             "mean_score": score.mean(), "mean_pd": pd_.mean(),
             "score_psi": M.psi(ref_score, score, edges=score_edges),
             "xgb_pd_psi": M.psi(ref_xgb, xgbm.predict_proba(prep.transform(X))[:, 1], edges=xgb_edges),
             "outcome_matured": matured}
        hist_y[m], hist_p[m] = y, pd_
        if matured:
            # Discrimination on a rolling 3-month window: one small month is too noisy to act on
            win = [k for k in range(m - 2, m + 1) if k in hist_y]
            yw, pw = np.concatenate([hist_y[k] for k in win]), np.concatenate([hist_p[k] for k in win])
            r.update({"actual_default_rate": y.mean(), "gini": M.gini(yw, pw), "ks": M.ks(yw, pw),
                      "gini_single_month": M.gini(y, pd_), "calibration_ratio": y.mean() / pd_.mean()})
            # approved book only: what credit cares about
            ap = score >= cutoff
            r["default_rate_approved"] = y[ap].mean() if ap.any() else np.nan

        # feature-level stability
        worst_csi, worst_feat, worst_miss = 0.0, None, 0.0
        for f in mon_feats:
            if pd.api.types.is_numeric_dtype(train[f]):
                v = M.psi(train[f], b[f])
            else:
                v = M.categorical_psi(train[f], b[f])
            miss_jump = b[f].isna().mean() - ref_missing[f]
            csi_rows.append({"month": m, "feature": f, "csi": v, "csi_status": M.psi_status(v),
                             "missing_rate": b[f].isna().mean(), "missing_jump_pp": miss_jump * 100})
            if v > worst_csi:
                worst_csi, worst_feat = v, f
        miss_jumps = b[feats].isna().mean() - ref_missing
        worst_miss, worst_miss_feat = float(miss_jumps.max()), miss_jumps.idxmax()
        r["max_csi"], r["max_csi_feature"] = worst_csi, worst_feat
        r["max_missing_jump_pp"] = worst_miss * 100

        # ---- alert logic
        s_psi = M.psi_status(r["score_psi"])
        s_csi = M.psi_status(worst_csi)
        s_dq = "Red" if worst_miss > 0.10 else "Green"
        s_gini = s_cal = "Green"
        if matured:
            s_gini = "Red" if val_gini - r["gini"] > trig["gini_drop_abs"] else "Green"
            lo, hi = trig["calibration_ratio_outside"]
            s_cal = "Green" if lo <= r["calibration_ratio"] <= hi else "Amber"
        r.update({"status_stability": s_psi, "status_features": s_csi, "status_data_quality": s_dq,
                  "status_discrimination": s_gini if matured else "Pending",
                  "status_calibration": s_cal if matured else "Pending"})
        r["overall_status"] = status_worst(s_psi, s_csi, s_dq, s_gini, s_cal)
        r["retrain_recommended"] = bool(r["score_psi"] > trig["score_psi"] or s_gini == "Red")
        for name, st, detail in [
            ("Score PSI", s_psi, f"PSI {r['score_psi']:.3f}"),
            ("Feature drift", s_csi, f"{worst_feat} CSI {worst_csi:.3f}"),
            ("Data quality", s_dq, f"{worst_miss_feat} missing rate up {worst_miss * 100:.0f}pp"),
            ("Discrimination", s_gini, f"rolling 3-month Gini {r.get('gini', np.nan):.3f} vs {val_gini:.3f} at validation"),
            ("Calibration", s_cal, f"actual/predicted {r.get('calibration_ratio', np.nan):.2f}")]:
            if st in ("Amber", "Red"):
                alerts.append({"month": m, "check": name, "status": st, "detail": detail})
        rows.append(r)

    mon = pd.DataFrame(rows)
    csi = pd.DataFrame(csi_rows)
    al = pd.DataFrame(alerts, columns=["month", "check", "status", "detail"])
    show = ["month", "approval_rate", "mean_pd", "actual_default_rate", "score_psi", "gini", "ks",
            "calibration_ratio", "max_csi_feature", "max_csi", "overall_status", "retrain_recommended"]
    print(mon[show].round(3).to_string(index=False))

    # ---- figures
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.bar(mon["month"], mon["score_psi"], color=[RAG[M.psi_status(v)] for v in mon["score_psi"]])
    ax.axhline(0.1, color="#999", ls="--", lw=1); ax.axhline(0.25, color="#999", ls="--", lw=1)
    ax.set(title="Score stability (PSI)", xlabel="Month", ylabel="PSI")
    ax = axes[0, 1]
    mm = mon[mon["outcome_matured"]]
    ax.plot(mm["month"], mm["gini"], "o-", color=COLORS[0], lw=2, label="Gini")
    ax.plot(mm["month"], mm["ks"], "o-", color=COLORS[1], lw=2, label="KS")
    ax.axhline(val_gini, color=COLORS[0], ls=":", lw=1, label="Validation Gini")
    ax.axhline(val_gini - trig["gini_drop_abs"], color=COLORS[3], ls="--", lw=1, label="Gini retrain trigger")
    ax.set(title="Discrimination, rolling 3 months (matured)", xlabel="Month", xlim=(0.5, N_MONTHS + 0.5))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    ax.plot(mm["month"], mm["mean_pd"] * 100, "o-", color=COLORS[0], lw=2, label="Predicted PD")
    ax.plot(mm["month"], mm["actual_default_rate"] * 100, "o-", color=COLORS[3], lw=2, label="Actual default rate")
    ax.set(title="Calibration over time", xlabel="Month", ylabel="%", xlim=(0.5, N_MONTHS + 0.5))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    piv = csi.pivot(index="feature", columns="month", values="csi")
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index[:8]]
    im = ax.imshow(piv.values, cmap="Oranges", vmin=0, vmax=0.3, aspect="auto")
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8)
    ax.set_xticks(range(N_MONTHS), piv.columns)
    ax.set(title="Feature drift (CSI) - top 8 features", xlabel="Month")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Model monitoring: simulated 12-month production replay", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "monitoring_dashboard.png", dpi=150); plt.close(fig)

    # ---- outputs
    mon.to_csv(REPORTS / "monitoring_monthly.csv", index=False)
    csi.to_csv(REPORTS / "monitoring_csi.csv", index=False)
    al.to_csv(REPORTS / "monitoring_alerts.csv", index=False)
    mon.to_csv(POWERBI / "monitoring_monthly.csv", index=False)
    csi.to_csv(POWERBI / "monitoring_csi.csv", index=False)
    al.to_csv(POWERBI / "monitoring_alerts.csv", index=False)
    write_health_report(mon, al, meta)


def write_health_report(mon, al, meta):
    """Plain-language monthly note for the Head of Data and risk stakeholders."""
    last = mon.iloc[-1]
    lm = mon[mon["outcome_matured"]].iloc[-1]
    first_red = mon.loc[mon["overall_status"] == "Red", "month"]
    lines = [
        f"# Model health report: {meta['model_version']}",
        "",
        "_Simulated production replay on a public dataset. Portfolio prototype, not a live lending system._",
        "",
        f"**Overall status, latest month ({int(last.month)}): {last.overall_status}.** "
        + ("Retraining is recommended." if last.retrain_recommended else "No retraining needed yet.")
        + (f" First Red month: {int(first_red.iloc[0])}." if not first_red.empty else ""),
        "",
        "## What changed",
        f"- Applicant mix: the score distribution has shifted (PSI {last.score_psi:.2f}; above 0.25 means a material shift). "
        f"The approval rate at the current cut-off moved from {mon.approval_rate.iloc[0]:.0%} in month 1 to {last.approval_rate:.0%}.",
        f"- The input that moved most is **{last.max_csi_feature}** (CSI {last.max_csi:.2f}).",
        f"- Latest month with known outcomes (month {int(lm.month)}): Gini {lm.gini:.2f} vs {meta['validation']['Scorecard (LR)']['Gini']:.2f} at validation; "
        f"predicted default rate {lm.mean_pd:.1%} vs actual {lm.actual_default_rate:.1%}.",
        "",
        "## Alerts raised",
    ]
    if al.empty:
        lines.append("- None.")
    else:
        for _, a in al.iterrows():
            lines.append(f"- Month {a.month}: **{a.check}** {a.status}. {a.detail}")
    lines += ["", "## Recommended actions",
              "- Confirm with data engineering whether any input feed failed (data-quality alerts).",
              "- If the applicant mix shift is a deliberate channel or product change, recalibrate on recent "
              "matured outcomes; if not, investigate the acquisition source.",
              "- Keep the cut-off under review until a recalibrated model is validated." ]
    (REPORTS / "model_health_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[monitor] health report -> {REPORTS / 'model_health_report.md'}")


if __name__ == "__main__":
    main()
