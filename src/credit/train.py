"""Train and validate the application PD models.

Champion  : WoE logistic-regression scorecard (transparent, points-based, reason codes)
Challengers: Random Forest and XGBoost (SHAP explanations)

Outputs
  artifacts/   versioned models + metadata used by the API and monitoring
  reports/     metrics, IV, points table, cut-off strategy, figures
  powerbi/data scored applications for the dashboard

Usage:  python -m src.credit.train
"""
import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from src.common import metrics as M
from src.common.config import (ARTIFACTS, REPORTS, FIGURES, POWERBI, RANDOM_STATE, LGD, DATA_DIR)
from src.common.plotting import style, COLORS
from src.credit.features import build_dataset, model_feature_list
from src.credit.preprocess import GRADES, assign_grade, TreePreprocessor
from src.credit.scorecard import WoEScorecard

MODEL_VERSION = "pd_app_v1.0"

def split(df):
    """60% train / 20% validation / 20% held back as the simulated 'production' stream.

    Home Credit has no application dates, so this is a stratified random split; the
    production stream is later fed to monitoring month by month (see src/monitoring).
    """
    train, rest = train_test_split(df, test_size=0.4, stratify=df["TARGET"], random_state=RANDOM_STATE)
    valid, prod = train_test_split(rest, test_size=0.5, stratify=rest["TARGET"], random_state=RANDOM_STATE)
    return train, valid, prod


def fit_models(train, features):
    X, y = train[features], train["TARGET"]

    sc = WoEScorecard().fit(X, y)
    print(f"[scorecard] {len(sc.features)} features selected from {len(features)} candidates")

    prep = TreePreprocessor(features).fit(X)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt",
                                n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(prep.transform(X, fill_na=True), y)

    # Early stopping on an internal split of TRAIN, so validation stays untouched.
    # No class re-weighting: weighting would distort the PDs, which must stay calibrated
    # because they drive pricing and expected loss. Imbalance is handled by the cut-off.
    Xt, Xe, yt, ye = train_test_split(prep.transform(X), y, test_size=0.15, stratify=y,
                                      random_state=RANDOM_STATE)
    xgbm = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=20,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=5, eval_metric="auc",
                             early_stopping_rounds=100, tree_method="hist", random_state=RANDOM_STATE,
                             n_jobs=-1)
    xgbm.fit(Xt, yt, eval_set=[(Xe, ye)], verbose=False)
    print(f"[xgboost] best iteration {xgbm.best_iteration}")
    return sc, prep, rf, xgbm


def predict_all(df, sc, prep, rf, xgbm):
    feats = prep.features
    return {
        "Scorecard (LR)": sc.predict_proba(df[feats]),
        "Random Forest": rf.predict_proba(prep.transform(df[feats], fill_na=True))[:, 1],
        "XGBoost": xgbm.predict_proba(prep.transform(df[feats]))[:, 1],
    }


def evaluate(valid, preds):
    rows, seg_rows = [], []
    y = valid["TARGET"].to_numpy()
    for name, p in preds.items():
        s = M.summary(y, p)
        s["HL_pvalue"] = M.hosmer_lemeshow(y, p)[1]
        rows.append({"model": name, **s})
        for seg_col, seg_name in [("THIN_FILE", "Thin file (no bureau)"), ("NEW_TO_LENDER", "New to lender")]:
            for v, lab in [(1, seg_name), (0, "Rest")]:
                m = (valid[seg_col] == v).to_numpy()
                if m.sum() > 50 and y[m].sum() > 5:
                    seg_rows.append({"model": name, "segment_type": seg_col, "segment": lab,
                                     "n": int(m.sum()), "default_rate": y[m].mean(),
                                     "Gini": M.gini(y[m], p[m])})
    return pd.DataFrame(rows), pd.DataFrame(seg_rows)


def cutoff_strategy(valid, score, pd_, n_points=12):
    """Approval-rate vs risk trade-off: who we'd approve at each score cut-off."""
    df = pd.DataFrame({"score": score, "pd": pd_, "y": valid["TARGET"].to_numpy(),
                       "ead": valid["AMT_CREDIT"].to_numpy()})
    df["el"] = df["pd"] * LGD * df["ead"]
    cuts = np.unique(np.quantile(df["score"], np.linspace(0.0, 0.6, n_points)).round())
    rows = []
    for c in cuts:
        a = df[df["score"] >= c]
        r = df[df["score"] < c]
        rows.append({
            "cutoff_score": int(c), "approval_rate": len(a) / len(df),
            "bad_rate_approved": a["y"].mean(), "bad_rate_declined": r["y"].mean() if len(r) else np.nan,
            "defaults_avoided_share": r["y"].sum() / df["y"].sum(),
            "expected_loss_approved": a["el"].sum(),
            "expected_loss_rate": a["el"].sum() / a["ead"].sum(),
            "actual_loss_approved_LGD": (a["y"] * a["ead"] * LGD).sum(),
        })
    return pd.DataFrame(rows)


def recommend_cutoff(table, max_bad_rate=0.06):
    ok = table[table["bad_rate_approved"] <= max_bad_rate]
    if ok.empty:
        return table.iloc[-1]
    return ok.sort_values("approval_rate", ascending=False).iloc[0]


# ---------------------------------------------------------------- figures
def plot_roc(y, preds):
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, p), col in zip(preds.items(), COLORS):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  (Gini {M.gini(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax.set(xlabel="False positive rate (goods flagged)", ylabel="True positive rate (bads caught)",
           title="Discrimination on validation set")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "roc_comparison.png", dpi=150); plt.close(fig)


def plot_calibration(y, preds):
    fig, ax = plt.subplots(figsize=(6, 5))
    mx = 0
    for (name, p), col in zip(preds.items(), COLORS):
        t = M.calibration_table(y, p)
        ax.plot(t["predicted_PD"], t["actual_DR"], "o-", color=col, lw=2, ms=5, label=name)
        mx = max(mx, t["predicted_PD"].max(), t["actual_DR"].max())
    ax.plot([0, mx], [0, mx], color="#999", lw=1, ls="--", label="Perfect calibration")
    ax.set(xlabel="Predicted PD (decile mean)", ylabel="Observed default rate",
           title="Calibration: predicted vs observed")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=150); plt.close(fig)


def plot_score_dist(y, score):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 40)
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.6, color=COLORS[0], label="Repaid (good)")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.6, color=COLORS[3], label="Defaulted (bad)")
    ax.set(xlabel="Scorecard points (higher = safer)", ylabel="Density",
           title=f"Score separation (KS = {M.ks(y, -score):.3f})")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "score_distribution.png", dpi=150); plt.close(fig)


def plot_iv(iv_table, selected):
    t = iv_table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    cols = [COLORS[0] if f in selected else "#bbbbbb" for f in t.index]
    ax.barh(t.index, t["IV"], color=cols)
    ax.axvline(0.02, color="#999", ls="--", lw=1)
    ax.set(xlabel="Information Value", title="Top 20 features by IV (blue = in scorecard)")
    fig.tight_layout(); fig.savefig(FIGURES / "information_value.png", dpi=150); plt.close(fig)


def shap_outputs(xgbm, prep, valid, pd_xgb):
    Xv = prep.transform(valid[prep.features])
    sample = Xv.sample(min(3000, len(Xv)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(xgbm)
    sv = explainer(sample)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("XGBoost: what drives predicted default risk")
    plt.tight_layout(); plt.savefig(FIGURES / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(FIGURES / "shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

    # One high-risk applicant, explained
    i = int(np.argmax(pd_xgb))
    one = explainer(Xv.iloc[[i]])
    plt.figure()
    shap.plots.waterfall(one[0], max_display=10, show=False)
    plt.title(f"Why applicant {int(valid['SK_ID_CURR'].iloc[i])} scored high risk (log-odds)")
    plt.tight_layout(); plt.savefig(FIGURES / "shap_waterfall_high_risk.png", dpi=150, bbox_inches="tight"); plt.close()

    imp = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False)


def main():
    style()
    df = build_dataset()
    features = model_feature_list(df)
    train, valid, prod = split(df)
    print(f"[split] train {len(train):,}  valid {len(valid):,}  production stream {len(prod):,}")

    sc, prep, rf, xgbm = fit_models(train, features)
    preds = predict_all(valid, sc, prep, rf, xgbm)
    y = valid["TARGET"].to_numpy()

    perf, seg = evaluate(valid, preds)
    print(perf.round(4).to_string(index=False))

    score_v = sc.score(valid[features])
    pd_v = preds["Scorecard (LR)"]
    cut = cutoff_strategy(valid, score_v, pd_v)
    rec = recommend_cutoff(cut)
    print(f"[policy] recommended cut-off {int(rec.cutoff_score)} pts: approve {rec.approval_rate:.1%}, "
          f"bad rate approved {rec.bad_rate_approved:.2%}, avoids {rec.defaults_avoided_share:.1%} of defaults")

    # grade table
    g = pd.DataFrame({"grade": assign_grade(pd_v), "y": y, "pd": pd_v})
    grade_tbl = g.groupby("grade").agg(n=("y", "size"), predicted_PD=("pd", "mean"), actual_DR=("y", "mean"))
    grade_tbl["share"] = grade_tbl["n"] / grade_tbl["n"].sum()
    pol = pd.DataFrame(GRADES, columns=["pd_upper", "grade", "decision", "pricing_tier", "limit_x_monthly_income"])
    grade_tbl = grade_tbl.reset_index().merge(pol, on="grade")

    # ---- figures
    plot_roc(y, preds)
    plot_calibration(y, preds)
    plot_score_dist(y, score_v)
    plot_iv(sc.iv_table, sc.features)
    shap_imp = shap_outputs(xgbm, prep, valid, preds["XGBoost"])

    # ---- reports
    perf.to_csv(REPORTS / "model_comparison.csv", index=False)
    seg.to_csv(REPORTS / "segment_performance.csv", index=False)
    sc.iv_table.to_csv(REPORTS / "information_value.csv")
    sc.points_table.to_csv(REPORTS / "scorecard_points.csv", index=False)
    cut.to_csv(REPORTS / "cutoff_strategy.csv", index=False)
    grade_tbl.to_csv(REPORTS / "risk_grades.csv", index=False)
    shap_imp.to_csv(REPORTS / "shap_importance.csv", index=False)

    # ---- versioned artifacts
    ref_scores = {"scorecard_score": sc.score(train[features]).tolist()[:50000],
                  "xgb_pd": xgbm.predict_proba(prep.transform(train[features]))[:, 1].tolist()[:50000]}
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(DATA_DIR), "n_train": len(train), "train_default_rate": float(train.TARGET.mean()),
        "champion": "Scorecard (LR)", "challenger": "XGBoost",
        "scorecard_features": sc.features, "all_features": features,
        "excluded_features": ["CODE_GENDER (protected attribute)"],
        "validation": perf.set_index("model").round(4).to_dict(orient="index"),
        "recommended_cutoff": int(rec.cutoff_score), "lgd_assumption": LGD,
        "grades": [dict(zip(["pd_upper", "grade", "decision", "pricing_tier", "limit_x_monthly_income"], r))
                   for r in GRADES],
        "retraining_triggers": {"score_psi": 0.25, "gini_drop_abs": 0.05, "calibration_ratio_outside": [0.8, 1.25]},
    }
    joblib.dump({"scorecard": sc, "prep": prep, "xgb": xgbm, "rf": rf, "meta": meta}, ARTIFACTS / "credit_models.joblib")
    (ARTIFACTS / "model_card.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    joblib.dump({"train": train, "valid": valid, "prod": prod, "features": features, "ref": ref_scores},
                ARTIFACTS / "credit_splits.joblib")

    # ---- Power BI: every applicant scored (valid + prod), with policy outputs
    scored = pd.concat([valid.assign(split="Validation"), prod.assign(split="Production")])
    sp = predict_all(scored, sc, prep, rf, xgbm)
    out = scored[["SK_ID_CURR", "split", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "NAME_CONTRACT_TYPE",
                  "NAME_INCOME_TYPE", "THIN_FILE", "NEW_TO_LENDER", "AGE_YEARS"]].copy()
    out["score"] = sc.score(scored[features])
    out["pd_scorecard"] = sp["Scorecard (LR)"]
    out["pd_xgboost"] = sp["XGBoost"]
    out["grade"] = assign_grade(out["pd_scorecard"])
    out = out.merge(pol[["grade", "decision", "pricing_tier", "limit_x_monthly_income"]], on="grade", how="left")
    out["recommended_limit"] = (out["AMT_INCOME_TOTAL"] / 12 * out["limit_x_monthly_income"]).round(-2)
    out["approved_at_cutoff"] = (out["score"] >= rec.cutoff_score).astype(int)
    out["expected_loss"] = out["pd_scorecard"] * LGD * out["AMT_CREDIT"]
    out["reason_codes"] = ["; ".join(r) for r in sc.reason_codes(scored[features])]
    out.to_csv(POWERBI / "fact_applications.csv", index=False)
    perf.to_csv(POWERBI / "model_comparison.csv", index=False)
    cut.to_csv(POWERBI / "cutoff_strategy.csv", index=False)
    grade_tbl.to_csv(POWERBI / "risk_grades.csv", index=False)
    print(f"[done] artifacts -> {ARTIFACTS}, reports -> {REPORTS}")


if __name__ == "__main__":
    main()
