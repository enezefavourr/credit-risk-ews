"""Credit-risk model metrics used in validation and in production monitoring."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p):
    """Kolmogorov-Smirnov: max separation between cumulative bad and good distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"AUC": roc_auc_score(y, p), "Gini": gini(y, p), "KS": ks(y, p),
            "Brier": brier_score_loss(y, p), "mean_PD": p.mean(), "actual_DR": y.mean()}


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    t = df.groupby("decile").agg(n=("y", "size"), predicted_PD=("p", "mean"), actual_DR=("y", "mean"))
    return t.reset_index()


def hosmer_lemeshow(y, p, bins=10):
    """HL chi-square calibration test (high p-value = no evidence of miscalibration)."""
    from scipy.stats import chi2
    t = calibration_table(y, p, bins)
    exp_b = t["n"] * t["predicted_PD"]
    obs_b = t["n"] * t["actual_DR"]
    stat = (((obs_b - exp_b) ** 2) / (exp_b * (1 - t["predicted_PD"]))).sum()
    return float(stat), float(1 - chi2.cdf(stat, bins - 2))


def psi(expected, actual, bins=10, edges=None) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 monitor, >0.25 significant shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if edges is None:
        edges = psi_edges(expected, bins)
    e = _bucket_share(expected, edges)
    a = _bucket_share(actual, edges)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_edges(expected, bins=10):
    expected = np.asarray(expected, float)
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1))[1:-1])
    return edges


def _bucket_share(x, edges):
    miss = np.isnan(x)
    idx = np.digitize(x[~miss], edges)
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, miss.sum())             # missing as its own bucket
    share = counts / max(len(x), 1)
    return np.clip(share, 1e-4, None)


def categorical_psi(expected, actual) -> float:
    e = pd.Series(expected).fillna("Missing").astype(str).value_counts(normalize=True)
    a = pd.Series(actual).fillna("Missing").astype(str).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = e.reindex(cats, fill_value=0).clip(lower=1e-4)
    a = a.reindex(cats, fill_value=0).clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def psi_status(v):
    return "Green" if v < 0.1 else ("Amber" if v < 0.25 else "Red")
