"""Weight-of-Evidence (WoE) binning, Information Value (IV) and points-based scorecard.

Conventions
-----------
WoE_bin = ln( %goods_in_bin / %bads_in_bin )   -> higher WoE = safer bin
IV      = sum( (%goods - %bads) * WoE )
IV guide: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious (check leakage)

Score   = Offset + Factor * ln(odds_good)       Factor = PDO / ln 2
          Offset = BASE - Factor * ln(ODDS_BASE)
So with the defaults, 600 points = 50:1 good:bad odds and every +20 points doubles the odds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.config import SCORE_BASE, ODDS_BASE, PDO

MISSING = "Missing"
SMOOTH = 0.5


def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(bins, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["goods", "bads"]
    tg, tb = t["goods"].sum(), t["bads"].sum()
    t["count"] = t["goods"] + t["bads"]
    t["share"] = t["count"] / t["count"].sum()
    t["bad_rate"] = t["bads"] / t["count"]
    pg = (t["goods"] + SMOOTH) / (tg + SMOOTH * len(t))
    pb = (t["bads"] + SMOOTH) / (tb + SMOOTH * len(t))
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    return t


class FeatureBinner:
    """Supervised binning for one feature (numeric or categorical)."""

    def __init__(self, name, max_bins=10, min_share=0.05, min_cat_share=0.01):
        self.name, self.max_bins, self.min_share, self.min_cat_share = name, max_bins, min_share, min_cat_share

    # ---------------- numeric ----------------
    def _fit_numeric(self, x, y):
        nonnull = x.notna()
        qs = np.unique(np.nanquantile(x[nonnull], np.linspace(0, 1, self.max_bins + 1)))
        edges = list(qs[1:-1])
        edges = self._merge_until_monotonic(x[nonnull], y[nonnull], edges)
        self.edges = edges
        self.kind = "numeric"

    def _cut(self, x, edges):
        return pd.cut(x, [-np.inf, *edges, np.inf], right=True, labels=False)

    def _merge_until_monotonic(self, x, y, edges):
        """Greedily remove cut points until bad rate is monotonic and bins are large enough.
        Monotonic bins make the scorecard defensible: e.g. risk never rises then falls with income."""
        while edges:
            b = self._cut(x, edges)
            t = pd.DataFrame({"b": b, "y": y}).groupby("b")["y"].agg(["mean", "size"])
            rates, sizes = t["mean"].to_numpy(), t["size"].to_numpy() / len(x)
            direction = np.sign(np.corrcoef(np.arange(len(rates)), rates)[0, 1]) if len(rates) > 2 else 1
            diffs = np.diff(rates) * direction
            bad_idx = None
            if sizes.min() < self.min_share:
                i = int(sizes.argmin())
                bad_idx = min(i, len(edges) - 1) if i == len(rates) - 1 else i
                if i > 0 and i < len(rates) - 1 and sizes[i - 1] < sizes[i + 1]:
                    bad_idx = i - 1
            elif (diffs < 0).any():
                bad_idx = int(np.argmin(diffs))           # most severe reversal
            if bad_idx is None:
                break
            edges = edges[:bad_idx] + edges[bad_idx + 1:]
        return edges

    def _label_numeric(self, x):
        b = self._cut(x, self.edges).astype("float")
        lab = b.map(lambda i: self._labels[int(i)] if not np.isnan(i) else MISSING)
        return lab

    # ---------------- categorical ----------------
    def _fit_categorical(self, x, y):
        share = x.value_counts(normalize=True)
        self.keep = set(share[share >= self.min_cat_share].index)
        self.kind = "categorical"

    def _label_categorical(self, x):
        x = self._as_category(x)
        return x.where(x.isin(self.keep), "Other").astype(str)

    def _as_category(self, x):
        # Low-cardinality numeric flags (0/1) -> stable string labels whether they arrive as int or float
        if getattr(self, "numeric_cat", False):
            v = pd.to_numeric(x, errors="coerce")
            return v.map(lambda a: MISSING if pd.isna(a) else f"{a:g}")
        return x.fillna(MISSING).astype(str)

    # ---------------- public ----------------
    def fit(self, x: pd.Series, y: pd.Series):
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 2:
            self._fit_numeric(x, y)
            e = [-np.inf, *self.edges, np.inf]
            self._labels = [f"({e[i]:.4g}, {e[i + 1]:.4g}]" for i in range(len(e) - 1)]
        else:
            self.numeric_cat = pd.api.types.is_numeric_dtype(x)
            self._fit_categorical(self._as_category(x), y)
        labels = self.labels(x)
        self.table = _woe_table(labels, y)
        if self.kind == "numeric":         # keep bins in numeric order, Missing last
            order = [l for l in self._labels if l in self.table.index]
            order += [l for l in self.table.index if l not in order]
            self.table = self.table.loc[order]
        self.table.index.name = "bin"
        self.iv = float(self.table["iv"].sum())
        self.woe_map = self.table["woe"].to_dict()
        return self

    def labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            return self._label_numeric(pd.to_numeric(x, errors="coerce"))
        return self._label_categorical(x)

    def transform(self, x: pd.Series) -> pd.Series:
        # unseen bin (e.g. no missing values in training) -> neutral WoE of 0
        return self.labels(x).map(self.woe_map).fillna(0.0).astype(float)


class WoEScorecard:
    def __init__(self, min_iv=0.02, max_iv=0.8, max_corr=0.7, C=0.1):
        self.min_iv, self.max_iv, self.max_corr, self.C = min_iv, max_iv, max_corr, C

    def fit_binning(self, X: pd.DataFrame, y: pd.Series):
        self.binners = {c: FeatureBinner(c).fit(X[c], y) for c in X.columns}
        self.iv_table = (pd.Series({c: b.iv for c, b in self.binners.items()}, name="IV")
                         .sort_values(ascending=False).to_frame())
        self.iv_table["strength"] = pd.cut(self.iv_table["IV"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                                           labels=["useless", "weak", "medium", "strong", "suspicious"])
        return self

    def woe(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or list(self.binners)
        return pd.DataFrame({c: self.binners[c].transform(X[c]).to_numpy() for c in cols}, index=X.index)

    def select_features(self, X: pd.DataFrame):
        """IV filter, then drop the weaker of any highly correlated WoE pair."""
        cand = self.iv_table[(self.iv_table.IV >= self.min_iv) & (self.iv_table.IV <= self.max_iv)].index.tolist()
        W = self.woe(X, cand)
        corr = W.corr().abs()
        selected = []
        for c in cand:                      # already ordered by IV, strongest first
            if all(corr.loc[c, s] < self.max_corr for s in selected):
                selected.append(c)
        self.dropped_corr = [c for c in cand if c not in selected]
        return selected

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_binning(X, y)
        feats = self.select_features(X)
        W = self.woe(X, feats)
        # Iteratively drop features whose sign contradicts their WoE (all coefs must be negative,
        # since higher WoE = safer). Keeps every points table intuitive for credit committees.
        while True:
            lr = LogisticRegression(C=self.C, max_iter=1000)
            lr.fit(W[feats], y)
            coefs = pd.Series(lr.coef_[0], index=feats)
            wrong = coefs[coefs > 0]
            if wrong.empty:
                break
            feats.remove(wrong.idxmax())
        self.features, self.model = feats, lr
        self.coefs = coefs
        self._build_points()
        return self

    def _build_points(self):
        self.factor = PDO / np.log(2)
        self.offset = SCORE_BASE - self.factor * np.log(ODDS_BASE)
        n = len(self.features)
        b0 = self.model.intercept_[0]
        rows = []
        for f in self.features:
            beta = self.coefs[f]
            for bin_label, r in self.binners[f].table.iterrows():
                # ln(odds_good) = -(b0 + sum beta*woe)  -> distribute intercept + offset evenly
                pts = self.offset / n + self.factor * (-(beta * r["woe"]) - b0 / n)
                rows.append({"feature": f, "bin": bin_label, "count": int(r["count"]),
                             "bad_rate": r["bad_rate"], "woe": r["woe"], "coef": beta,
                             "points": round(pts)})
        self.points_table = pd.DataFrame(rows)
        self._points_map = {f: g.set_index("bin")["points"].to_dict()
                            for f, g in self.points_table.groupby("feature")}
        self._max_points = self.points_table.groupby("feature")["points"].max().to_dict()

    # ---------------- scoring ----------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe(X, self.features))[:, 1]

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for f in self.features:
            lab = self.binners[f].labels(X[f])
            default = round(self.offset / len(self.features) - self.factor * self.model.intercept_[0] / len(self.features))
            out[f] = lab.map(self._points_map[f]).fillna(default).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.points(X).sum(axis=1).to_numpy()

    def reason_codes(self, X: pd.DataFrame, top=3) -> list[list[str]]:
        """Adverse-action style reasons: features where the applicant lost most points vs the best bin."""
        P = self.points(X)
        lost = np.column_stack([self._max_points[f] - P[f].to_numpy() for f in self.features])
        order = np.argsort(-lost, axis=1, kind="stable")[:, :top]      # vectorised: fast on 100k+ rows
        names = np.array(self.features)
        out = []
        for i in range(len(lost)):
            out.append([f"{names[j]} (-{int(lost[i, j])} pts)" for j in order[i] if lost[i, j] > 0])
        return out

    def score_to_pd(self, score):
        return 1 / (1 + np.exp((np.asarray(score) - self.offset) / self.factor))
