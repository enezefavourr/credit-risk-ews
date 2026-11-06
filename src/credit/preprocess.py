"""Shared objects that are pickled with the models (must live in an importable module)."""
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from src.credit.features import CATEGORICAL

# PD bands -> risk grades and illustrative credit policy
GRADES = [  # (upper PD, grade, decision, pricing tier, limit as x monthly income)
    (0.02, "A", "Approve", "Prime", 3.0),
    (0.04, "B", "Approve", "Standard", 2.5),
    (0.08, "C", "Approve", "Standard +", 1.5),
    (0.15, "D", "Refer", "High risk", 1.0),
    (1.01, "E", "Decline", "-", 0.0),
]


def assign_grade(pd_):
    pd_ = np.atleast_1d(np.asarray(pd_, dtype=float))
    out = np.empty(len(pd_), dtype=object)
    lower = 0
    for upper, g, *_ in GRADES:
        out[(pd_ >= lower) & (pd_ < upper)] = g
        lower = upper
    return out


def grade_policy(grade):
    for upper, g, decision, tier, limit in GRADES:
        if g == grade:
            return {"decision": decision, "pricing_tier": tier, "limit_x_monthly_income": limit}
    raise KeyError(grade)


class TreePreprocessor:
    """Ordinal-encode categoricals; trees handle the rest (XGBoost handles NaN natively)."""

    def __init__(self, features):
        self.features = features
        self.cats = [c for c in features if c in CATEGORICAL]
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                  encoded_missing_value=-1)

    def fit(self, X):
        self.enc.fit(X[self.cats].astype(str))
        return self

    def transform(self, X, fill_na=False):
        Z = X[self.features].copy()
        Z[self.cats] = self.enc.transform(Z[self.cats].astype(str))
        Z = Z.astype(float)
        if fill_na:
            Z = Z.fillna(-999)
        return Z
