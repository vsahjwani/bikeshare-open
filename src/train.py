"""Stage 3: temporal split, fit, evaluate, write metrics.

Run: python -m src.train

Two baselines and two models, on purpose.

  baseline_mean         predict the training mean (the 2011 average)
  baseline_persistence  predict yesterday's count, cnt_lag_1, unchanged
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence is
the baseline that actually matters: ridge's features are dominated by lagged
demand, so if it cannot beat "yesterday again" it has learned nothing worth
reporting. The booster is here because it fails, and the reason it fails is
the headline result.

Nothing in this module is selected using the test set. That is enforced, not
merely promised -- see _assert_never_fits_test() and fit_models(). The test
split enters scope exactly once, in score(), after every estimator is fitted
and every hyperparameter is frozen.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns

# Pre-registered, not tuned. Ridge alpha is deliberately fixed rather than
# searched: on the H1-2012 validation split the curve is flat between 0.3
# (R^2 0.651) and 1.0 (R^2 0.648), a gap far smaller than the 184-day test
# set can resolve, so "selecting" it would be fitting noise and dressing it
# up as a decision. 1.0 stays because DECISIONS.md already reports the number
# it produces, and comparability across runs beats a third decimal place.
RIDGE_ALPHA = 1.0
HGB_MAX_ITER = 300


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict one constant: the training mean.

    Written out rather than borrowed from DummyRegressor so that "the 2011
    average" is visible in the code that defines it, and so every entry in
    MODELS is the same kind of object.
    """

    def fit(self, X, y=None):
        self.constant_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_)


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit.

    A real baseline, not a DummyRegressor: it reads yesterday's count from
    the feature table, so it survives a reindex or a reordered split where a
    constant predictor would look deceptively steady. fit() validates the
    column exists, which turns a renamed lag into an error at train time
    instead of a silent fallback.
    """

    PERSISTENCE_COL = "cnt_lag_1"

    def __init__(self, column: str = PERSISTENCE_COL):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(
                f"persistence baseline needs column {self.column!r}; "
                f"available: {sorted(X.columns)}"
            )
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": MeanBaseline(),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=HGB_MAX_ITER),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test.

    Returns (train, valid, test) for 2011, H1 2012, H2 2012. The three parts
    are asserted disjoint and exhaustive: a silent gap or overlap here would
    move rows between splits without changing any score's sign, which is the
    kind of error a R^2 never reports.
    """
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]

    for name, part in (("train", train), ("valid", valid), ("test", test)):
        if part.empty:
            raise ValueError(f"{name} split is empty -- check TRAIN_END/VALID_END")
    ordered = train.dteday.max() < valid.dteday.min() < valid.dteday.max() < test.dteday.min()
    if not ordered:
        raise ValueError("splits are not ordered in time")
    seen = pd.concat([train.dteday, valid.dteday, test.dteday])
    if not seen.is_unique:
        raise ValueError("splits overlap: a date appears in more than one part")
    if len(seen) != len(df):
        raise ValueError(f"splits cover {len(seen)} of {len(df)} rows")
    return train, valid, test


def _assert_never_fits_test(dates: pd.Series, what: str) -> None:
    """Fail loudly if anything on the fitting path has seen the test window."""
    leaked = dates[dates > VALID_END]
    if len(leaked):
        raise RuntimeError(
            f"{what} was fitted on {len(leaked)} test-set rows "
            f"(earliest {leaked.min().date()}). Nothing may be tuned on H2 2012."
        )


def fit_models(train: pd.DataFrame, cols: list[str]) -> dict:
    """Fit every model on the training window only.

    The guard is per-model, not global: a future entry in MODELS that does
    internal resampling still cannot reach H2 2012 through this door.
    """
    _assert_never_fits_test(train.dteday, "train split")
    fitted = {}
    for name, model in MODELS.items():
        _assert_never_fits_test(train.dteday, f"model {name!r}")
        model.fit(train[cols], train[TARGET])
        fitted[name] = model
    return fitted


def score(fitted: dict, valid: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict:
    """Evaluate the frozen models. The only place the test set is read."""
    out = {}
    for name, model in fitted.items():
        out[name] = {
            split_name: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
            }
            for split_name, part in (("valid", valid), ("test", test))
        }
    return out


def fit_and_score() -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    # Fitting and hyperparameters are frozen before `test` is passed anywhere.
    fitted = fit_models(train, cols)
    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "ridge_alpha": RIDGE_ALPHA,
    }
    results.update(score(fitted, valid, test, cols))
    return results


if __name__ == "__main__":
    res = fit_and_score()
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
