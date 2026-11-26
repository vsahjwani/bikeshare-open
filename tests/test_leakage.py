"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import pandas as pd
import pytest

from src.config import FORBIDDEN, LAG_WARMUP, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique


def test_lag_7_is_a_true_calendar_lag(feats, raw):
    """cnt_lag_7 must equal cnt exactly LAG_WARMUP calendar days earlier.

    Nothing asserted anything about this column before, which means
    shift(LAG_WARMUP) at src/features.py:75 could become shift(1) -- an
    exact copy of cnt_lag_1 -- or be deleted, and the suite stayed green
    while the model kept using it.

    raw rather than feats supplies the truth: by construction, every
    LAG_WARMUP-day window after the gap was dropped from feats, so
    checking only against feats would find nothing to check.
    """
    by_date = raw.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_7.items():
        prev = date - pd.Timedelta(days=LAG_WARMUP)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), (
                f"cnt_lag_7 wrong at {date.date()}: {lag} != cnt({prev.date()}) "
                f"= {by_date[prev]}"
            )
            checked += 1
    assert checked > 600, f"only {checked} rows verified"
    # a duplicate of cnt_lag_1 would satisfy nothing; prove the two differ
    assert (feats.cnt_lag_7 != feats.cnt_lag_1).mean() > 0.5, (
        "cnt_lag_7 is nearly identical to cnt_lag_1 -- check its shift"
    )
