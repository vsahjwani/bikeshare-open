"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import LAG_WARMUP, TARGET, unclassified_columns
from src.features import (
    build_features,
    encode_weather,
    feature_columns,
    reindex_to_calendar,
)
from src.ingest import load_raw
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_beats_the_mean_baseline(feats):
    """A model that cannot beat predicting the training mean is not a result."""
    cols = feature_columns(feats)
    train, _, test = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(test[TARGET], ridge.predict(test[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS


def test_every_raw_column_is_classified(raw):
    """Every column in data/raw must be forbidden, used, or knowingly unused.

    This is the guard that feature_columns() cannot be. It filters on dtype
    before it filters on FORBIDDEN, so a leaky string column disappears with
    no trace and a leaky numeric one is admitted only if someone remembered
    to list it. Either way the model keeps scoring well and nobody learns
    anything. Failing here forces the decision into src/config.py, where the
    next reader sees it.
    """
    unclassified = unclassified_columns(raw.columns)
    assert not unclassified, (
        f"unclassified raw columns: {unclassified}. Add each name to "
        "FORBIDDEN, USED_COLS or KNOWN_UNUSED_COLS in src/config.py, with a "
        "one-line reason."
    )


def test_unclassified_columns_detects_a_new_column():
    """Proves the guard bites rather than merely passing.

    test_every_raw_column_is_classified passes on today's data, which is no
    evidence it would notice tomorrow's. Feed the same function a column
    that does not exist yet -- here a plausible upstream addition that is
    numeric and would otherwise go straight into the model -- and it must
    flag it.
    """
    made_up = ["dteday", "temp", "cnt", "casual", "cnt_2d_bfr", "registered_yesterday"]
    assert unclassified_columns(made_up) == ["registered_yesterday"]
    # a fully covered frame reports nothing
    assert unclassified_columns(["dteday", "temp", "season", "cnt"]) == []


def test_encode_weather_rejects_unknown_label():
    """An unseen weather label must raise, not become NaN.

    `.map()` silently yields NaN for anything not in WEATHER_ORDER, and the
    `dropna()` at the end of build_features() then deletes the row with no
    error and no trace. The 20 BAD-weather days are the most informative
    rows in this dataset, so a fourth label arriving upstream would cost
    us exactly the days we care about while every test stayed green.

    A unit test: needs no CSV, so it still runs offline.
    """
    with pytest.raises(ValueError, match="unmapped weather labels"):
        encode_weather(pd.Series(["GOOD", "MISTY", "HURRICANE"]))
    # and the known labels still encode, so the guard cannot pass by
    # raising unconditionally
    got = encode_weather(pd.Series(["GOOD", "MISTY", "BAD"]))
    assert list(got) == [0.0, 1.0, 2.0]


def test_no_lag_crosses_an_absent_day(feats, raw):
    """No surviving row may sit within LAG_WARMUP days after a missing day.

    This is the test that checks the CALL SITE. The gap itself is fine in
    isolation -- reindex_to_calendar() passes it back a complete grid, and
    test_calendar_is_complete_before_lagging asserts that. None of that
    matters if build_features() forgets to call it. Here we assert that the
    hole actually reached the output, so deleting the reindex makes the row
    after the gap carry a two-day lag labelled as one-day.

    Derived from the data rather than hard-coded to 2011-03-10, so it keeps
    working if upstream repairs the mirror -- and the count is asserted
    too, so a *new* hole fails here instead of being absorbed quietly.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing = full.difference(raw.dteday)
    assert len(missing) == 1, (
        f"expected the one documented gap (2011-03-10), found {len(missing)}: "
        f"{[str(m.date()) for m in missing]}. Update LAG_WARMUP accounting."
    )
    for day in missing:
        # every lag we build looks back at most LAG_WARMUP days, so a row on
        # day+1 .. day+LAG_WARMUP would read across the hole
        tainted = pd.date_range(day, day + pd.Timedelta(days=LAG_WARMUP), freq="D")
        present = sorted(str(d.date()) for d in feats.dteday if d in set(tainted))
        assert not present, (
            f"rows survive inside the {LAG_WARMUP}-day window after missing "
            f"{day.date()}: {present}. Their lags span an absent day, so "
            f"cnt_lag_1/cnt_roll_7 are silently mislabelled -- is "
            f"reindex_to_calendar() still called in build_features()?"
        )
    # every remaining date jump must be explained by a missing day. One hole
    # costs LAG_WARMUP+1 rows (itself plus the warm-up), so the real table
    # jumps 9 days at 2011-03-09 -> 2011-03-18; claiming contiguity is wrong.
    missing_set = set(missing)
    ds = feats.dteday.sort_values().reset_index(drop=True)
    for i in range(1, len(ds)):
        gap = (ds[i] - ds[i - 1]).days
        if gap > 1:
            between = set(pd.date_range(ds[i - 1] + pd.Timedelta(days=1),
                                        ds[i] - pd.Timedelta(days=1)))
            assert between & missing_set, (
                f"{gap}-day jump {ds[i-1].date()} -> {ds[i].date()} contains no "
                "known-absent day: an unexplained hole in the feature table"
            )
