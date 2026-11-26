"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import LAG_WARMUP, TARGET, unclassified_columns
from src.features import build_features, feature_columns, reindex_to_calendar
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
