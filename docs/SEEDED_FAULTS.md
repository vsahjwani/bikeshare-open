# Seeded faults

Each fault was introduced with a one-line `sed` edit to `src/features.py`, run against `pytest tests/ -q`, committed as its own commit, and then undone with `git revert` (see `git log`). The output below is the real captured pytest output (last 12 lines).

All three faults were caught by an existing test. Before the Step 5 tests, fault 3 would not have been caught (the weather encoder test was added in Step 5 because mutation testing showed it was uncovered).

## Fault 1: leaky columns reach the model
Edit: `feature_columns()` no longer filters out `FORBIDDEN` (line 84).
Caught by: `tests/test_leakage.py::test_no_forbidden_features`.

```
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'instant', 'registered', 'casual', 'days_since_2011', 'cnt_2d_bfr', 'cnt', 'yr'}
E       assert not ({'atemp', 'casual', 'cnt', 'cnt_2d_bfr', 'cnt_lag_1', 'cnt_lag_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:36: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 18 passed in 2.11s
```

## Fault 2: rolling mean includes today
Edit: `.shift(1)` removed from `cnt_roll_7` (line 76), so the window includes the day being predicted.
Caught by: `tests/test_leakage.py::test_rolling_mean_excludes_today`.

```
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
E             
E             comparison failed
E             Obtained: 4389.0
E             Expected: 4414.857142857143 ± 0.00441486

tests/test_leakage.py:62: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 18 passed in 0.99s
```

## Fault 3: unknown weather label silently becomes NaN
Edit: the `raise ValueError` in `encode_weather` replaced by `pass` (line 30). An unseen label would become NaN and `dropna()` would silently delete the row.
Caught by: `tests/test_contract.py::test_encode_weather_rejects_unknown_label` (written in Step 5).

```
        us exactly the days we care about while every test stayed green.
    
        A unit test: needs no CSV, so it still runs offline.
        """
>       with pytest.raises(ValueError, match="unmapped weather labels"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_contract.py:110: Failed
=========================== short test summary info ============================
FAILED tests/test_contract.py::test_encode_weather_rejects_unknown_label - Fa...
1 failed, 18 passed in 0.98s
```

## Fourth fault (copying a file into data/raw/ from Bash)
Not tested. I ran out of time. The deny rules in `.claude/settings.json` cover the Edit/Write tools, not Bash, so a Bash `cp` into `data/raw/` is a known gap that I did not close or verify.
