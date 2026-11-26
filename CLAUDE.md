# Bikeshare daily demand (UCI, 2011-2012)

Predict `cnt` (rides per day) from weather, calendar and past demand.

## Commands
- `make all` runs ingest, features, train; reproduces `reports/metrics.json` (the headline numbers).
- `make test` or `python -m pytest tests/ -q` runs the 19 tests. Run them after every change in `src/`.

## Never do
- Never modify `data/raw/`. Raw data is read-only; write derived files to `data/processed/`.
- Never use these as features: `casual`, `registered`, `cnt` (casual + registered == cnt exactly, so they leak the target), `cnt_2d_bfr` (unverified), `instant`, `yr`, `days_since_2011` (time index, lets the model memorise the trend). The list is `FORBIDDEN` in `src/config.py`.
- Never tune, select or look at anything on the test split.

## Split and baselines
- Split by date, never randomly: train 2011, validate H1 2012, test H2 2012 (`TRAIN_END`, `VALID_END` in `src/config.py`).
- Always report both baselines next to any model: `baseline_mean` and `baseline_persistence` (predict `cnt_lag_1`). A model that does not beat persistence is not worth shipping.

## Features
- Lags use past values only. 2011-03-10 is missing from the data, so `reindex_to_calendar` must run before any `shift`; otherwise a lag labelled one day spans two.
- If you add a longer lag, raise `LAG_WARMUP` in `src/config.py`.
- Every raw column must be listed in `USED_COLS`, `KNOWN_UNUSED_COLS` or `FORBIDDEN`; a new column fails `test_every_raw_column_is_classified` until you classify it.
- `encode_weather` raises on an unknown label on purpose. Do not turn the raise into a silent NaN.

## Agent setup
- A PostToolUse hook re-runs the leakage tests after every Edit or Write in `src/`.
- Skills: `new-feature` (add a feature with an availability argument), `run-pipeline`.
- Deny rules block Edit/Write to `data/raw/` but NOT Bash: do not `cp` or redirect into `data/raw/`.
