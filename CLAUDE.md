# Bikeshare forecasting

## Data
- Daily bikeshare counts, 2011-2012. Target: `cnt`.
- Raw data is `data/raw/bike_daily.csv`. Never modify it.

## Rules
- Leakage: never use `casual` or `registered` as features, because casual + registered == cnt exactly. `src/config.py` lists every excluded column and why.
- Splits: never split randomly. Use `src.train.temporal_split`. Train on 2011, validate on H1 2012, test on H2 2012. Never tune on the test set.
- Baselines: report the mean baseline and the persistence baseline (yesterday's count) with every model score.
