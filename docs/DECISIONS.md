# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-26: Checked the agent's data profile against pandas: sha256 prefix a6bed1577f6e4de5, 728 rows, 2011-01-03 to 2012-12-31, GOOD/MISTY/BAD labels, one missing day (2011-03-10), casual+registered==cnt on all rows, cnt stats, holiday/workday/weather counts, lag-1/2/3/7 autocorrelation, the 22-ride day on 2012-10-29, yearly means, and the cnt_2d_bfr values around the gap. Every agent fact matched; none was wrong. Changed nothing in the pipeline.
- 2026-11-26: Treat `atemp` as redundant with `temp` and never use atemp - temp: corr 0.9917, atemp ~ 22.36 + 0.639*temp, and atemp is above temp in 727 of 728 rows (range 18.69 to 44.59 vs -5.22 to 32.50), so it is a rescaled copy. H2 2012 (test) has no SPRING days and only 4 BAD days; REPORT.md will say so.
