# Teammate audit

Repository audited: https://github.com/TC960/bikeshare-open

## /leakage-auditor output

NOT AVAILABLE AT SUBMISSION. I ran the `leakage-auditor` subagent on a clone of the repo at about 12:46 IST. It was still running at 13:09 IST (it was reading files and grepping, and I asked it to stop and report what it had), and it had returned no report. Nothing in this section is from the auditor. The shared NRP qwen3 key also stalled my other agent run (`agent/audit.py`), so I think the delay is the endpoint, not the repo, but I did not confirm that.

## My own manual review (not auditor output)

I read `src/features.py` and `src/config.py` in the repo (via a web fetch that was summarised for me, so I have not checked these claims line by line against the files). I could not read `reports/metrics.json` or the repo front page.

- The lag and rolling features use only past days; the rolling window excludes the current day.
- `FORBIDDEN` matches the starter (`casual`, `registered`, `cnt`, `instant`, `yr`, `days_since_2011`, `cnt_2d_bfr`) and the split dates are 2011-12-31 and 2012-06-30.
- They add one feature, `cnt_wow_change = cnt_lag_1 - cnt_lag_7`, which uses only past days.
- Same-day weather (`temp`, `hum`, `windspeed`, `weather_severity`) is not known before the day starts. If the task is next-day forecasting, these would leak. This is a question about the contract, not a code bug.
- `feature_columns` admits any numeric column not in `FORBIDDEN`, so a new raw numeric column would get through unless a test classifies columns. I did not see such a test.
- `cnt_lag_7` is `shift(LAG_WARMUP)`, so raising `LAG_WARMUP` would silently change what the column means. (My own repo has the same starter code; I added a test that pins it to exactly 7 days.)

## Verdict

Nothing in my manual review changes a reported number. The same-day weather question is the only finding that could, and only if the intended use is forecasting a day ahead. This verdict is mine and is not the auditor's. If the auditor's report arrives, it should replace the section above and be compared with this review.
