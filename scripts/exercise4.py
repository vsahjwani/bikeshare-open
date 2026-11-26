import json
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.config import TARGET
from src.features import feature_columns
from src.train import temporal_split

df = pd.read_parquet("data/processed/features.parquet")
tr, va, te = temporal_split(df)
cols = feature_columns(df)

# Transformation: predict the change from yesterday, add yesterday back.
model = HistGradientBoostingRegressor(random_state=0)
model.fit(tr[cols], tr[TARGET] - tr["cnt_lag_1"])

out = {}
for name, part in [("valid", va), ("test", te)]:
    pred = model.predict(part[cols]) + part["cnt_lag_1"]
    out[name] = {"r2": float(r2_score(part[TARGET], pred)),
                 "mae": float(mean_absolute_error(part[TARGET], pred))}
    print(name, out[name])

json.dump({"hgb_delta": out}, open("reports/exercise4.json", "w"), indent=2)
