"""
Predict monthly NetSales using:
1) Seasonal-naive baseline (forecast month t uses last year's same month).
2) RandomForestRegressor with simple time features + lags.

Outputs:
- data/predictions_monthly.csv (forecasts for next 3 months per Region+Category)
- screenshots/forecast_example.png (one example plot)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/dummy_retail_sales.csv")
OUT_CSV = Path("data/predictions_monthly.csv")
PLOT_PATH = Path("screenshots/forecast_example.png")

# Grouping level to forecast at (edit to taste):
GROUP_COLS = ["Region", "Category"]   # or ["StoreID"], or ["Region"], etc.

# Forecast horizon (months)
HORIZON = 3

# Minimum history (months) required to train ML model
MIN_HISTORY = 15

# RandomForest settings (lightweight)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Helpers
# -----------------------------
def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Keep Month as month-start Timestamp for consistency
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby(GROUP_COLS + ["Month"], as_index=False)["NetSales"].sum()
        .sort_values(GROUP_COLS + ["Month"])
    )
    return monthly

def add_time_features(fr: pd.DataFrame) -> pd.DataFrame:
    fr = fr.copy()
    fr["year"] = fr["Month"].dt.year
    fr["month"] = fr["Month"].dt.month
    # Cyclical encoding for month
    fr["month_sin"] = np.sin(2 * np.pi * fr["month"] / 12)
    fr["month_cos"] = np.cos(2 * np.pi * fr["month"] / 12)
    return fr

def add_lags(fr: pd.DataFrame, target="NetSales", lags=(1, 2, 12), roll=(3, 6)):
    fr = fr.copy()
    for l in lags:
        fr[f"lag_{l}"] = fr.groupby(GROUP_COLS)[target].shift(l)
    for w in roll:
        fr[f"roll_mean_{w}"] = fr.groupby(GROUP_COLS)[target].shift(1).rolling(w).mean()
    return fr

def seasonal_naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """
    Forecast using same-month-last-year. If not enough data, fallback to last-observation.
    """
    y = history.values
    if len(y) >= 12:
        return y[-12:][:horizon]  # first H of last 12 months
    else:
        return np.repeat(y[-1], horizon)

def train_rf_and_forecast(df_grp: pd.DataFrame, horizon: int):
    """
    Train RF with expanding-window CV; produce rolling-origin forecast for last known periods (for scoring),
    then fit on full history and produce next H months forecasts.
    """
    fr = add_time_features(df_grp)
    fr = add_lags(fr)

    feature_cols = ["month", "month_sin", "month_cos",
                    "lag_1", "lag_2", "lag_12",
                    "roll_mean_3", "roll_mean_6"]
    fr = fr.dropna(subset=feature_cols + ["NetSales"]).reset_index(drop=True)

    if len(fr) < MIN_HISTORY:
        return None, np.inf  # not enough history for ML

    X = fr[feature_cols].values
    y = fr["NetSales"].values

    # time series split (no shuffling)
    tscv = TimeSeriesSplit(n_splits=3)
    preds, actuals = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        preds.append(p)
        actuals.append(y_te)
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mape = MAPE(actuals, preds)

    # Fit on all
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)

    # Create future feature rows for next H months (Period-based to avoid int+Timestamp issues)
    last_period = df_grp["Month"].max().to_period("M")
    future_months = pd.period_range(last_period + 1, periods=horizon, freq="M").to_timestamp()

    # Build a small frame to compute lags/rolls
    full = pd.concat([
        df_grp[["Month", "NetSales"]].copy(),
        pd.DataFrame({"Month": future_months, "NetSales": np.nan})
    ], ignore_index=True)

    # Ensure Month is month-start Timestamp
    full["Month"] = pd.to_datetime(full["Month"]).dt.to_period("M").dt.to_timestamp()

    full = add_time_features(full)
    full = add_lags(full)

    # For future rows, take features
    future = full[full["Month"].isin(future_months)].copy()
    future = future.dropna(subset=["lag_1", "lag_2", "lag_12", "roll_mean_3", "roll_mean_6"], how="any")

    # If we lost rows due to missing lags, fallback entirely to seasonal naive
    if len(future) < horizon:
        return None, mape

    X_future = future[feature_cols].values
    y_future = model.predict(X_future)
    return y_future, mape

def forecast_group(df_grp: pd.DataFrame, horizon: int):
    # Baseline
    baseline = seasonal_naive_forecast(df_grp["NetSales"], horizon)
    # RF
    rf_preds, rf_mape = train_rf_and_forecast(df_grp, horizon)

    # Backtest MAPE for baseline (compare last 12 months to previous 12)
    if len(df_grp) >= 24:
        y = df_grp["NetSales"].values
        baseline_backtest = y[-12:]       # predicted last 12 as same-month prev year
        prev_year = y[-24:-12]
        base_mape = MAPE(baseline_backtest, prev_year)
    else:
        base_mape = np.inf

    # Choose model with lower MAPE; if RF invalid, use baseline
    use_rf = (rf_preds is not None) and (rf_mape < base_mape)
    preds = rf_preds if use_rf else baseline
    model_used = "RandomForest" if use_rf else "SeasonalNaive"

    # Build result frame (Period-based to avoid int+Timestamp issues)
    last_period = df_grp["Month"].max().to_period("M")
    future_months = pd.period_range(last_period + 1, periods=horizon, freq="M").to_timestamp()

    out = pd.DataFrame({
        "Month": future_months,
        "Forecast_NetSales": preds,
        "Model": model_used
    })
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    monthly = prepare_monthly(df)

    forecasts = []
    for keys, grp in monthly.groupby(GROUP_COLS, as_index=True):
        grp = grp.sort_values("Month").reset_index(drop=True)
        fc = forecast_group(grp, HORIZON)
        # attach group keys
        if isinstance(keys, tuple):
            for k, v in zip(GROUP_COLS, keys):
                fc[k] = v
        else:
            fc[GROUP_COLS[0]] = keys
        forecasts.append(fc)

    result = pd.concat(forecasts, ignore_index=True)
    # tidy columns order
    result = result[GROUP_COLS + ["Month", "Forecast_NetSales", "Model"]].sort_values(GROUP_COLS + ["Month"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote forecasts to {OUT_CSV} ({len(result)} rows)")
    
    # Optional: plot one example series + forecast
    try:
        # pick first group
        first_keys = next(iter(monthly.groupby(GROUP_COLS)))[0]
        if not isinstance(first_keys, tuple):
            first_keys = (first_keys,)
        filt = (monthly[GROUP_COLS] == pd.Series(first_keys, index=GROUP_COLS)).all(axis=1)
        hist = monthly[filt].sort_values("Month")
        fut = result[(result[GROUP_COLS] == pd.Series(first_keys, index=GROUP_COLS)).all(axis=1)]

        plt.figure(figsize=(11, 6))
        plt.plot(hist["Month"], hist["NetSales"], label="History")
        plt.plot(fut["Month"], fut["Forecast_NetSales"], marker="o", label="Forecast")
        plt.title(f"Forecast example: {', '.join(map(str, first_keys))}")
        plt.ylabel("NetSales")
        plt.xticks(rotation=45)
        plt.legend()
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        print(f"🖼️  Saved example plot to {PLOT_PATH}")
    except Exception as e:
        print("Plotting skipped:", e)

if __name__ == "__main__":
    main()