# -*- coding: utf-8 -*-
"""CSI 300 six-session log-return model comparison using the Wind workbook.

The prediction target is log(Close[t+6] / Close[t]): six sessions after the
feature date (e.g. 2026-11-03 -> 2026-11-11). Predicted returns
are converted back to index points for scoring and the fixed-date forecast.
Run with the project's TensorFlow environment:
    D:/Anaconda/envs/tf/python.exe model.py
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "CSI300_index_wind.xlsx"
OUT = ROOT / "model_outputs" / "11.11_six_session"
OUT.mkdir(exist_ok=True)

TRAIN_START = pd.Timestamp("2025-10-16")
CUTOFF = pd.Timestamp("2026-07-16")  # fixed historical holdout boundary
LOOKBACK = 10
HORIZON_SESSIONS = 6  # 11/04, 11/05, 11/06, 11/09, 11/10, 11/11 after 11/03
FORECAST_FEATURE_DATE = pd.Timestamp("2026-11-03")
FORECAST_TARGET_DATE = pd.Timestamp("2026-11-11")
TREND_MA_WINDOW = 20
TREND_SLOPE_WINDOW = 5
STRATEGY_NAME = "RF_if_Uptrend_else_RandomWalk"
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def find_column(columns, prefix):
    matches = [col for col in columns if str(col).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"需要唯一的列前缀 {prefix!r}，实际匹配到 {matches}")
    return matches[0]


def make_models():
    return {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", n_estimators=180, max_depth=2,
            learning_rate=0.025, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=8, reg_alpha=0.1,
            random_state=SEED, n_jobs=1, verbosity=0,
        ),
        "SVR": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel="rbf")),
        "RIDGE": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RF": RandomForestRegressor(
            n_estimators=500, max_depth=3, min_samples_leaf=5,
            max_features=0.8, random_state=SEED, n_jobs=-1,
        ),
        "FCNN": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16, 8, 4), activation="relu", solver="adam",
                alpha=0.05, learning_rate_init=0.003, max_iter=2000,
                early_stopping=True, validation_fraction=0.2,
                n_iter_no_change=50, tol=1e-5, random_state=SEED,
            ),
        ),
    }


def make_lstm(n_features):
    net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, n_features)),
        tf.keras.layers.LSTM(12),
        tf.keras.layers.Dense(1),
    ])
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss="mse")
    return net


def score(actual, predicted, previous_close):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "MAE_points": float(np.mean(np.abs(error))),
        "RMSE_points": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_percent": float(np.mean(np.abs(error / actual)) * 100),
        "Directional_accuracy_percent": float(
            np.mean(np.sign(predicted - previous_close) == np.sign(actual - previous_close)) * 100
        ),
    }


# The workbook has one leading instrument-code row; invalid dates remove it.
raw = pd.read_excel(SOURCE, sheet_name="sheet1")
date_col = raw.columns[0]
raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
raw = raw.loc[raw[date_col].notna()].copy().sort_values(date_col).reset_index(drop=True)
raw = raw.rename(columns={date_col: "Date"})
close_col = find_column(raw.columns, "收盘价")
raw["Close"] = pd.to_numeric(raw[close_col], errors="coerce")
raw["TrendMA"] = raw["Close"].rolling(TREND_MA_WINDOW).mean()
raw["TrendMAChange"] = raw["TrendMA"] / raw["TrendMA"].shift(TREND_SLOPE_WINDOW) - 1
raw["TrendUp"] = (raw["Close"] > raw["TrendMA"]) & (raw["TrendMAChange"] > 0)

# Columns 8 onward are the Wind technical indicators. Ignore the synchronous
# broad-market reference factor to avoid directly feeding another index series.
factor_cols = [
    col for col in raw.columns[7:]
    if col not in {"Close", close_col, "TrendMA", "TrendMAChange", "TrendUp"}
    and "大盘同步指标" not in str(col)
]
features = pd.DataFrame({col: pd.to_numeric(raw[col], errors="coerce") for col in factor_cols})
features = features.replace([np.inf, -np.inf], np.nan)
all_rows = pd.concat([raw[["Date", "Close", "TrendUp"]], features], axis=1)
all_rows["TargetDate"] = all_rows["Date"].shift(-HORIZON_SESSIONS)
all_rows["TargetClose"] = all_rows["Close"].shift(-HORIZON_SESSIONS)
all_rows["TargetLogReturn"] = np.log(all_rows["TargetClose"] / all_rows["Close"])
valid = all_rows[factor_cols].notna().all(axis=1) & all_rows["TargetLogReturn"].notna()
frame = all_rows.loc[valid].reset_index(drop=True)
if frame.empty:
    raise ValueError("Wind 文件中没有可用于建模的完整指标行。")

train_mask = frame["TargetDate"].between(TRAIN_START, CUTOFF, inclusive="both")
test_mask = frame["TargetDate"].gt(CUTOFF)
train_idx = np.flatnonzero(train_mask.to_numpy())
test_idx = np.flatnonzero(test_mask.to_numpy())
if len(train_idx) < 30 or len(test_idx) == 0:
    raise ValueError(f"训练/测试样本不足：train={len(train_idx)}, test={len(test_idx)}")

X = frame[factor_cols].to_numpy(dtype=float)
y = frame["TargetLogReturn"].to_numpy(dtype=float)
close = frame["Close"].to_numpy(dtype=float)
dates = frame["Date"].reset_index(drop=True)
target_dates = frame["TargetDate"].reset_index(drop=True)
X_train, y_train = X[train_idx], y[train_idx]
X_test = X[test_idx]
base_close = close[test_idx]
actual_close = frame["TargetClose"].to_numpy(dtype=float)[test_idx]
test_dates = target_dates.iloc[test_idx].reset_index(drop=True)

# Standardize the return target using training observations only.
y_mean = float(y_train.mean())
y_std = float(y_train.std() or 1.0)
y_train_scaled = (y_train - y_mean) / y_std

models = make_models()
fitted, predicted_close = {}, {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train_scaled)
    fitted[name] = estimator
    predicted_return = estimator.predict(X_test) * y_std + y_mean
    predicted_close[name] = base_close * np.exp(predicted_return)

# LSTM over the latest ten sessions of indicator data.
x_scaler = StandardScaler().fit(X_train)
X_scaled = x_scaler.transform(X)
sequences, labels, sequence_indices = [], [], []
for i in train_idx:
    start = i - LOOKBACK + 1
    if start >= 0:
        sequences.append(X_scaled[start:i + 1])
        labels.append((y[i] - y_mean) / y_std)
        sequence_indices.append(i)
if len(sequences) < 25:
    raise ValueError(f"LSTM 序列数量不足：{len(sequences)}")
sequences = np.asarray(sequences, dtype=np.float32)
labels = np.asarray(labels, dtype=np.float32)
val_split = max(1, int(len(sequences) * 0.8))
val_split = min(val_split, len(sequences) - 1)
lstm = make_lstm(X.shape[1])
lstm.fit(
    sequences[:val_split], labels[:val_split],
    validation_data=(sequences[val_split:], labels[val_split:]),
    epochs=350, batch_size=16, shuffle=False, verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True,
    )],
)

def predict_lstm_return(sequence):
    scaled = x_scaler.transform(np.asarray(sequence, dtype=float))
    scaled_return = float(lstm.predict(scaled[None, :, :], verbose=0).ravel()[0])
    return scaled_return * y_std + y_mean

lstm_returns = []
for i in test_idx:
    start = i - LOOKBACK + 1
    if start < 0:
        lstm_returns.append(y_mean)
    else:
        lstm_returns.append(predict_lstm_return(X[start:i + 1]))
predicted_close["LSTM"] = base_close * np.exp(np.asarray(lstm_returns))
predicted_close["RandomWalk"] = base_close.copy()
test_trend_up = frame["TrendUp"].to_numpy(dtype=bool)[test_idx]
predicted_close[STRATEGY_NAME] = np.where(
    test_trend_up, predicted_close["RF"], predicted_close["RandomWalk"]
)

# In-sample fitted predictions for the training interval.
train_base_close = close[train_idx]
train_actual_close = frame["TargetClose"].to_numpy(dtype=float)[train_idx]
train_dates = target_dates.iloc[train_idx].reset_index(drop=True)
train_predictions = pd.DataFrame({
    "Date": train_dates,
    "PreviousClose": train_base_close,
    "ActualClose": train_actual_close,
    "RandomWalk": train_base_close,
})
for name, estimator in fitted.items():
    train_return = estimator.predict(X_train) * y_std + y_mean
    train_predictions[name] = train_base_close * np.exp(train_return)

sequence_indices = np.asarray(sequence_indices, dtype=int)
sequence_returns = lstm.predict(sequences, batch_size=32, verbose=0).ravel() * y_std + y_mean
train_lstm_returns = pd.Series(sequence_returns, index=sequence_indices).reindex(train_idx).to_numpy()
train_predictions["LSTM"] = train_base_close * np.exp(train_lstm_returns)
train_trend_up = frame["TrendUp"].to_numpy(dtype=bool)[train_idx]
train_predictions[STRATEGY_NAME] = np.where(
    train_trend_up, train_predictions["RF"], train_predictions["RandomWalk"]
)

train_metric_rows = [{"Model": "RandomWalk", **score(train_actual_close, train_base_close, train_base_close)}]
for name in [*models.keys(), "LSTM"]:
    train_metric_rows.append({"Model": name, **score(train_actual_close, train_predictions[name], train_base_close)})
train_metric_rows.append({"Model": STRATEGY_NAME, **score(
    train_actual_close, train_predictions[STRATEGY_NAME], train_base_close
)})
train_metrics = pd.DataFrame(train_metric_rows).sort_values("MAE_points").reset_index(drop=True)

predictions = pd.DataFrame({
    "Date": test_dates,
    "PreviousClose": base_close,
    "ActualClose": actual_close,
})
for name, values in predicted_close.items():
    predictions[name] = values

metric_rows = [{"Model": "RandomWalk", **score(actual_close, base_close, base_close)}]
for name in [*models.keys(), "LSTM"]:
    metric_rows.append({"Model": name, **score(actual_close, predicted_close[name], base_close)})
metric_rows.append({"Model": STRATEGY_NAME, **score(
    actual_close, predicted_close[STRATEGY_NAME], base_close
)})
metrics = pd.DataFrame(metric_rows).sort_values("MAE_points").reset_index(drop=True)

# Compare prediction quality separately on actual up days and down days.
regime_metrics = []
up_mask = actual_close > base_close
down_mask = actual_close < base_close
for regime, mask in [("Up", up_mask), ("Down", down_mask)]:
    n = int(mask.sum())
    for name in [*models.keys(), "LSTM", "RandomWalk", STRATEGY_NAME]:
        actual_part = actual_close[mask]
        previous_part = base_close[mask]
        predicted_part = predictions[name].to_numpy(dtype=float)[mask]
        error = predicted_part - actual_part
        regime_metrics.append({
            "Regime": regime,
            "N": n,
            "Model": name,
            "MAE_points": float(np.mean(np.abs(error))) if n else np.nan,
            "RMSE_points": float(np.sqrt(np.mean(error ** 2))) if n else np.nan,
            "MAPE_percent": float(np.mean(np.abs(error / actual_part)) * 100) if n else np.nan,
            "Mean_error_points": float(np.mean(error)) if n else np.nan,
            "Directional_accuracy_percent": float(
                np.mean(np.sign(predicted_part - previous_part) ==
                         np.sign(actual_part - previous_part)) * 100
            ) if n else np.nan,
        })
regime_metrics = pd.DataFrame(regime_metrics)

# Forecast the requested target from the requested feature date. This requires
# the 2026-11-03 close and available indicators, but does not use later data.
feature_matches = np.flatnonzero(raw["Date"].eq(FORECAST_FEATURE_DATE).to_numpy())
if len(feature_matches) != 1:
    raise ValueError(
        f"数据中需要恰好一行 {FORECAST_FEATURE_DATE.date()} 的输入；"
        "请提供该日收盘价和技术指标。预测不需要 2026-11-04 至 11-11 的数据。"
    )
feature_pos = int(feature_matches[0])
forecast_factor_cols = [col for col in factor_cols if pd.notna(features.iloc[feature_pos][col])]
if not forecast_factor_cols:
    raise ValueError(f"{FORECAST_FEATURE_DATE.date()} 没有可用于预测的技术指标。")
forecast_feature_values = features[forecast_factor_cols].to_numpy(dtype=float)
forecast_rows = all_rows.loc[
    all_rows[forecast_factor_cols].notna().all(axis=1)
    & all_rows["TargetLogReturn"].notna()
    & all_rows["TargetDate"].le(FORECAST_FEATURE_DATE)
].copy()
forecast_train_raw_idx = forecast_rows.index.to_numpy(dtype=int)
if len(forecast_train_raw_idx) < 30:
    raise ValueError(f"截至 {FORECAST_FEATURE_DATE.date()} 的可用训练样本不足：{len(forecast_train_raw_idx)}")
forecast_X_train = forecast_feature_values[forecast_train_raw_idx]
forecast_y_train = forecast_rows["TargetLogReturn"].to_numpy(dtype=float)
forecast_y_mean = float(forecast_y_train.mean())
forecast_y_std = float(forecast_y_train.std() or 1.0)
forecast_y_scaled = (forecast_y_train - forecast_y_mean) / forecast_y_std
forecast_estimators = make_models()
for estimator in forecast_estimators.values():
    estimator.fit(forecast_X_train, forecast_y_scaled)

# Refit the LSTM on every labeled sample available by the forecast origin.
forecast_x_scaler = StandardScaler().fit(forecast_X_train)
forecast_X_scaled = forecast_x_scaler.transform(forecast_feature_values)
forecast_sequences, forecast_labels = [], []
for i, target in zip(forecast_train_raw_idx, forecast_y_train):
    start = i - LOOKBACK + 1
    if start >= 0:
        forecast_sequences.append(forecast_X_scaled[start:i + 1])
        forecast_labels.append((target - forecast_y_mean) / forecast_y_std)
if len(forecast_sequences) < 25:
    raise ValueError(f"最终 LSTM 序列数量不足：{len(forecast_sequences)}")
forecast_sequences = np.asarray(forecast_sequences, dtype=np.float32)
forecast_labels = np.asarray(forecast_labels, dtype=np.float32)
forecast_val_split = max(1, int(len(forecast_sequences) * 0.8))
forecast_val_split = min(forecast_val_split, len(forecast_sequences) - 1)
forecast_lstm = make_lstm(len(forecast_factor_cols))
forecast_lstm.fit(
    forecast_sequences[:forecast_val_split], forecast_labels[:forecast_val_split],
    validation_data=(forecast_sequences[forecast_val_split:], forecast_labels[forecast_val_split:]),
    epochs=350, batch_size=16, shuffle=False, verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True,
    )],
)

def predict_final_lstm_return(sequence):
    scaled = forecast_x_scaler.transform(np.asarray(sequence, dtype=float))
    scaled_return = float(forecast_lstm.predict(scaled[None, :, :], verbose=0).ravel()[0])
    return scaled_return * forecast_y_std + forecast_y_mean

forecast_features = features.iloc[[feature_pos]][forecast_factor_cols].to_numpy(dtype=float)
forecast_base_close = float(raw["Close"].iloc[feature_pos])
forecast_trend_up = bool(raw["TrendUp"].iloc[feature_pos])
if pd.isna(raw["TrendMA"].iloc[feature_pos]) or pd.isna(raw["TrendMAChange"].iloc[feature_pos]):
    raise ValueError(f"判断 {FORECAST_FEATURE_DATE.date()} 趋势至少需要 {TREND_MA_WINDOW + TREND_SLOPE_WINDOW} 个交易日的收盘价。")
forecast_sequence = features.iloc[feature_pos - LOOKBACK + 1:feature_pos + 1][forecast_factor_cols].to_numpy(dtype=float)
if len(forecast_sequence) != LOOKBACK or not np.isfinite(forecast_sequence).all():
    raise ValueError(f"预测 {FORECAST_FEATURE_DATE.date()} 时缺少完整的最近 {LOOKBACK} 个交易日指标。")
next_predictions = [{"FeatureDate": FORECAST_FEATURE_DATE,
                     "ForecastDate": FORECAST_TARGET_DATE,
                     "BaseClose": forecast_base_close,
                     "Model": "RandomWalk", "ForecastLogReturn": 0.0,
                     "ForecastReturnPercent": 0.0,
                     "ForecastClose": forecast_base_close}]
for name, estimator in forecast_estimators.items():
    pred_ret = float(estimator.predict(forecast_features)[0] * forecast_y_std + forecast_y_mean)
    next_predictions.append({"FeatureDate": FORECAST_FEATURE_DATE,
                             "ForecastDate": FORECAST_TARGET_DATE,
                             "BaseClose": forecast_base_close,
                             "Model": name,
                             "ForecastLogReturn": pred_ret,
                             "ForecastReturnPercent": (float(np.exp(pred_ret)) - 1) * 100,
                             "ForecastClose": forecast_base_close * float(np.exp(pred_ret))})
next_lstm_ret = predict_final_lstm_return(forecast_sequence)
next_predictions.append({"FeatureDate": FORECAST_FEATURE_DATE,
                         "ForecastDate": FORECAST_TARGET_DATE,
                         "BaseClose": forecast_base_close,
                         "Model": "LSTM",
                         "ForecastLogReturn": next_lstm_ret,
                         "ForecastReturnPercent": (float(np.exp(next_lstm_ret)) - 1) * 100,
                         "ForecastClose": forecast_base_close * float(np.exp(next_lstm_ret))})
strategy_source = "RF" if forecast_trend_up else "RandomWalk"
strategy_row = next(row for row in next_predictions if row["Model"] == strategy_source)
next_predictions.append({**strategy_row, "Model": STRATEGY_NAME,
                         "TrendRegime": "Uptrend" if forecast_trend_up else "Downtrend_or_Sideways",
                         "SelectedModel": strategy_source})
for row in next_predictions:
    row.setdefault("TrendRegime", "Uptrend" if forecast_trend_up else "Downtrend_or_Sideways")
    row.setdefault("SelectedModel", row["Model"])
forecast = pd.DataFrame(next_predictions)

predictions.to_csv(OUT / "model_predictions.csv", index=False,
                   encoding="utf-8-sig", date_format="%Y-%m-%d", float_format="%.6f")
train_predictions.to_csv(OUT / "model_train_predictions.csv", index=False,
                          encoding="utf-8-sig", date_format="%Y-%m-%d", float_format="%.6f")
train_metrics.to_csv(OUT / "model_train_metrics.csv", index=False,
                     encoding="utf-8-sig", float_format="%.6f")
metrics.to_csv(OUT / "model_metrics.csv", index=False,
               encoding="utf-8-sig", float_format="%.6f")
regime_metrics.to_csv(OUT / "up_down_regime_metrics.csv", index=False,
                      encoding="utf-8-sig", float_format="%.6f")
forecast.to_csv(OUT / "2026-11-11_forecast.csv", index=False,
                encoding="utf-8-sig", date_format="%Y-%m-%d", float_format="%.6f")

# Held-out actual and all seven model predictions on one shared chart.
fig, ax = plt.subplots(figsize=(15, 7.5))
palette = plt.get_cmap("tab10").colors
ax.plot(test_dates, actual_close, color="#111111", lw=2.4, zorder=5, label="Actual")
for color, name in zip(palette, [*models.keys(), "LSTM", "RandomWalk", STRATEGY_NAME]):
    row = metrics.loc[metrics["Model"].eq(name)].iloc[0]
    ax.plot(test_dates, predicted_close[name], color=color, lw=1.45,
            label=f"{name} (MAE {row.MAE_points:.1f})")
ax.set_title("CSI 300 test set: actual vs model predictions")
ax.set_xlabel("Target date")
ax.set_ylabel("CSI 300 index")
ax.grid(alpha=0.25)
ax.tick_params(axis="x", rotation=30)
ax.legend(ncol=2, fontsize=9, frameon=True)
fig.tight_layout()
fig.savefig(OUT / "model_predictions_up_down.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "model_predictions_all_models_up_down.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Training-set actual versus fitted values. This is an in-sample fit plot.
fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
axes = axes.ravel()
for ax, name in zip(axes, [*models.keys(), "LSTM", "RandomWalk", STRATEGY_NAME]):
    ax.plot(train_dates, train_actual_close, color="#222222", lw=1.7, label="Actual")
    ax.plot(train_dates, train_predictions[name], lw=1.25, label="Fitted prediction")
    row = train_metrics.loc[train_metrics["Model"].eq(name)].iloc[0]
    ax.set_title(f"{name} | Train MAE {row.MAE_points:.1f}, RMSE {row.RMSE_points:.1f}")
    ax.set_ylabel("CSI 300 index")
    ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
axes[-1].axis("off")
fig.suptitle("CSI 300: training-set actual closes vs in-sample fitted values", y=1.01, fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "model_train_predictions.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.5))
plot_metrics = metrics.sort_values("MAE_points")
ax.bar(plot_metrics["Model"], plot_metrics["MAE_points"], color="#3977b8")
ax.set_title("Held-out MAE comparison (lower is better)")
ax.set_ylabel("MAE (index points)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "model_metrics.png", dpi=180)
plt.close(fig)

regime_mae = regime_metrics.pivot(index="Model", columns="Regime", values="MAE_points")
regime_mae = regime_mae.reindex([*models.keys(), "LSTM", "RandomWalk", STRATEGY_NAME])
fig, ax = plt.subplots(figsize=(11, 6))
regime_mae.plot(kind="bar", ax=ax, color={"Up": "#3a9d5d", "Down": "#d85858"})
ax.set_title("Test MAE by actual market direction")
ax.set_ylabel("MAE (index points)")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Actual direction")
fig.tight_layout()
fig.savefig(OUT / "up_down_regime_mae.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Source: {SOURCE}")
print(f"Data: {dates.min().date()} to {dates.max().date()}, {len(frame)} usable rows")
print(f"Indicators: {len(factor_cols)}; train labels: {len(train_idx)}; test labels: {len(test_idx)}")
print(f"Test targets: {test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()}")
print("Test metrics (index points):")
print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("Up/down test metrics:")
print(regime_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("Training-set in-sample fit metrics (index points):")
print(train_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"Forecast origin: {FORECAST_FEATURE_DATE.date()} close {forecast_base_close:.2f}")
print(f"Trend filter: {'Uptrend' if forecast_trend_up else 'Downtrend_or_Sideways'}; selected model: {strategy_source}")
print(f"Forecast target: {FORECAST_TARGET_DATE.date()} ({HORIZON_SESSIONS} trading sessions after origin)")
print(forecast[["Model", "ForecastReturnPercent", "ForecastClose"]].to_string(
    index=False, float_format=lambda value: f"{value:.2f}"))
print(f"Outputs: {OUT}")
