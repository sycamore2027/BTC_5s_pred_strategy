from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from paper_lstm_model import (
    PAPER_SELECTED_FEATURES,
    apply_paper_market_filters,
    load_feature_dataset,
    make_five_second_mid_price_return_target,
)


DATASET_PATH = Path(__file__).with_name("FINAL_FEATURE_DATASET.parquet")
SITE_DATA_DIR = Path(__file__).with_name("site_data")

INITIAL_CAPITAL = 10_000.0
POSITION_NOTIONAL = 100.0
FEE_RATE = 0.072

THETA_GRID = [0.020, 0.025, 0.030, 0.035, 0.040]
PROFIT_TARGET_GRID = [0.030, 0.040, 0.050]
STOP_LOSS_GRID = [0.015, 0.020]
TIMEOUT_GRID = [30, 45, 60]


@dataclass(slots=True)
class StrategyConfig:
    theta: float
    profit_take: float
    stop_loss: float
    timeout_seconds: int
    position_notional: float = POSITION_NOTIONAL
    fee_rate: float = FEE_RATE
    initial_capital: float = INITIAL_CAPITAL


def load_strategy_frame(source: str | Path = DATASET_PATH) -> pd.DataFrame:
    frame = load_feature_dataset(source)
    frame = apply_paper_market_filters(frame)
    frame = make_five_second_mid_price_return_target(frame)

    columns = []
    for column in [
        "timestamp",
        "slug",
        "Up_Mid_Price",
        "Down_Mid_Price",
        "paper_target_return",
        "target_logic",
        *PAPER_SELECTED_FEATURES,
    ]:
        if column not in columns:
            columns.append(column)

    frame = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["timestamp", "slug"]).reset_index(drop=True)
    return frame


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * 0.60)
    val_end = int(len(frame) * 0.80)
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def train_signal_model(train_frame: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=300,
        learning_rate=0.03,
        min_samples_leaf=100,
        random_state=42,
    )
    model.fit(train_frame[PAPER_SELECTED_FEATURES], train_frame["paper_target_return"])
    return model


def attach_predictions(
    frame: pd.DataFrame,
    model: HistGradientBoostingRegressor,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["pred_return"] = model.predict(enriched[PAPER_SELECTED_FEATURES])
    enriched["pred_up_price"] = np.clip(
        enriched["Up_Mid_Price"].to_numpy() * (1.0 + enriched["pred_return"].to_numpy()),
        0.01,
        0.99,
    )
    enriched["pred_down_price"] = np.clip(1.0 - enriched["pred_up_price"], 0.01, 0.99)
    enriched["alpha"] = enriched["pred_up_price"] - enriched["Up_Mid_Price"]
    return enriched


def build_contract_views(frame: pd.DataFrame) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for slug, group in frame.groupby("slug", sort=False):
        contracts.append(
            {
                "slug": slug,
                "timestamps": group["timestamp"].to_numpy(),
                "alpha": group["alpha"].to_numpy(dtype=np.float64),
                "up": group["Up_Mid_Price"].to_numpy(dtype=np.float64),
                "down": group["Down_Mid_Price"].to_numpy(dtype=np.float64),
                "pred_up": group["pred_up_price"].to_numpy(dtype=np.float64),
                "pred_down": group["pred_down_price"].to_numpy(dtype=np.float64),
            }
        )
    return contracts


def round_trip_fee(price: float, quantity: float, fee_rate: float) -> float:
    return price * (1.0 - price) * quantity * fee_rate


def run_backtest(
    contract_views: list[dict[str, Any]],
    config: StrategyConfig,
) -> pd.DataFrame:
    trade_rows: list[dict[str, Any]] = []
    trade_id = 1

    for contract in contract_views:
        timestamps = contract["timestamps"]
        alpha = contract["alpha"]
        up = contract["up"]
        down = contract["down"]
        pred_up = contract["pred_up"]
        pred_down = contract["pred_down"]
        slug = contract["slug"]

        i = 0
        while i < len(alpha):
            mispricing = float(alpha[i])

            if mispricing > config.theta:
                direction = "UP"
                entry_price = float(up[i])
                price_path = up
                model_price = float(pred_up[i])
            elif mispricing < -config.theta:
                direction = "DOWN"
                entry_price = float(down[i])
                price_path = down
                model_price = float(pred_down[i])
            else:
                i += 1
                continue

            quantity = config.position_notional / max(entry_price, 1e-9)
            entry_fee = round_trip_fee(entry_price, quantity, config.fee_rate)
            max_exit_idx = min(i + config.timeout_seconds, len(alpha) - 1)
            exit_idx = max_exit_idx
            exit_type = "timeout"
            exit_price = float(price_path[exit_idx])

            for j in range(i + 1, max_exit_idx + 1):
                current_price = float(price_path[j])
                gross_return = (current_price - entry_price) / entry_price

                if gross_return >= config.profit_take:
                    exit_idx = j
                    exit_price = current_price
                    exit_type = "success"
                    break

                if gross_return <= -config.stop_loss:
                    exit_idx = j
                    exit_price = current_price
                    exit_type = "stop_loss"
                    break

            exit_fee = round_trip_fee(exit_price, quantity, config.fee_rate)
            gross_pnl = quantity * (exit_price - entry_price)
            net_pnl = gross_pnl - entry_fee - exit_fee

            trade_rows.append(
                {
                    "trade_id": trade_id,
                    "slug": slug,
                    "entry_time": pd.Timestamp(timestamps[i]).isoformat(),
                    "exit_time": pd.Timestamp(timestamps[exit_idx]).isoformat(),
                    "direction": direction,
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "position_notional": round(config.position_notional, 2),
                    "quantity": round(quantity, 6),
                    "p_model_up": round(float(pred_up[i]), 6),
                    "p_market_up": round(float(up[i]), 6),
                    "alpha": round(mispricing, 6),
                    "gross_pnl": round(gross_pnl, 6),
                    "fees_paid": round(entry_fee + exit_fee, 6),
                    "net_pnl": round(net_pnl, 6),
                    "gross_return_pct": round(((exit_price - entry_price) / entry_price) * 100.0, 4),
                    "net_return_pct": round((net_pnl / config.position_notional) * 100.0, 4),
                    "hold_seconds": int(exit_idx - i),
                    "exit_type": exit_type,
                }
            )
            trade_id += 1
            i = exit_idx + 1

    return pd.DataFrame(trade_rows)


def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float((returns.mean() / std) * np.sqrt(len(returns)))


def build_ledger(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    ledger = trades.sort_values("exit_time").reset_index(drop=True).copy()
    ledger["timestamp"] = ledger["exit_time"]
    ledger["capital_before"] = initial_capital + ledger["net_pnl"].cumsum().shift(fill_value=0.0)
    ledger["capital_after"] = initial_capital + ledger["net_pnl"].cumsum()
    ledger["cumulative_pnl"] = ledger["net_pnl"].cumsum()
    ledger["drawdown"] = ledger["capital_after"] - ledger["capital_after"].cummax()
    ledger["drawdown_pct"] = ledger["drawdown"] / ledger["capital_after"].cummax()
    ledger["trade_return"] = ledger["net_pnl"] / ledger["position_notional"]
    ledger["rolling_avg_return_30"] = ledger["trade_return"].rolling(30).mean()
    ledger["rolling_win_rate_30"] = (ledger["net_pnl"] > 0).astype(float).rolling(30).mean()
    ledger["rolling_sharpe_30"] = ledger["trade_return"].rolling(30).apply(
        lambda x: 0.0 if np.std(x, ddof=1) == 0 else (np.mean(x) / np.std(x, ddof=1)) * np.sqrt(len(x)),
        raw=True,
    )
    return ledger


def summarize_performance(
    trades: pd.DataFrame,
    ledger: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, Any]:
    returns = trades["net_pnl"] / trades["position_notional"]
    gross_returns = trades["gross_pnl"] / trades["position_notional"]

    summary = {
        "trade_count": int(len(trades)),
        "gross_pnl_sum": round(float(trades["gross_pnl"].sum()), 4),
        "net_pnl_sum": round(float(trades["net_pnl"].sum()), 4),
        "fees_paid_sum": round(float(trades["fees_paid"].sum()), 4),
        "expected_return_per_trade": round(float(returns.mean()), 6),
        "expected_gross_return_per_trade": round(float(gross_returns.mean()), 6),
        "sharpe_ratio": round(sharpe_ratio(returns), 4),
        "win_rate": round(float((trades["net_pnl"] > 0).mean()), 4),
        "success_rate": round(float((trades["exit_type"] == "success").mean()), 4),
        "stop_loss_rate": round(float((trades["exit_type"] == "stop_loss").mean()), 4),
        "timeout_rate": round(float((trades["exit_type"] == "timeout").mean()), 4),
        "average_trade_duration_seconds": round(float(trades["hold_seconds"].mean()), 2),
        "max_drawdown": round(float(ledger["drawdown"].min()), 4),
        "max_drawdown_pct": round(float(ledger["drawdown_pct"].min()), 6),
        "turnover": round(float(trades["position_notional"].sum() / config.initial_capital), 4),
        "final_capital": round(float(ledger["capital_after"].iloc[-1]), 4) if len(ledger) else config.initial_capital,
    }
    return summary


def rolling_floor(series: pd.Series, percentile: float, default: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return default
    return float(np.percentile(clean, percentile))


def build_monitoring_summary(validation_ledger: pd.DataFrame, test_ledger: pd.DataFrame) -> dict[str, Any]:
    validation_sharpe_floor = min(0.0, rolling_floor(validation_ledger["rolling_sharpe_30"], 10, 0.0))
    validation_return_floor = rolling_floor(validation_ledger["rolling_avg_return_30"], 10, 0.0)
    validation_win_floor = rolling_floor(validation_ledger["rolling_win_rate_30"], 10, 0.0)

    test_last_sharpe = float(test_ledger["rolling_sharpe_30"].dropna().iloc[-1]) if test_ledger["rolling_sharpe_30"].dropna().size else 0.0
    test_last_return = float(test_ledger["rolling_avg_return_30"].dropna().iloc[-1]) if test_ledger["rolling_avg_return_30"].dropna().size else 0.0
    test_last_win = float(test_ledger["rolling_win_rate_30"].dropna().iloc[-1]) if test_ledger["rolling_win_rate_30"].dropna().size else 0.0

    return {
        "benchmark_window_trades": 30,
        "validation_sharpe_floor": round(validation_sharpe_floor, 4),
        "validation_expected_return_floor": round(validation_return_floor, 6),
        "validation_win_rate_floor": round(validation_win_floor, 4),
        "latest_test_rolling_sharpe": round(test_last_sharpe, 4),
        "latest_test_rolling_expected_return": round(test_last_return, 6),
        "latest_test_rolling_win_rate": round(test_last_win, 4),
        "is_on_track": bool(
            test_last_sharpe >= validation_sharpe_floor
            and test_last_return >= validation_return_floor
            and test_last_win >= validation_win_floor
        ),
    }


def chart_points(frame: pd.DataFrame, x_col: str, y_col: str) -> list[dict[str, Any]]:
    points = frame[[x_col, y_col]].dropna().copy()
    points[x_col] = points[x_col].astype(str)
    return points.rename(columns={x_col: "x", y_col: "y"}).to_dict(orient="records")


def preview_rows(frame: pd.DataFrame, count: int = 20) -> list[dict[str, Any]]:
    return frame.head(count).to_dict(orient="records")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def choose_best_config(validation_contracts: list[dict[str, Any]]) -> tuple[StrategyConfig, pd.DataFrame]:
    best_config: StrategyConfig | None = None
    best_trades: pd.DataFrame | None = None
    best_score = (-np.inf, -np.inf)

    for theta in THETA_GRID:
        for profit_take in PROFIT_TARGET_GRID:
            for stop_loss in STOP_LOSS_GRID:
                for timeout_seconds in TIMEOUT_GRID:
                    config = StrategyConfig(
                        theta=theta,
                        profit_take=profit_take,
                        stop_loss=stop_loss,
                        timeout_seconds=timeout_seconds,
                    )
                    trades = run_backtest(validation_contracts, config)
                    if trades.empty:
                        continue
                    score = (float(trades["net_pnl"].sum()), float(trades["net_pnl"].mean()))
                    if score > best_score:
                        best_score = score
                        best_config = config
                        best_trades = trades

    if best_config is None or best_trades is None:
        raise RuntimeError("Parameter search produced no trades.")
    return best_config, best_trades


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def generate_site_payload() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = load_strategy_frame(DATASET_PATH)
    train_frame, validation_frame, test_frame = split_frame(frame)

    model = train_signal_model(train_frame)
    validation_frame = attach_predictions(validation_frame, model)
    test_frame = attach_predictions(test_frame, model)

    validation_contracts = build_contract_views(validation_frame)
    test_contracts = build_contract_views(test_frame)

    chosen_config, validation_trades = choose_best_config(validation_contracts)
    validation_ledger = build_ledger(validation_trades, chosen_config.initial_capital)
    validation_summary = summarize_performance(validation_trades, validation_ledger, chosen_config)

    test_trades = run_backtest(test_contracts, chosen_config)
    test_ledger = build_ledger(test_trades, chosen_config.initial_capital)
    test_summary = summarize_performance(test_trades, test_ledger, chosen_config)
    monitoring_summary = build_monitoring_summary(validation_ledger, test_ledger)

    payload = {
        "strategy_name": "BTC 5s Alpha Mispricing Strategy",
        "model_name": "Gradient-Boosted Fair-Value Proxy",
        "feature_count": len(PAPER_SELECTED_FEATURES),
        "features": PAPER_SELECTED_FEATURES,
        "config": asdict(chosen_config),
        "data_overview": {
            "rows_after_filtering": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int(len(test_frame)),
            "contract_count": int(frame["slug"].nunique()),
            "time_range_start": str(frame["timestamp"].min()),
            "time_range_end": str(frame["timestamp"].max()),
        },
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "monitoring": monitoring_summary,
        "charts": {
            "capital_curve": chart_points(test_ledger, "timestamp", "capital_after"),
            "cumulative_pnl": chart_points(test_ledger, "timestamp", "cumulative_pnl"),
            "rolling_sharpe": chart_points(test_ledger, "timestamp", "rolling_sharpe_30"),
            "rolling_avg_return": chart_points(test_ledger, "timestamp", "rolling_avg_return_30"),
        },
        "blotter_preview": preview_rows(test_trades),
        "ledger_preview": preview_rows(test_ledger),
    }
    return payload, test_trades, test_ledger


def write_site_artifacts() -> dict[str, Any]:
    SITE_DATA_DIR.mkdir(exist_ok=True)
    payload, blotter, ledger = generate_site_payload()

    safe_payload = json_safe(payload)
    (SITE_DATA_DIR / "summary.json").write_text(
        json.dumps(safe_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(blotter, SITE_DATA_DIR / "blotter.csv")
    write_csv(ledger, SITE_DATA_DIR / "ledger.csv")
    return safe_payload


def main() -> None:
    payload = write_site_artifacts()
    print("Strategy backtest complete")
    print(f"Selected theta: {payload['config']['theta']}")
    print(f"Profit target: {payload['config']['profit_take']}")
    print(f"Stop loss: {payload['config']['stop_loss']}")
    print(f"Timeout: {payload['config']['timeout_seconds']}s")
    print(f"Test trades: {payload['test_summary']['trade_count']}")
    print(f"Net PnL: {payload['test_summary']['net_pnl_sum']}")
    print(f"Sharpe ratio: {payload['test_summary']['sharpe_ratio']}")
    print(f"Win rate: {payload['test_summary']['win_rate']}")


if __name__ == "__main__":
    main()
