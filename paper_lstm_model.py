from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Relative, regime-agnostic features aligned with the paper's final LSTM setup.
PAPER_SELECTED_FEATURES = [
    "time_to_expiry",
    "moneyness",
    "btc_return_1s",
    "ewma_rv_1s",
    "Up_price_momentum_15s",
    "sigma_annualized",
    "bsm_delta",
    "Up_OBI",
    "Up_spread",
    "Up_Mid_Price",
    "Down_Mid_Price",
    "Up_micro_price",
    "Net_OFI",
    "Net_Nominal_Flow",
    "Net_Flow_ema_15s",
    "Net_Flow_ema_60s",
    "CVD_Net_Flow",
    "Whale_Imbalance_ma_10s",
    "Price_Discovery_Spread",
    "Up_BUY_Urgency",
    "Down_BUY_Urgency",
    "BSM_Mispricing",
    "Micro_Price_Imbalance",
]


@dataclass(slots=True)
class PreparedPaperDataset:
    X: np.ndarray
    y: np.ndarray
    feature_columns: list[str]
    sequence_timestamps: pd.Series
    sequence_groups: pd.Series
    filtered_frame: pd.DataFrame


def load_feature_dataset(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    escaped = str(path).replace("'", "''")
    return duckdb.sql(f"SELECT * FROM read_parquet('{escaped}')").df()


def apply_paper_market_filters(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    group_col: str = "slug",
    spread_col: str = "Up_spread",
    up_mid_col: str = "Up_Mid_Price",
    down_mid_col: str = "Down_Mid_Price",
    global_warmup: str = "1h",
    contract_warmup_seconds: int = 60,
    max_spread: float = 0.03,
    min_mid_price: float = 0.02,
    max_mid_price: float = 0.98,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered[timestamp_col] = pd.to_datetime(filtered[timestamp_col], utc=True)
    filtered = filtered.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    global_cutoff = filtered[timestamp_col].min() + pd.Timedelta(global_warmup)
    filtered = filtered.loc[filtered[timestamp_col] >= global_cutoff].copy()

    filtered["_contract_age_seconds"] = (
        filtered[timestamp_col]
        - filtered.groupby(group_col)[timestamp_col].transform("min")
    ).dt.total_seconds()
    filtered = filtered.loc[
        filtered["_contract_age_seconds"] >= contract_warmup_seconds
    ].copy()

    if spread_col in filtered.columns:
        filtered = filtered.loc[filtered[spread_col] <= max_spread].copy()

    if up_mid_col in filtered.columns:
        filtered = filtered.loc[
            filtered[up_mid_col].between(min_mid_price, max_mid_price)
        ].copy()

    if down_mid_col in filtered.columns:
        filtered = filtered.loc[
            filtered[down_mid_col].between(min_mid_price, max_mid_price)
        ].copy()

    filtered = filtered.drop(columns="_contract_age_seconds")
    return filtered.reset_index(drop=True)


def make_five_second_mid_price_return_target(
    df: pd.DataFrame,
    *,
    group_col: str = "slug",
    mid_price_col: str = "Up_Mid_Price",
    horizon_seconds: int = 5,
    output_col: str = "paper_target_return",
) -> pd.DataFrame:
    if mid_price_col not in df.columns:
        raise KeyError(
            f"Cannot create paper target because '{mid_price_col}' is missing."
        )

    targeted = df.copy()
    future_mid = targeted.groupby(group_col)[mid_price_col].shift(-horizon_seconds)
    denom = targeted[mid_price_col].replace(0.0, np.nan)
    targeted[output_col] = (future_mid - targeted[mid_price_col]) / denom
    return targeted


def build_paper_lstm_regressor(
    n_features: int,
    *,
    sequence_length: int = 60,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
) -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the paper LSTM model. "
            "Install it with `pip install tensorflow` before training."
        ) from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, n_features)),
            keras.layers.LSTM(lstm_units, dropout=dropout),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


def prepare_paper_lstm_sequences(
    source: str | Path | pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
    timestamp_col: str = "timestamp",
    group_col: str = "slug",
    mid_price_col: str = "Up_Mid_Price",
    sequence_length: int = 60,
    horizon_seconds: int = 5,
    apply_filters: bool = True,
    filter_kwargs: dict[str, Any] | None = None,
) -> PreparedPaperDataset:
    df = load_feature_dataset(source)

    if apply_filters:
        df = apply_paper_market_filters(
            df,
            timestamp_col=timestamp_col,
            group_col=group_col,
            **(filter_kwargs or {}),
        )
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    target_name = target_column or "paper_target_return"
    if target_column is None:
        df = make_five_second_mid_price_return_target(
            df,
            group_col=group_col,
            mid_price_col=mid_price_col,
            horizon_seconds=horizon_seconds,
            output_col=target_name,
        )

    features = feature_columns or PAPER_SELECTED_FEATURES
    missing_features = [column for column in features if column not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")
    if target_name not in df.columns:
        raise KeyError(f"Missing target column: '{target_name}'")

    model_frame = df[[timestamp_col, group_col, *features, target_name]].copy()
    model_frame = model_frame.replace([np.inf, -np.inf], np.nan).dropna()
    model_frame = model_frame.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    seq_timestamps: list[pd.Timestamp] = []
    seq_groups: list[str] = []

    for group_value, group_frame in model_frame.groupby(group_col, sort=False):
        feature_matrix = group_frame[features].to_numpy(dtype=np.float32)
        target_vector = group_frame[target_name].to_numpy(dtype=np.float32)
        time_vector = group_frame[timestamp_col].reset_index(drop=True)

        if len(group_frame) < sequence_length:
            continue

        for end_idx in range(sequence_length - 1, len(group_frame)):
            start_idx = end_idx - sequence_length + 1
            sequences.append(feature_matrix[start_idx : end_idx + 1])
            targets.append(target_vector[end_idx])
            seq_timestamps.append(time_vector.iloc[end_idx])
            seq_groups.append(group_value)

    if not sequences:
        raise ValueError("No LSTM sequences were created. Check filters and sequence length.")

    return PreparedPaperDataset(
        X=np.asarray(sequences, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        feature_columns=list(features),
        sequence_timestamps=pd.Series(seq_timestamps, name=timestamp_col),
        sequence_groups=pd.Series(seq_groups, name=group_col),
        filtered_frame=model_frame,
    )


def train_paper_final_lstm_model(
    source: str | Path | pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
    timestamp_col: str = "timestamp",
    group_col: str = "slug",
    mid_price_col: str = "Up_Mid_Price",
    sequence_length: int = 60,
    horizon_seconds: int = 5,
    test_size: float = 0.20,
    validation_split: float = 0.10,
    epochs: int = 50,
    batch_size: int = 256,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    verbose: int = 1,
    filter_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Recreate the paper's final LSTM training pipeline.

    Paper-aligned pieces:
    - 60-step rolling window over high-frequency market states.
    - Relative, regime-agnostic default features.
    - Warm-up, liquidity, and extreme-probability filters.
    - Regression target defined as the 5-second forward mid-price return.

    The paper does not publish exact hidden-layer sizes, so this function uses a
    compact single-layer LSTM regressor by default and exposes all sizing knobs.
    """

    prepared = prepare_paper_lstm_sequences(
        source,
        feature_columns=feature_columns,
        target_column=target_column,
        timestamp_col=timestamp_col,
        group_col=group_col,
        mid_price_col=mid_price_col,
        sequence_length=sequence_length,
        horizon_seconds=horizon_seconds,
        filter_kwargs=filter_kwargs,
    )

    split_index = int(len(prepared.X) * (1.0 - test_size))
    if split_index <= 0 or split_index >= len(prepared.X):
        raise ValueError("`test_size` leaves no samples in train or test.")

    X_train = prepared.X[:split_index].copy()
    X_test = prepared.X[split_index:].copy()
    y_train = prepared.y[:split_index].copy()
    y_test = prepared.y[split_index:].copy()

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_train = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

    model = build_paper_lstm_regressor(
        n_features=len(prepared.feature_columns),
        sequence_length=sequence_length,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = []
    try:
        from tensorflow import keras

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]
    except ImportError:
        # build_paper_lstm_regressor already raises a helpful error if TensorFlow
        # is unavailable. This keeps type-checkers and linters happy.
        pass

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )

    evaluation = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    return {
        "model": model,
        "history": history.history,
        "evaluation": evaluation,
        "scaler": scaler,
        "feature_columns": prepared.feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_timestamps": prepared.sequence_timestamps.iloc[:split_index].reset_index(drop=True),
        "test_timestamps": prepared.sequence_timestamps.iloc[split_index:].reset_index(drop=True),
        "train_groups": prepared.sequence_groups.iloc[:split_index].reset_index(drop=True),
        "test_groups": prepared.sequence_groups.iloc[split_index:].reset_index(drop=True),
    }


def _format_terminal_summary(
    prepared: PreparedPaperDataset,
    *,
    sequence_length: int,
    horizon_seconds: int,
) -> str:
    timestamps = prepared.sequence_timestamps
    groups = prepared.sequence_groups
    lines = [
        "Paper LSTM data pipeline summary",
        f"rows after filtering: {len(prepared.filtered_frame):,}",
        f"sequence count: {len(prepared.X):,}",
        f"sequence shape: {prepared.X.shape}",
        f"target shape: {prepared.y.shape}",
        f"feature count: {len(prepared.feature_columns)}",
        f"window length: {sequence_length}s",
        f"prediction horizon: {horizon_seconds}s",
        f"time range: {timestamps.min()} -> {timestamps.max()}",
        f"distinct contracts: {groups.nunique()}",
    ]
    return "\n".join(lines)


def print_paper_model_outcome_to_terminal(
    source: str | Path | pd.DataFrame,
    *,
    train_model: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    sequence_length = kwargs.get("sequence_length", 60)
    horizon_seconds = kwargs.get("horizon_seconds", 5)

    if train_model:
        artifacts = train_paper_final_lstm_model(source, **kwargs)
        print("Paper LSTM training outcome")
        for key, value in artifacts["evaluation"].items():
            print(f"{key}: {value}")
        return artifacts

    prepared = prepare_paper_lstm_sequences(
        source,
        feature_columns=kwargs.get("feature_columns"),
        target_column=kwargs.get("target_column"),
        timestamp_col=kwargs.get("timestamp_col", "timestamp"),
        group_col=kwargs.get("group_col", "slug"),
        mid_price_col=kwargs.get("mid_price_col", "Up_Mid_Price"),
        sequence_length=sequence_length,
        horizon_seconds=horizon_seconds,
        apply_filters=kwargs.get("apply_filters", True),
        filter_kwargs=kwargs.get("filter_kwargs"),
    )
    print(
        _format_terminal_summary(
            prepared,
            sequence_length=sequence_length,
            horizon_seconds=horizon_seconds,
        )
    )
    return {
        "prepared": prepared,
        "summary": _format_terminal_summary(
            prepared,
            sequence_length=sequence_length,
            horizon_seconds=horizon_seconds,
        ),
    }


if __name__ == "__main__":
    default_dataset = Path(__file__).with_name("FINAL_FEATURE_DATASET.parquet")
    if not default_dataset.exists():
        raise FileNotFoundError(f"Default dataset not found: {default_dataset}")
    print_paper_model_outcome_to_terminal(default_dataset)
