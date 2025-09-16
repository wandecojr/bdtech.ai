"""Forecast pipeline tuned for lower WMAPE scores.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    data_dir: Path
    output_csv: Path
    model_path: Path | None
    validation_weeks: int = 5
    forecast_weeks: int = 5
    random_state: int = 42


LAG_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 8, 12)
ROLLING_WINDOWS: Tuple[int, ...] = (2, 4, 8, 12)


def load_transactions(data_dir: Path) -> pd.DataFrame:
    """Load raw transactions keeping only the required columns."""
    path = data_dir / "transacoes_2022.parquet"
    columns = [
        "internal_store_id",
        "internal_product_id",
        "transaction_date",
        "quantity",
    ]
    df = pd.read_parquet(path, columns=columns)

    df = df.rename(
        columns={
            "internal_store_id": "pdv",
            "internal_product_id": "produto",
            "transaction_date": "data",
            "quantity": "quantidade",
        }
    )

    df["data"] = pd.to_datetime(df["data"], utc=True).dt.tz_convert(None)
    df["quantidade"] = df["quantidade"].astype(np.float32)

    # Store identifiers as int64 to reduce memory usage. Any parsing
    # issues will raise and make debugging easier than silent coercion.
    df["pdv"] = df["pdv"].astype("int64")
    df["produto"] = df["produto"].astype("int64")

    return df


def build_weekly_history(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into ISO weeks for each PDV/SKU pair."""
    iso = transactions["data"].dt.isocalendar()
    transactions["iso_year"] = iso["year"].astype(np.int16)
    transactions["iso_week"] = iso["week"].astype(np.int16)

    # Monday of the respective ISO week gives a stable chronological key.
    week_start = pd.to_datetime(
        transactions["iso_year"].astype(str)
        + transactions["iso_week"].astype(str)
        + "1",
        format="%G%V%u",
    )
    transactions["week_start"] = week_start

    weekly = (
        transactions.groupby(
            ["pdv", "produto", "iso_year", "iso_week", "week_start"],
            as_index=False,
        )["quantidade"].sum()
    )

    weekly["quantidade"] = weekly["quantidade"].astype(np.float32)
    weekly["iso_year"] = weekly["iso_year"].astype(np.int16)
    weekly["iso_week"] = weekly["iso_week"].astype(np.int16)

    # Additional calendar helpers used later as model features.
    weekly["month"] = weekly["week_start"].dt.month.astype(np.int8)
    weekly["quarter"] = weekly["week_start"].dt.quarter.astype(np.int8)

    return weekly


def add_sequential_features(base: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and cumulative statistics for each pair."""
    df = base.sort_values(["pdv", "week_start", "produto"]).reset_index(drop=True)

    group_pair = df.groupby(["pdv", "produto"], sort=False)

    for lag in LAG_VALUES:
        df[f"lag_{lag}"] = group_pair["quantidade"].shift(lag)

    shifted = group_pair["quantidade"].shift(1)
    for window in ROLLING_WINDOWS:
        rolling_view = shifted.rolling(window, min_periods=1)
        df[f"roll_mean_{window}"] = rolling_view.mean()
        df[f"roll_std_{window}"] = rolling_view.std()
        df[f"roll_sum_{window}"] = rolling_view.sum()
        df[f"roll_median_{window}"] = rolling_view.median()

    df["weeks_since_prev_obs"] = (
        group_pair["week_start"].diff().dt.days.div(7).fillna(-1)
    )

    # Pair cumulative averages excluding the current observation.
    pair_counts = group_pair.cumcount()
    pair_cumsum = group_pair["quantidade"].cumsum() - df["quantidade"]
    df["pair_avg_prior"] = pair_cumsum / pair_counts.replace(0, np.nan)

    # Store-level cumulative average.
    store_group = df.groupby("pdv", sort=False)["quantidade"]
    store_counts = store_group.cumcount()
    store_cumsum = store_group.cumsum() - df["quantidade"]
    df["store_avg_prior"] = store_cumsum / store_counts.replace(0, np.nan)

    # Product-level cumulative average.
    prod_group = df.groupby("produto", sort=False)["quantidade"]
    prod_counts = prod_group.cumcount()
    prod_cumsum = prod_group.cumsum() - df["quantidade"]
    df["product_avg_prior"] = prod_cumsum / prod_counts.replace(0, np.nan)

    df = df.drop(
        columns=["iso_year", "iso_week"], errors="ignore"
    )  # Not needed after feature creation.

    return df


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare final feature matrix for model training."""
    df = add_sequential_features(df)

    df["store_id"] = df["pdv"].astype("category").cat.codes.astype(np.int32)
    df["product_id"] = df["produto"].astype("category").cat.codes.astype(np.int32)

    feature_columns = [
        "store_id",
        "product_id",
        "month",
        "quarter",
        "weeks_since_prev_obs",
        "pair_avg_prior",
        "store_avg_prior",
        "product_avg_prior",
    ]

    feature_columns += [f"lag_{lag}" for lag in LAG_VALUES]
    for window in ROLLING_WINDOWS:
        feature_columns.append(f"roll_mean_{window}")
        feature_columns.append(f"roll_std_{window}")
        feature_columns.append(f"roll_sum_{window}")
        feature_columns.append(f"roll_median_{window}")

    df[feature_columns] = df[feature_columns].fillna(0.0)

    return df, feature_columns


def split_train_validation(
    df: pd.DataFrame, validation_weeks: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `validation_weeks` ISO weeks for validation."""
    unique_weeks = df["week_start"].drop_duplicates().sort_values()
    val_weeks = unique_weeks.iloc[-validation_weeks:]

    train_df = df[~df["week_start"].isin(val_weeks)].copy()
    valid_df = df[df["week_start"].isin(val_weeks)].copy()

    return train_df, valid_df


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error."""
    numerator = np.abs(y_true - y_pred).sum()
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("inf")
    return numerator / denominator


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: Iterable[str],
    random_state: int,
) -> lgb.LGBMRegressor:
    train_x = train_df[feature_columns]
    train_y = train_df["quantidade"]

    valid_x = valid_df[feature_columns]
    valid_y = valid_df["quantidade"]

    model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        objective="quantile",
        alpha=0.5,
        metric="quantile",
        n_estimators=1600,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=50,
        min_child_weight=1.0,
        reg_alpha=0.3,
        reg_lambda=20.0,
        min_split_gain=0.1,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
        print(f"Best iteration: {model.best_iteration_}")

    return model


def forecast_january(
    base_history: pd.DataFrame,
    feature_columns: List[str],
    model: lgb.LGBMRegressor,
    forecast_weeks: int,
) -> pd.DataFrame:
    """Generate forecasts for the first `forecast_weeks` of January 2023."""
    history = base_history.copy()
    history["forecast_round"] = np.int16(0)

    all_pairs = history[["pdv", "produto"]].drop_duplicates()
    last_week = history["week_start"].max()

    predictions: List[pd.DataFrame] = []

    for step in range(1, forecast_weeks + 1):
        next_week = last_week + pd.Timedelta(weeks=step)
        iso = next_week.isocalendar()

        new_rows = all_pairs.copy()
        new_rows["week_start"] = next_week
        new_rows["iso_year"] = np.int16(iso.year)
        new_rows["iso_week"] = np.int16(iso.week)
        new_rows["month"] = np.int8(next_week.month)
        new_rows["quarter"] = np.int8(next_week.quarter)
        new_rows["quantidade"] = np.float32(np.nan)
        new_rows["forecast_round"] = np.int16(step)

        history = pd.concat([history, new_rows], ignore_index=True)

        engineered = add_sequential_features(history)
        engineered["store_id"] = engineered["pdv"].astype("category").cat.codes.astype(
            np.int32
        )
        engineered["product_id"] = (
            engineered["produto"].astype("category").cat.codes.astype(np.int32)
        )
        engineered[feature_columns] = engineered[feature_columns].fillna(0.0)

        mask = (engineered["forecast_round"] == step) & engineered["quantidade"].isna()
        step_df = engineered.loc[mask].copy()

        preds = model.predict(step_df[feature_columns])
        preds = preds.astype(np.float32)
        step_df["quantidade"] = preds
        step_df["ano"] = step_df["week_start"].dt.isocalendar()["year"].astype(int)
        step_df["semana"] = step

        predictions.append(step_df[["semana", "pdv", "produto", "quantidade"]])

        history.loc[history["forecast_round"] == step, "quantidade"] = preds

    result = pd.concat(predictions, ignore_index=True)
    result["quantidade"] = result["quantidade"].clip(lower=0).round().astype(int)
    result = result.sort_values(["semana", "pdv", "produto"]).reset_index(drop=True)

    return result


def run_pipeline(config: PipelineConfig) -> None:
    transactions = load_transactions(config.data_dir)
    weekly = build_weekly_history(transactions)

    train_data, feature_columns = prepare_training_data(weekly)
    train_df, valid_df = split_train_validation(train_data, config.validation_weeks)

    if valid_df.empty:
        raise RuntimeError("Validation set is empty; check the number of validation weeks.")

    model = train_model(train_df, valid_df, feature_columns, config.random_state)

    val_pred = model.predict(valid_df[feature_columns])
    score = wmape(valid_df["quantidade"].to_numpy(), val_pred)
    print(f"Validation WMAPE (ratio): {score:.6f}")

    predictions = forecast_january(weekly, feature_columns, model, config.forecast_weeks)

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.output_csv, sep=";", index=False, encoding="utf-8")

    print(f"Saved forecast to {config.output_csv}")

    if config.model_path is not None:
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(config.model_path))
        print(f"Saved LightGBM model to {config.model_path}")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Weekly PDV/SKU forecast pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/train/parquet"),
        help="Directory containing pdvs.parquet, produtos.parquet, and transacoes_2022.parquet",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/jan_2023_forecast.csv"),
        help="Path to the forecast CSV file (semicolon separated)",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to persist the trained LightGBM model",
    )
    parser.add_argument(
        "--validation-weeks",
        type=int,
        default=5,
        help="Number of ISO weeks reserved for validation",
    )
    parser.add_argument(
        "--forecast-weeks",
        type=int,
        default=5,
        help="Number of consecutive weeks to forecast",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LightGBM",
    )

    args = parser.parse_args()

    return PipelineConfig(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        model_path=args.save_model,
        validation_weeks=args.validation_weeks,
        forecast_weeks=args.forecast_weeks,
        random_state=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    run_pipeline(config)
