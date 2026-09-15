"""Leakage-controlled core for the planned-versus-actual route experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import kendalltau


RANDOM_SEED = 20260822
COVERAGE_TARGETS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
WIDTH_CAPS = (60.0, 75.0, 90.0, 120.0, 180.0, 210.0, 225.0, 240.0, 300.0, 360.0)
PRIMARY_WIDTHS = (120.0, 210.0, 240.0)
MIN_INTERVAL_WIDTH = 30.0
MIN_CALIBRATION_GROUP = 200
HORIZON_EDGES = (-np.inf, 180.0, 360.0, 540.0, 720.0, np.inf)
HORIZON_LABELS = ("H1", "H2", "H3", "H4", "H5")

BASE_FEATURES = [
    "Country",
    "earliest_rel",
    "latest_rel",
    "time_window_width",
    "time_window_midpoint",
    "Depot",
    "Delivery",
    "route_stop_count",
    "route_delivery_count",
    "route_pickup_count",
    "route_depot_count",
    "driver_hist_target_mean",
    "address_hist_target_mean",
    "driver_hist_count_log",
    "address_hist_count_log",
]

GENERATOR_FEATURES = BASE_FEATURES + [
    "driver_hist_plan_rank_mean",
    "address_hist_plan_rank_mean",
]

POSITION_FEATURES = [
    "seq_rank",
    "seq_rank_norm",
    "seq_remaining_stops",
]

ADJACENCY_FEATURES = [
    "seq_prev_time_window_midpoint",
    "seq_next_time_window_midpoint",
    "seq_prev_delivery",
    "seq_next_delivery",
    "seq_prev_depot",
    "seq_next_depot",
    "seq_prev_address_hist_target_mean",
    "seq_next_address_hist_target_mean",
]

ORDER_FEATURES = POSITION_FEATURES + ADJACENCY_FEATURES

DISTANCE_FEATURES = [
    "seq_edge_distance",
    "seq_cumulative_distance",
    "seq_remaining_distance",
    "seq_total_distance",
    "seq_distance_available",
]

SEQUENCE_FEATURES = ORDER_FEATURES + DISTANCE_FEATURES

SEQUENCE_REGIMES = {
    "P": ("IndexP", "DistanceP"),
    "E": ("IndexA", "DistanceA"),
    "G_time_window": ("generated_rank_time_window", None),
    "G_historical": ("generated_rank_historical", None),
}

# Nested recorded-plan information sets used for the post-audit decomposition.
# DistanceP is defined along IndexP, so the defensible estimand is the incremental
# value of distance conditional on order, not an order-free distance effect.
PLAN_COMPONENT_REGIMES = {
    "P_position": ("IndexP", None, POSITION_FEATURES),
    "P_order": ("IndexP", None, ORDER_FEATURES),
}

FIXED_MODEL_PARAMS = {
    "objective": "quantile",
    "alpha": 0.5,
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

GENERATOR_MODEL_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 300,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED + 1,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def higher_quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="higher"))


def add_route_fields(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().reset_index(drop=True)
    if frame.duplicated(["Route ID", "IndexP"]).any():
        raise ValueError("(Route ID, IndexP) must uniquely identify row events")

    grouped = frame.groupby("Route ID", sort=False)
    frame["manifest_anchor"] = grouped["Earliest Time"].transform("min")
    frame["target_minutes"] = frame["Arrived Time"] - frame["manifest_anchor"]
    frame["earliest_rel"] = frame["Earliest Time"] - frame["manifest_anchor"]
    frame["latest_rel"] = frame["Latest Time"] - frame["manifest_anchor"]
    frame["time_window_width"] = frame["latest_rel"] - frame["earliest_rel"]
    frame["time_window_midpoint"] = (
        frame["earliest_rel"] + frame["latest_rel"]
    ) / 2.0
    frame["route_stop_count"] = grouped["IndexP"].transform("size").astype(int)
    frame["route_delivery_count"] = grouped["Delivery"].transform("sum").astype(int)
    frame["route_depot_count"] = grouped["Depot"].transform("sum").astype(int)
    frame["route_pickup_count"] = (
        frame["route_stop_count"]
        - frame["route_delivery_count"]
        - frame["route_depot_count"]
    ).clip(lower=0)
    denominator = (frame["route_stop_count"] - 1).clip(lower=1)
    frame["plan_rank_norm"] = frame["IndexP"] / denominator
    frame["actual_rank_norm"] = frame["IndexA"] / denominator
    frame["horizon_bin"] = pd.cut(
        frame["time_window_midpoint"],
        bins=HORIZON_EDGES,
        labels=HORIZON_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    frame["event_id"] = (
        frame["Route ID"].astype(str) + ":" + frame["IndexP"].astype(str)
    )
    return frame


def _mapping(frame: pd.DataFrame, key: str, value: str) -> dict[int, float]:
    return {
        int(k): float(v)
        for k, v in frame.groupby(key, sort=False)[value].mean().to_dict().items()
    }


def _counts(frame: pd.DataFrame, key: str) -> dict[int, int]:
    return {
        int(k): int(v) for k, v in frame.groupby(key, sort=False).size().to_dict().items()
    }


def fit_history_encoders(training: pd.DataFrame) -> dict[str, Any]:
    frame = add_route_fields(training)
    return {
        "global_target_mean": float(frame["target_minutes"].mean()),
        "global_plan_rank_mean": float(frame["plan_rank_norm"].mean()),
        "driver_target_mean": _mapping(frame, "Driver ID", "target_minutes"),
        "address_target_mean": _mapping(frame, "Address ID", "target_minutes"),
        "driver_plan_rank_mean": _mapping(frame, "Driver ID", "plan_rank_norm"),
        "address_plan_rank_mean": _mapping(frame, "Address ID", "plan_rank_norm"),
        "driver_count": _counts(frame, "Driver ID"),
        "address_count": _counts(frame, "Address ID"),
    }


def _leave_route_out_mean(
    frame: pd.DataFrame, key: str, value: str, fallback: float
) -> pd.Series:
    group_sum = frame.groupby(key, sort=False)[value].transform("sum")
    group_count = frame.groupby(key, sort=False)[value].transform("count")
    route_group_sum = frame.groupby([key, "Route ID"], sort=False)[value].transform("sum")
    route_group_count = frame.groupby([key, "Route ID"], sort=False)[value].transform("count")
    denominator = group_count - route_group_count
    result = (group_sum - route_group_sum) / denominator.where(denominator > 0)
    route_sum = frame.groupby("Route ID", sort=False)[value].transform("sum")
    route_count = frame.groupby("Route ID", sort=False)[value].transform("count")
    global_denominator = len(frame) - route_count
    global_without_route = (
        (float(frame[value].sum()) - route_sum)
        / global_denominator.where(global_denominator > 0)
    ).fillna(fallback)
    return result.fillna(global_without_route).astype(float)


def add_history_features(
    data: pd.DataFrame,
    encoders: dict[str, Any],
    *,
    training_crossfit: bool,
) -> pd.DataFrame:
    frame = add_route_fields(data)
    if training_crossfit:
        frame["driver_hist_target_mean"] = _leave_route_out_mean(
            frame, "Driver ID", "target_minutes", encoders["global_target_mean"]
        )
        frame["address_hist_target_mean"] = _leave_route_out_mean(
            frame, "Address ID", "target_minutes", encoders["global_target_mean"]
        )
        frame["driver_hist_plan_rank_mean"] = _leave_route_out_mean(
            frame, "Driver ID", "plan_rank_norm", encoders["global_plan_rank_mean"]
        )
        frame["address_hist_plan_rank_mean"] = _leave_route_out_mean(
            frame, "Address ID", "plan_rank_norm", encoders["global_plan_rank_mean"]
        )
    else:
        frame["driver_hist_target_mean"] = (
            frame["Driver ID"]
            .map(encoders["driver_target_mean"])
            .fillna(encoders["global_target_mean"])
        )
        frame["address_hist_target_mean"] = (
            frame["Address ID"]
            .map(encoders["address_target_mean"])
            .fillna(encoders["global_target_mean"])
        )
        frame["driver_hist_plan_rank_mean"] = (
            frame["Driver ID"]
            .map(encoders["driver_plan_rank_mean"])
            .fillna(encoders["global_plan_rank_mean"])
        )
        frame["address_hist_plan_rank_mean"] = (
            frame["Address ID"]
            .map(encoders["address_plan_rank_mean"])
            .fillna(encoders["global_plan_rank_mean"])
        )
    frame["driver_hist_count_log"] = np.log1p(
        frame["Driver ID"].map(encoders["driver_count"]).fillna(0).astype(float)
    )
    frame["address_hist_count_log"] = np.log1p(
        frame["Address ID"].map(encoders["address_count"]).fillna(0).astype(float)
    )
    return frame


def _stable_address_tie(address: pd.Series) -> pd.Series:
    values = address.astype("int64").to_numpy(dtype=np.uint64)
    hashed = (values * np.uint64(2_654_435_761)) % np.uint64(2**32)
    return pd.Series(hashed.astype(np.float64), index=address.index)


def add_time_window_generated_rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_manifest_tie"] = _stable_address_tie(result["Address ID"])
    result["_depot_priority"] = np.where(result["Depot"].eq(1), 0, 1)
    result["_pickup_priority"] = np.where(
        result["Depot"].eq(1), 0, np.where(result["Delivery"].eq(0), 1, 2)
    )
    ordered = result.sort_values(
        [
            "Route ID",
            "_depot_priority",
            "_pickup_priority",
            "latest_rel",
            "earliest_rel",
            "_manifest_tie",
        ],
        kind="stable",
    )
    rank = ordered.groupby("Route ID", sort=False).cumcount()
    result.loc[ordered.index, "generated_rank_time_window"] = rank.to_numpy()
    return result.drop(columns=["_manifest_tie", "_depot_priority", "_pickup_priority"])


def fit_historical_ranker(training: pd.DataFrame, n_estimators: int = 300) -> LGBMRegressor:
    params = dict(GENERATOR_MODEL_PARAMS)
    params["n_estimators"] = n_estimators
    model = LGBMRegressor(**params)
    model.fit(training[GENERATOR_FEATURES], training["plan_rank_norm"])
    return model


def add_historical_generated_rank(
    frame: pd.DataFrame, model: LGBMRegressor
) -> pd.DataFrame:
    result = frame.copy()
    result["_historical_rank_score"] = model.predict(result[GENERATOR_FEATURES])
    result["_manifest_tie"] = _stable_address_tie(result["Address ID"])
    ordered = result.sort_values(
        [
            "Route ID",
            "_historical_rank_score",
            "latest_rel",
            "earliest_rel",
            "_manifest_tie",
        ],
        kind="stable",
    )
    rank = ordered.groupby("Route ID", sort=False).cumcount()
    result.loc[ordered.index, "generated_rank_historical"] = rank.to_numpy()
    return result.drop(columns=["_manifest_tie"])


def add_generated_ranks(
    frame: pd.DataFrame, historical_ranker: LGBMRegressor
) -> pd.DataFrame:
    return add_historical_generated_rank(
        add_time_window_generated_rank(frame), historical_ranker
    )


def add_sequence_features(
    frame: pd.DataFrame, rank_column: str, distance_column: str | None
) -> pd.DataFrame:
    result = frame.copy()
    result["_original_order"] = np.arange(len(result))
    ordered = result.sort_values(["Route ID", rank_column], kind="stable").copy()
    grouped = ordered.groupby("Route ID", sort=False)
    ordered["seq_rank"] = grouped.cumcount().astype(float)
    denominator = (ordered["route_stop_count"] - 1).clip(lower=1)
    ordered["seq_rank_norm"] = ordered["seq_rank"] / denominator
    ordered["seq_remaining_stops"] = (
        ordered["route_stop_count"] - ordered["seq_rank"] - 1
    )
    ordered["seq_prev_time_window_midpoint"] = grouped[
        "time_window_midpoint"
    ].shift(1)
    ordered["seq_next_time_window_midpoint"] = grouped[
        "time_window_midpoint"
    ].shift(-1)
    ordered["seq_prev_delivery"] = grouped["Delivery"].shift(1)
    ordered["seq_next_delivery"] = grouped["Delivery"].shift(-1)
    ordered["seq_prev_depot"] = grouped["Depot"].shift(1)
    ordered["seq_next_depot"] = grouped["Depot"].shift(-1)
    ordered["seq_prev_address_hist_target_mean"] = grouped[
        "address_hist_target_mean"
    ].shift(1)
    ordered["seq_next_address_hist_target_mean"] = grouped[
        "address_hist_target_mean"
    ].shift(-1)

    if distance_column is None:
        for column in [
            "seq_edge_distance",
            "seq_cumulative_distance",
            "seq_remaining_distance",
            "seq_total_distance",
        ]:
            ordered[column] = np.nan
        ordered["seq_distance_available"] = 0.0
    else:
        ordered["seq_edge_distance"] = ordered[distance_column].astype(float)
        ordered["seq_cumulative_distance"] = grouped[distance_column].cumsum()
        ordered["seq_total_distance"] = grouped[distance_column].transform("sum")
        ordered["seq_remaining_distance"] = (
            ordered["seq_total_distance"] - ordered["seq_cumulative_distance"]
        )
        ordered["seq_distance_available"] = 1.0

    return ordered.sort_values("_original_order").drop(columns=["_original_order"])


def regime_frame(frame: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, list[str]]:
    if regime == "M":
        return frame, list(BASE_FEATURES)
    if regime in PLAN_COMPONENT_REGIMES:
        rank_column, distance_column, component_features = PLAN_COMPONENT_REGIMES[
            regime
        ]
        sequenced = add_sequence_features(frame, rank_column, distance_column)
        return sequenced, list(BASE_FEATURES + component_features)
    rank_column, distance_column = SEQUENCE_REGIMES[regime]
    sequenced = add_sequence_features(frame, rank_column, distance_column)
    return sequenced, list(BASE_FEATURES + SEQUENCE_FEATURES)


def _fit_arrival_model(
    frame: pd.DataFrame, features: list[str], n_estimators: int
) -> LGBMRegressor:
    params = dict(FIXED_MODEL_PARAMS)
    params["n_estimators"] = n_estimators
    model = LGBMRegressor(**params)
    deliveries = frame["Delivery"].eq(1)
    model.fit(frame.loc[deliveries, features], frame.loc[deliveries, "target_minutes"])
    return model


def fit_model_bundle(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    include_generated: bool,
    n_estimators: int = 500,
    generator_estimators: int = 300,
    regimes: Iterable[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    encoders = fit_history_encoders(training)
    train = add_history_features(training, encoders, training_crossfit=True)
    valid = add_history_features(validation, encoders, training_crossfit=False)
    historical_ranker = None
    selected_regimes = list(regimes) if regimes is not None else ["M", "P", "E"]
    generated_requested = any(regime.startswith("G_") for regime in selected_regimes)
    if include_generated and regimes is None:
        selected_regimes += ["G_time_window", "G_historical"]
        generated_requested = True
    if generated_requested and not include_generated:
        raise ValueError("Generated regimes require include_generated=True")
    if generated_requested:
        historical_ranker = fit_historical_ranker(train, generator_estimators)
        train = add_generated_ranks(train, historical_ranker)
        valid = add_generated_ranks(valid, historical_ranker)

    models: dict[str, LGBMRegressor] = {}
    feature_columns: dict[str, list[str]] = {}
    for regime in selected_regimes:
        train_regime, columns = regime_frame(train, regime)
        models[regime] = _fit_arrival_model(train_regime, columns, n_estimators)
        feature_columns[regime] = columns

    bundle = {
        "encoders": encoders,
        "historical_ranker": historical_ranker,
        "models": models,
        "feature_columns": feature_columns,
        "include_generated": include_generated,
        "model_params": {**FIXED_MODEL_PARAMS, "n_estimators": n_estimators},
        "generator_model_params": {
            **GENERATOR_MODEL_PARAMS,
            "n_estimators": generator_estimators,
        },
    }
    return bundle, predict_bundle(validation, bundle)


def predict_bundle(data: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    frame = add_history_features(data, bundle["encoders"], training_crossfit=False)
    if bundle["include_generated"]:
        frame = add_generated_ranks(frame, bundle["historical_ranker"])
    deliveries = frame["Delivery"].eq(1)
    base = frame.loc[
        deliveries,
        [
            "Route ID",
            "event_id",
            "IndexP",
            "IndexA",
            "Country",
            "Driver ID",
            "Address ID",
            "route_stop_count",
            "route_delivery_count",
            "time_window_width",
            "time_window_midpoint",
            "horizon_bin",
            "target_minutes",
            "driver_hist_count_log",
        ],
    ].copy()
    predictions = []
    for regime, model in bundle["models"].items():
        regime_data, columns = regime_frame(frame, regime)
        part = base.copy()
        part["regime"] = regime
        part["prediction"] = model.predict(regime_data.loc[deliveries, columns])
        part["residual"] = part["target_minutes"] - part["prediction"]
        part["absolute_error"] = part["residual"].abs()
        predictions.append(part)
    return pd.concat(predictions, ignore_index=True)


def point_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in predictions.groupby("regime", sort=True):
        per_route = group.groupby("Route ID")["residual"].agg(
            route_mae=lambda x: float(np.mean(np.abs(x))),
            route_bias="mean",
        )
        rows.append(
            {
                "regime": regime,
                "route_balanced_mae": float(per_route["route_mae"].mean()),
                "customer_weighted_mae": float(group["absolute_error"].mean()),
                "route_balanced_bias": float(per_route["route_bias"].mean()),
                "routes": int(group["Route ID"].nunique()),
                "deliveries": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def route_sequence_metrics(frame: pd.DataFrame, rank_column: str) -> pd.DataFrame:
    rows = []
    for route_id, route in frame.groupby("Route ID", sort=True):
        plan = route["IndexP"].to_numpy(dtype=float)
        generated = route[rank_column].to_numpy(dtype=float)
        tau = kendalltau(plan, generated).statistic
        plan_top = set(route.nsmallest(min(5, len(route)), "IndexP")["event_id"])
        generated_top = set(route.nsmallest(min(5, len(route)), rank_column)["event_id"])
        denominator = max(len(route) - 1, 1)
        rows.append(
            {
                "Route ID": int(route_id),
                "kendall_tau": float(0.0 if pd.isna(tau) else tau),
                "top_five_overlap": float(len(plan_top & generated_top) / len(plan_top)),
                "normalized_displacement": float(
                    np.mean(np.abs(plan - generated) / denominator)
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_sequence_summary(
    data: pd.DataFrame, bundle: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_history_features(data, bundle["encoders"], training_crossfit=False)
    frame = add_generated_ranks(frame, bundle["historical_ranker"])
    route_parts = []
    summary = []
    for candidate, column in [
        ("G_time_window", "generated_rank_time_window"),
        ("G_historical", "generated_rank_historical"),
    ]:
        route = route_sequence_metrics(frame, column)
        route["candidate"] = candidate
        route_parts.append(route)
        summary.append(
            {
                "candidate": candidate,
                "route_balanced_kendall_tau": float(route["kendall_tau"].mean()),
                "route_balanced_top_five_overlap": float(
                    route["top_five_overlap"].mean()
                ),
                "route_balanced_normalized_displacement": float(
                    route["normalized_displacement"].mean()
                ),
            }
        )
    return pd.DataFrame(summary), pd.concat(route_parts, ignore_index=True)


def select_generated_candidate(
    validation_metrics: pd.DataFrame, sequence_summary: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    candidates = validation_metrics[
        validation_metrics["regime"].isin(["G_time_window", "G_historical"])
    ][["regime", "route_balanced_mae"]].rename(columns={"regime": "candidate"})
    table = candidates.merge(sequence_summary, on="candidate", validate="one_to_one")
    table = table.sort_values(
        [
            "route_balanced_mae",
            "route_balanced_top_five_overlap",
            "candidate",
        ],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    table["selected"] = False
    table.loc[0, "selected"] = True
    return str(table.loc[0, "candidate"]), table


def fit_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, regime_data in predictions.groupby("regime", sort=True):
        global_residual = regime_data["residual"].to_numpy(dtype=float)
        global_center = float(np.median(global_residual))
        global_abs = np.abs(global_residual - global_center)
        for country in sorted(regime_data["Country"].unique()):
            for horizon in HORIZON_LABELS:
                group = regime_data[
                    regime_data["Country"].eq(country)
                    & regime_data["horizon_bin"].eq(horizon)
                ]
                use_group = len(group) >= MIN_CALIBRATION_GROUP
                residual = (
                    group["residual"].to_numpy(dtype=float)
                    if use_group
                    else global_residual
                )
                center = float(np.median(residual)) if use_group else global_center
                absolute = np.abs(residual - center) if use_group else global_abs
                row: dict[str, Any] = {
                    "regime": regime,
                    "Country": int(country),
                    "horizon_bin": horizon,
                    "source": "country_horizon" if use_group else "global_fallback",
                    "group_observations": int(len(group)),
                    "calibration_observations": int(len(residual)),
                    "median_residual": center,
                }
                for target in COVERAGE_TARGETS:
                    row[f"abs_q_{target:.2f}"] = higher_quantile(absolute, target)
                rows.append(row)
    return pd.DataFrame(rows)


def attach_calibration(
    predictions: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    return predictions.merge(
        calibration,
        on=["regime", "Country", "horizon_bin"],
        how="left",
        validate="many_to_one",
    )


def capacity_frontier(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    widths: Iterable[float] = WIDTH_CAPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = attach_calibration(predictions, calibration)
    route_parts = []
    rows = []
    for width in widths:
        center = scored["prediction"] + scored["median_residual"]
        hit = (scored["target_minutes"] - center).abs().le(width / 2.0).astype(float)
        temp = scored[["Route ID", "event_id", "regime"]].copy()
        temp["width_cap"] = float(width)
        temp["hit"] = hit
        per_route = temp.groupby(["Route ID", "regime", "width_cap"], as_index=False)[
            "hit"
        ].mean()
        route_parts.append(per_route)
        for regime, group in temp.groupby("regime", sort=True):
            route_coverage = per_route[per_route["regime"].eq(regime)]["hit"]
            rows.append(
                {
                    "regime": regime,
                    "width_cap": float(width),
                    "route_balanced_coverage": float(route_coverage.mean()),
                    "customer_weighted_coverage": float(group["hit"].mean()),
                    "mean_width": float(width),
                    "routes": int(group["Route ID"].nunique()),
                    "deliveries": int(len(group)),
                }
            )
    return pd.DataFrame(rows), pd.concat(route_parts, ignore_index=True)


def calibrated_interval_performance(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    widths: Iterable[float] = WIDTH_CAPS,
    targets: Iterable[float] = COVERAGE_TARGETS,
) -> pd.DataFrame:
    scored = attach_calibration(predictions, calibration)
    rows = []
    for target in targets:
        alpha = 1.0 - float(target)
        raw_width = 2.0 * scored[f"abs_q_{target:.2f}"]
        for cap in widths:
            width = np.minimum(float(cap), np.maximum(MIN_INTERVAL_WIDTH, raw_width))
            center = scored["prediction"] + scored["median_residual"]
            lower = center - width / 2.0
            upper = center + width / 2.0
            below = np.maximum(lower - scored["target_minutes"], 0.0)
            above = np.maximum(scored["target_minutes"] - upper, 0.0)
            hit = ((below == 0.0) & (above == 0.0)).astype(float)
            interval_score = width + (2.0 / alpha) * (below + above)
            temp = scored[["Route ID", "regime"]].copy()
            temp["hit"] = hit
            temp["width"] = width
            temp["interval_score"] = interval_score
            per_route = temp.groupby(["Route ID", "regime"])[
                ["hit", "width", "interval_score"]
            ].mean()
            for regime, group in temp.groupby("regime", sort=True):
                route = per_route.xs(regime, level="regime")
                rows.append(
                    {
                        "regime": regime,
                        "target_coverage": float(target),
                        "width_cap": float(cap),
                        "route_balanced_coverage": float(route["hit"].mean()),
                        "customer_weighted_coverage": float(group["hit"].mean()),
                        "route_balanced_mean_width": float(route["width"].mean()),
                        "customer_weighted_mean_width": float(group["width"].mean()),
                        "route_balanced_interval_score": float(
                            route["interval_score"].mean()
                        ),
                        "customer_weighted_interval_score": float(
                            group["interval_score"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def actual_plan_adherence(data: pd.DataFrame) -> pd.DataFrame:
    frame = add_route_fields(data)
    route = route_sequence_metrics(frame, "IndexA").rename(
        columns={
            "kendall_tau": "plan_actual_kendall_tau",
            "top_five_overlap": "plan_actual_top_five_overlap",
            "normalized_displacement": "plan_actual_normalized_displacement",
        }
    )
    attributes = []
    for route_id, group in frame.groupby("Route ID", sort=True):
        deliveries = group[group["Delivery"].eq(1)]
        attributes.append(
            {
                "Route ID": int(route_id),
                "Country": int(group["Country"].iloc[0]),
                "route_delivery_count": int(len(deliveries)),
                "driver_history_log": float(group["driver_hist_count_log"].iloc[0])
                if "driver_hist_count_log" in group
                else np.nan,
                "narrow_window_share": float(
                    deliveries["time_window_width"].le(180.0).mean()
                ),
            }
        )
    return route.merge(pd.DataFrame(attributes), on="Route ID", validate="one_to_one")


def freeze_quartile_thresholds(values: pd.Series) -> list[float]:
    return [
        float(x)
        for x in np.quantile(values.to_numpy(dtype=float), [0.25, 0.50, 0.75])
    ]


def apply_quartiles(values: pd.Series, thresholds: list[float]) -> pd.Series:
    array = values.to_numpy(dtype=float)
    labels = np.searchsorted(np.asarray(thresholds, dtype=float), array, side="right") + 1
    return pd.Series([f"Q{int(x)}" for x in labels], index=values.index)


def route_level_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["Route ID", "regime"], as_index=False)
        .agg(
            mae=("absolute_error", "mean"),
            bias=("residual", "mean"),
            deliveries=("event_id", "size"),
        )
    )


def paired_bootstrap_mean(
    differences: np.ndarray,
    *,
    draws: int = 2000,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    bootstrap = np.empty(draws, dtype=float)
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = rng.integers(0, n, size=(size, n))
        bootstrap[start : start + size] = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
