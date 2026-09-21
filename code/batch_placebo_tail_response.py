import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
BUDGET_OUT = ROOT / "outputs/TRE_query_budget_validation_20260913"
OP_OUT = ROOT / "outputs/TRE_operational_batch_validation_20260913"
OUT = OP_OUT

sys.path.insert(0, str(ROOT / "code"))
import query_budget_frontier as qb
import route_adaptive_allocation as ra
import operational_batch_validation as op


REPS = 100
PROTOCOL = {
    "date": "2026-09-13",
    "status": "Specified after the operational daily-batch result and before inspecting these outputs.",
    "questions": [
        "Is the daily-batch gain explained by DG information rather than route length or random concentration?",
        "Does the result survive metrics other than mean absolute error and both countries?",
        "Are cross-route DG scores ordered with realized proxy-stage value in forward data?",
        "Under the frozen severe nonresponse scenario, does a response model trained only on past "
        "synthetic responses improve allocation at matched attempts?",
    ],
    "placebos": {
        "random_capacity": "Random task priorities with the same daily capacity, minimum one, and cap ten.",
        "route_size_only": "Favor short routes, randomize tasks within equal route sizes.",
        "permuted_DG": "Permute DG scores among tasks inside each daily country batch.",
        "ER": "Use predicted baseline error rather than marginal gain.",
        "oracle": "Use realized proxy-stage gain; this is an unattainable diagnostic upper bound.",
    },
    "monte_carlo_repetitions": REPS,
    "primary_budget": "Previously frozen K=5 DG minimum-MAE budget for each window.",
    "operational_constraints": "Daily country batches, minimum one query per route, maximum ten.",
    "alternative_metrics": [
        "delivery-equal MAE",
        "delivery-equal RMSE",
        "median absolute error",
        "90th and 95th percentile absolute error",
        "within 30 minutes",
        "over 60 minutes",
        "route-equal RMSE and tail error",
    ],
    "response_model": (
        "Logistic regression of the frozen synthetic response indicator on stage entropy, fitted on "
        "the two-week budget-tuning period and applied to the following evaluation window."
    ),
    "statistics": (
        "Paired driver-cluster bootstrap, 2,000 resamples, seed 20260913. The exact repeated primary "
        "MAE contrast uses the final-audit seed 20260914 so it has one reported interval throughout."
    ),
    "limits": (
        "The response experiment validates the response-aware objective only within a synthetic "
        "entropy-linked scenario and is not evidence about human response behavior."
    ),
}


def freeze_protocol():
    path = OUT / "placebo_tail_response_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def load_eval(origin, k=5):
    name = f"w{origin}_k{k}"
    freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
    threshold = freeze["refusal_thresholds"]["0.5"]
    return op.add_fields(ra.load_window(origin, k), threshold).reset_index(drop=True)


def route_rows(q, selected, origin, label, condition="proxy_clean"):
    return ra.make_route_rows(q, selected, origin, 5, "DG", "past_min", label, condition)


def expected_placebo_rows(q, origin, budget, kind, reps=REPS):
    route_ids = np.sort(q["Route ID"].unique())
    route_sizes = q.groupby("Route ID").size().reindex(route_ids).to_numpy()
    drivers = q.groupby("Route ID")["Driver ID"].first().reindex(route_ids).to_numpy()
    baseline = q.groupby("Route ID").base_error.mean().reindex(route_ids).to_numpy()
    gains = np.zeros((reps, len(route_ids)), dtype=float)
    attempts = np.zeros((reps, len(route_ids)), dtype=float)
    route_position = {route: i for i, route in enumerate(route_ids)}
    batch_groups = [ids.index.to_numpy() for _, ids in q.groupby(op.BATCH_COLUMNS, sort=True)]
    rng = np.random.default_rng(20260913 + {"random_capacity": 11, "route_size_only": 23, "permuted_DG": 37}[kind])

    for rep in range(reps):
        if kind == "random_capacity":
            priority = rng.random(len(q))
        elif kind == "route_size_only":
            priority = 1.0 / q.route_size.to_numpy() + rng.random(len(q)) * 1e-10
        else:
            permuted = np.empty(len(q), dtype=float)
            for ids in batch_groups:
                permuted[ids] = rng.permutation(q.DG_score.to_numpy()[ids])
            priority = permuted / q.route_size.to_numpy()
        selected = op.daily_batch_selection(q, priority, budget, cap=10, minimum=1)
        z = q.assign(_selected=selected)
        gain_by_route = z.loc[z._selected].groupby("Route ID").clean_gain.sum()
        attempts_by_route = z.groupby("Route ID")._selected.sum()
        for route, value in gain_by_route.items():
            gains[rep, route_position[route]] = value / route_sizes[route_position[route]]
        attempts[rep] = attempts_by_route.reindex(route_ids, fill_value=0).to_numpy()

    mean_gain = gains.mean(axis=0)
    result = pd.DataFrame(
        {
            "Route ID": route_ids,
            "driver": drivers,
            "baseline_mae": baseline,
            "route_size": route_sizes,
            "attempts": attempts.mean(axis=0),
            "system_gain": mean_gain,
            "mae": baseline - mean_gain,
            "window": origin,
            "allocation": kind,
        }
    )
    return result


def placebo_experiment():
    choices = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    choices = choices[
        choices.k.eq(5)
        & choices.method.eq("DG")
        & choices.condition.eq("proxy_clean")
        & choices.rule.eq("minimum_mae")
    ]
    rows = []
    for choice in choices.itertuples(index=False):
        q = load_eval(choice.window)
        budget = int(choice.chosen_budget)
        selections = {
            "uniform_DG": qb.deterministic_selection(q, "DG_score", budget),
            "daily_DG": op.daily_batch_selection(q, q.DG_score / q.route_size, budget, cap=10, minimum=1),
            "daily_raw_DG": op.daily_batch_selection(q, q.DG_score, budget, cap=10, minimum=1),
            "daily_ER": op.daily_batch_selection(q, q.ER_score / q.route_size, budget, cap=10, minimum=1),
            "daily_oracle": op.daily_batch_selection(q, q.clean_gain / q.route_size, budget, cap=10, minimum=1),
        }
        for label, selected in selections.items():
            rows.append(route_rows(q, selected, choice.window, label))
        for kind in ["random_capacity", "route_size_only", "permuted_DG"]:
            rows.append(expected_placebo_rows(q, choice.window, budget, kind))
        print(time.strftime("%H:%M:%S"), "placebos", choice.window, flush=True)

    routes = pd.concat(rows, ignore_index=True)
    routes.to_parquet(OUT / "tables/placebo_route_results.parquet", index=False)
    summary = routes.groupby("allocation").agg(
        routes=("Route ID", "size"),
        drivers=("driver", "nunique"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "tables/placebo_summary.csv", index=False)

    dg = routes[routes.allocation.eq("daily_DG")]
    comparisons = []
    for label in ["uniform_DG", "daily_raw_DG", "daily_ER", "random_capacity", "route_size_only", "permuted_DG"]:
        comparator = routes[routes.allocation.eq(label)]
        seed = 20260914 if label == "uniform_DG" else 20260913
        stats = ra.driver_bootstrap_difference(dg, comparator, seed=seed)
        comparisons.append({"comparison": f"daily_DG_minus_{label}", **stats})
    pd.DataFrame(comparisons).to_csv(OUT / "tables/placebo_driver_cluster_comparisons.csv", index=False)
    return routes


def hybrid_errors(q, selected):
    return np.where(selected, q.proxy_error.to_numpy(), q.base_error.to_numpy())


def metric_values(errors):
    errors = np.asarray(errors, dtype=float)
    return {
        "mae": errors.mean(),
        "rmse": np.sqrt(np.mean(errors**2)),
        "median_ae": np.median(errors),
        "p90_ae": np.quantile(errors, 0.90),
        "p95_ae": np.quantile(errors, 0.95),
        "within_30_percent": 100 * np.mean(errors <= 30),
        "over_60_percent": 100 * np.mean(errors > 60),
    }


def alternative_metrics():
    choices = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    choices = choices[
        choices.k.eq(5)
        & choices.method.eq("DG")
        & choices.condition.eq("proxy_clean")
        & choices.rule.eq("minimum_mae")
    ]
    task_rows = []
    route_rows_all = []
    subgroup_rows = []
    for choice in choices.itertuples(index=False):
        q = load_eval(choice.window)
        budget = int(choice.chosen_budget)
        selections = {
            "baseline": np.zeros(len(q), dtype=bool),
            "uniform_DG": qb.deterministic_selection(q, "DG_score", budget),
            "daily_DG": op.daily_batch_selection(q, q.DG_score / q.route_size, budget, cap=10, minimum=1),
        }
        route_batch_size = (
            q[["Route ID"] + op.BATCH_COLUMNS]
            .drop_duplicates("Route ID")
            .groupby(op.BATCH_COLUMNS)["Route ID"]
            .transform("size")
        )
        route_batch_map = dict(
            zip(q[["Route ID"] + op.BATCH_COLUMNS].drop_duplicates("Route ID")["Route ID"], route_batch_size)
        )
        for policy, selected in selections.items():
            errors = hybrid_errors(q, selected)
            task_rows.append({"window": choice.window, "policy": policy, "tasks": len(q), **metric_values(errors)})
            z = q[["Route ID", "Driver ID", "_batch_country", "route_size"]].copy()
            z["error"] = errors
            per_route = z.groupby("Route ID").agg(
                driver=("Driver ID", "first"),
                country=("_batch_country", "first"),
                route_size=("route_size", "first"),
                mae=("error", "mean"),
                rmse=("error", lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))),
                p90_ae=("error", lambda x: np.quantile(x, 0.9)),
                over_60_percent=("error", lambda x: 100 * np.mean(np.asarray(x) > 60)),
            ).reset_index()
            per_route["window"] = choice.window
            per_route["policy"] = policy
            per_route["batch_routes"] = per_route["Route ID"].map(route_batch_map)
            route_rows_all.append(per_route)

    task_windows = pd.DataFrame(task_rows)
    # Pool task-level observations by reconstructing weighted first and second moments where possible;
    # exact pooled quantiles are computed below from the saved route-window inputs.
    exact_task = []
    for policy in ["baseline", "uniform_DG", "daily_DG"]:
        errors = []
        for choice in choices.itertuples(index=False):
            q = load_eval(choice.window)
            budget = int(choice.chosen_budget)
            if policy == "baseline":
                selected = np.zeros(len(q), dtype=bool)
            elif policy == "uniform_DG":
                selected = qb.deterministic_selection(q, "DG_score", budget)
            else:
                selected = op.daily_batch_selection(q, q.DG_score / q.route_size, budget, cap=10, minimum=1)
            errors.append(hybrid_errors(q, selected))
        exact_task.append({"policy": policy, "tasks": sum(map(len, errors)), **metric_values(np.concatenate(errors))})
    pd.DataFrame(exact_task).to_csv(OUT / "tables/alternative_task_metrics.csv", index=False)
    task_windows.to_csv(OUT / "tables/alternative_task_metrics_by_window.csv", index=False)

    routes = pd.concat(route_rows_all, ignore_index=True)
    routes.to_parquet(OUT / "tables/alternative_route_metrics.parquet", index=False)
    route_summary = routes.groupby("policy").agg(
        routes=("Route ID", "size"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        p90_ae=("p90_ae", "mean"),
        over_60_percent=("over_60_percent", "mean"),
    ).reset_index()
    route_summary.to_csv(OUT / "tables/alternative_route_metrics_summary.csv", index=False)

    comparisons = []
    daily = routes[routes.policy.eq("daily_DG")]
    uniform = routes[routes.policy.eq("uniform_DG")]
    for metric in ["mae", "rmse", "p90_ae", "over_60_percent"]:
        left = daily[["window", "Route ID", "driver", metric]].rename(columns={metric: "system_gain"})
        right = uniform[["window", "Route ID", metric]].rename(columns={metric: "system_gain"})
        seed = 20260914 if metric == "mae" else 20260913
        stats = ra.driver_bootstrap_difference(left, right, seed=seed)
        comparisons.append({"metric": metric, "direction": "negative favors daily_DG", **stats})
    pd.DataFrame(comparisons).to_csv(OUT / "tables/alternative_metric_driver_cluster_comparisons.csv", index=False)

    paired = daily.merge(
        uniform[["window", "Route ID", "mae"]], on=["window", "Route ID"], suffixes=("_daily", "_uniform")
    )
    paired["improvement"] = paired.mae_uniform - paired.mae_daily
    country = paired.groupby("country").agg(
        routes=("Route ID", "size"),
        uniform_mae=("mae_uniform", "mean"),
        daily_mae=("mae_daily", "mean"),
        improvement=("improvement", "mean"),
    ).reset_index()
    country.to_csv(OUT / "tables/country_stability.csv", index=False)
    paired["batch_size_group"] = pd.qcut(
        paired.batch_routes, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"], duplicates="drop"
    )
    batch_size = paired.groupby("batch_size_group", observed=True).agg(
        routes=("Route ID", "size"),
        mean_batch_routes=("batch_routes", "mean"),
        uniform_mae=("mae_uniform", "mean"),
        daily_mae=("mae_daily", "mean"),
        improvement=("improvement", "mean"),
    ).reset_index()
    batch_size.to_csv(OUT / "tables/batch_size_stability.csv", index=False)


def score_calibration():
    task_parts = []
    batch_rows = []
    for origin in [13, 19, 25]:
        q = load_eval(origin)
        q["priority"] = q.DG_score / q.route_size
        q["realized_route_contribution"] = q.clean_gain / q.route_size
        q["priority_decile"] = (
            q.groupby(op.BATCH_COLUMNS).priority.rank(method="first", pct=True).mul(10).apply(np.ceil).clip(1, 10).astype(int)
        )
        q["window"] = origin
        task_parts.append(q[["window", "_rowkey", "priority_decile", "DG_score", "priority", "clean_gain", "realized_route_contribution"]])
        for batch, group in q.groupby(op.BATCH_COLUMNS):
            batch_rows.append(
                {
                    "window": origin,
                    "country": batch[0],
                    "week": batch[1],
                    "day": batch[2],
                    "tasks": len(group),
                    "spearman_DG_gain": group.DG_score.corr(group.clean_gain, method="spearman"),
                    "spearman_priority_contribution": group.priority.corr(
                        group.realized_route_contribution, method="spearman"
                    ),
                }
            )
    tasks = pd.concat(task_parts, ignore_index=True)
    deciles = tasks.groupby("priority_decile").agg(
        tasks=("_rowkey", "size"),
        mean_DG_score=("DG_score", "mean"),
        mean_priority=("priority", "mean"),
        mean_clean_gain=("clean_gain", "mean"),
        mean_route_contribution=("realized_route_contribution", "mean"),
        positive_percent=("clean_gain", lambda x: 100 * np.mean(np.asarray(x) > 0)),
    ).reset_index()
    deciles.to_csv(OUT / "tables/DG_forward_calibration_deciles.csv", index=False)
    batches = pd.DataFrame(batch_rows)
    batches.to_csv(OUT / "tables/DG_forward_rank_correlations_by_batch.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "tasks": len(tasks),
                "global_spearman_DG_gain": tasks.DG_score.corr(tasks.clean_gain, method="spearman"),
                "global_spearman_priority_contribution": tasks.priority.corr(
                    tasks.realized_route_contribution, method="spearman"
                ),
                "mean_batch_spearman_DG_gain": batches.spearman_DG_gain.mean(),
                "median_batch_spearman_DG_gain": batches.spearman_DG_gain.median(),
                "positive_batch_correlation_percent": 100 * batches.spearman_DG_gain.gt(0).mean(),
            }
        ]
    )
    summary.to_csv(OUT / "tables/DG_forward_calibration_summary.csv", index=False)


def response_aware_experiment():
    choices = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    choices = choices[
        choices.k.eq(5)
        & choices.method.eq("DG")
        & choices.condition.eq("selective_nonresponse_severe")
        & choices.rule.eq("minimum_mae")
    ]
    route_parts = []
    model_rows = []
    for choice in choices.itertuples(index=False):
        tune = pd.read_parquet(OP_OUT / f"cache/w{choice.window}_k5_budget_tune.parquet")
        q = load_eval(choice.window)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=20260913)
        model.fit(tune[["entropy"]], tune.responds.astype(int))
        p_response = model.predict_proba(q[["entropy"]])[:, 1]
        model_rows.append(
            {
                "window": choice.window,
                "tuning_rows": len(tune),
                "evaluation_rows": len(q),
                "evaluation_response_rate": q.responds.mean(),
                "evaluation_auc": roc_auc_score(q.responds.astype(int), p_response),
                "coefficient_entropy": model.coef_[0, 0],
                "intercept": model.intercept_[0],
            }
        )
        budget = int(choice.chosen_budget)
        selections = {
            "uniform_DG": qb.deterministic_selection(q, "DG_score", budget),
            "daily_DG": op.daily_batch_selection(q, q.DG_score / q.route_size, budget, cap=10, minimum=1),
            "daily_response_DG": op.daily_batch_selection(
                q, p_response * q.DG_score.to_numpy() / q.route_size.to_numpy(), budget, cap=10, minimum=1
            ),
        }
        for label, selected in selections.items():
            route_parts.append(route_rows(q, selected, choice.window, label, "selective_nonresponse_severe"))
    routes = pd.concat(route_parts, ignore_index=True)
    routes.to_parquet(OUT / "tables/response_aware_route_results.parquet", index=False)
    summary = routes.groupby("allocation").agg(
        routes=("Route ID", "size"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
        responses=("responses", "sum"),
        attempts=("attempts", "sum"),
    ).reset_index()
    summary["response_rate"] = summary.responses / summary.attempts
    summary.to_csv(OUT / "tables/response_aware_summary.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT / "tables/synthetic_response_model.csv", index=False)
    response = routes[routes.allocation.eq("daily_response_DG")]
    comparisons = []
    for label in ["uniform_DG", "daily_DG"]:
        stats = ra.driver_bootstrap_difference(response, routes[routes.allocation.eq(label)])
        comparisons.append({"comparison": f"daily_response_DG_minus_{label}", **stats})
    pd.DataFrame(comparisons).to_csv(OUT / "tables/response_aware_driver_cluster_comparisons.csv", index=False)


def main():
    freeze_protocol()
    placebo_experiment()
    alternative_metrics()
    score_calibration()
    response_aware_experiment()
    completion_path = OUT / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf8"))
    completion.update(
        {
            "placebo_tail_response_complete": True,
            "placebo_repetitions": REPS,
            "extended_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
