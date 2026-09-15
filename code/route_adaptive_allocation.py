import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
NEAREST = ROOT / "outputs/TRE_nearest_controls_20260913"
BUDGET = ROOT / "outputs/TRE_query_budget_validation_20260913"
OUT = ROOT / "outputs/TRE_route_adaptive_allocation_20260913"

sys.path.insert(0, str(ROOT / "code"))
import query_budget_frontier as qb


BUDGETS = [1, 2, 3, 5, 10]
CAP = 10
METHODS = {"DG": "DG_score", "ER": "ER_score"}
CONDITIONS = {"proxy_clean": "clean_gain", "selective_nonresponse_severe": "response_gain"}

PROTOCOL = {
    "date": "2026-09-13",
    "status": "Prospectively specified extension after observing the fixed per-route budget frontier.",
    "question": (
        "At the same total number of attempted queries, does allocating a fleet-level query capacity "
        "across routes by predicted value improve on assigning the same quota to every route? Can a "
        "zero predicted-gain stopping rule improve on a workload-matched balanced quota?"
    ),
    "data_and_splits": (
        "Reuse the three frozen forward evaluation windows, task eligibility, five- and three-stage "
        "proxy answers, DG and ER scores, and severe entropy-linked nonresponse sensitivity scenario."
    ),
    "allocations": {
        "uniform_route": "Top b tasks within every route.",
        "fleet_raw_cap10": (
            "Top scores across the evaluation fleet subject to at most 10 queries per route and the "
            "same exact total attempts as uniform_route."
        ),
        "fleet_objective_cap10": (
            "As fleet_raw_cap10, but divide each task score by route size to align selection with the "
            "pre-specified route-equal MAE objective."
        ),
        "DG_positive_stop_cap10": (
            "Within each route, query DG-ranked tasks only while predicted gain is strictly positive, "
            "with at most 10 attempts. Threshold zero is fixed because DG is trained in minutes of "
            "absolute-error reduction."
        ),
        "balanced_matched_DG": (
            "Allocate exactly the same total attempts as DG_positive_stop_cap10 as evenly as route "
            "sizes permit, then use the same DG ranking within each route. Route assignment ties are "
            "resolved by a fixed hash independent of outcomes."
        ),
    },
    "primary_comparisons": (
        "Paired fleet_objective_cap10 minus uniform_route at each fixed budget; past-only budget choices "
        "from the previous validation are reused without change. DG_positive_stop_cap10 is compared "
        "with balanced_matched_DG at exactly matched attempts."
    ),
    "metrics": (
        "Route-equal MAE and gain are primary. Delivery-equal gain, gain per attempt, harmful-update "
        "share, response rate, route workload distribution, and driver-cluster bootstrap intervals "
        "are secondary."
    ),
    "statistics": "Paired driver-cluster bootstrap, 2,000 resamples, seed 20260913.",
    "limits": (
        "This is retrospective replay with software plan stages as proxy answers. It does not observe "
        "human response, interruption cost, customer outcomes, or online feedback."
    ),
}


def freeze_protocol():
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def route_hash(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:12], 16)


def load_window(origin, k):
    name = f"w{origin}_k{k}"
    q = pd.read_parquet(SCENARIO / f"cache/{name}_evaluation.parquet")
    q = qb.add_tie(q)
    nearest = pd.read_parquet(NEAREST / f"tables/{name}_scores.parquet")
    q = q.merge(nearest[["_rowkey", "error_only"]], on="_rowkey", how="left", validate="one_to_one")
    freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
    threshold = freeze["refusal_thresholds"]["0.5"]
    q["clean_gain"] = q.base_error - q.proxy_error
    q["responds"] = q.entropy.le(threshold)
    q["response_gain"] = q.clean_gain * q.responds
    q["DG_score"] = q.gain_score
    q["ER_score"] = q.error_only
    q["route_size"] = q.groupby("Route ID")["Route ID"].transform("size")
    return q


def global_capacity_selection(q, priority, total_attempts, cap=CAP):
    selected = np.zeros(len(q), dtype=bool)
    route_codes, _ = pd.factorize(q["Route ID"], sort=True)
    counts = np.zeros(route_codes.max() + 1, dtype=np.int32)
    values = np.asarray(priority, dtype=float)
    order = np.lexsort((q._tie.to_numpy(), -values))
    used = 0
    for i in order:
        code = route_codes[i]
        if counts[code] >= cap:
            continue
        selected[i] = True
        counts[code] += 1
        used += 1
        if used == total_attempts:
            break
    assert used == total_attempts
    return selected


def positive_stop_selection(q, score_column="DG_score", cap=CAP):
    selected = np.zeros(len(q), dtype=bool)
    values = q[score_column].to_numpy()
    route_codes, _ = pd.factorize(q["Route ID"], sort=True)
    for code in range(route_codes.max() + 1):
        ids = np.flatnonzero(route_codes == code)
        eligible = ids[values[ids] > 0]
        order = eligible[np.lexsort((q._tie.to_numpy()[eligible], -values[eligible]))]
        selected[order[:cap]] = True
    return selected


def balanced_quota_selection(q, score_column, total_attempts):
    """Match a total workload while keeping route quotas as even as possible."""
    route_sizes = q.groupby("Route ID").size().sort_index()
    route_order = sorted(route_sizes.index, key=route_hash)
    quotas = {route: 0 for route in route_order}
    remaining = int(total_attempts)
    level = 1
    while remaining:
        eligible = [route for route in route_order if route_sizes.loc[route] >= level]
        if not eligible:
            raise RuntimeError("Requested attempts exceed available tasks")
        take = min(remaining, len(eligible))
        for route in eligible[:take]:
            quotas[route] += 1
        remaining -= take
        level += 1

    selected = np.zeros(len(q), dtype=bool)
    values = q[score_column].to_numpy()
    route_codes, route_labels = pd.factorize(q["Route ID"], sort=True)
    for code, route in enumerate(route_labels):
        ids = np.flatnonzero(route_codes == code)
        order = ids[np.lexsort((q._tie.to_numpy()[ids], -values[ids]))]
        selected[order[: quotas[route]]] = True
    assert int(selected.sum()) == int(total_attempts)
    return selected


def make_route_rows(q, selected, origin, k, method, budget, allocation, condition):
    gain_column = CONDITIONS[condition]
    z = q.assign(_selected=selected)
    route = z.groupby("Route ID").agg(
        driver=("Driver ID", "first"),
        baseline_mae=("base_error", "mean"),
        route_size=("base_error", "size"),
        attempts=("_selected", "sum"),
    )
    gain_sum = z.loc[z._selected].groupby("Route ID")[gain_column].sum()
    clean_sum = z.loc[z._selected].groupby("Route ID")["clean_gain"].sum()
    response_count = z.loc[z._selected].groupby("Route ID")["responds"].sum()
    route["selected_gain_sum"] = gain_sum.reindex(route.index, fill_value=0.0)
    route["selected_clean_gain_sum"] = clean_sum.reindex(route.index, fill_value=0.0)
    route["responses"] = response_count.reindex(route.index, fill_value=0).astype(int)
    route["system_gain"] = route.selected_gain_sum / route.route_size
    route["mae"] = route.baseline_mae - route.system_gain
    route["window"] = origin
    route["k"] = k
    route["method"] = method
    route["budget"] = str(budget)
    route["allocation"] = allocation
    route["condition"] = condition
    return route.reset_index()


def gini(values):
    values = np.asarray(values, dtype=float)
    if values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    return (2 * np.dot(np.arange(1, n + 1), values) / (n * values.sum())) - (n + 1) / n


def summarize(route_rows):
    keys = ["condition", "k", "method", "budget", "allocation"]
    rows = []
    for key, group in route_rows.groupby(keys):
        attempts = group.attempts.to_numpy()
        rows.append(
            {
                **dict(zip(keys, key)),
                "routes": group["Route ID"].nunique(),
                "drivers": group.driver.nunique(),
                "baseline_mae": group.baseline_mae.mean(),
                "mae": group.mae.mean(),
                "route_equal_gain": group.system_gain.mean(),
                "delivery_equal_gain": group.selected_gain_sum.sum() / group.route_size.sum(),
                "queries_per_route": attempts.mean(),
                "gain_per_attempt": group.selected_gain_sum.sum() / max(attempts.sum(), 1),
                "response_rate": group.responses.sum() / max(attempts.sum(), 1),
                "zero_query_routes_percent": 100 * np.mean(attempts == 0),
                "routes_at_cap_percent": 100 * np.mean(attempts >= CAP),
                "attempts_p10": np.quantile(attempts, 0.10),
                "attempts_median": np.median(attempts),
                "attempts_p90": np.quantile(attempts, 0.90),
                "attempt_gini": gini(attempts),
            }
        )
    return pd.DataFrame(rows)


def task_summary(task_rows):
    rows = []
    keys = ["condition", "k", "method", "budget", "allocation"]
    for key, group in task_rows.groupby(keys):
        selected = group[group.selected]
        gain_col = CONDITIONS[key[0]]
        rows.append(
            {
                **dict(zip(keys, key)),
                "attempts": len(selected),
                "selected_gain_mean": selected[gain_col].mean(),
                "selected_gain_median": selected[gain_col].median(),
                "positive_percent": 100 * selected[gain_col].gt(0).mean(),
                "harm_percent": 100 * selected[gain_col].lt(0).mean(),
                "over_30_percent": 100 * selected[gain_col].gt(30).mean(),
            }
        )
    return pd.DataFrame(rows)


def driver_bootstrap_difference(left, right, reps=2000, seed=20260913):
    keys = ["window", "Route ID"]
    z = left[keys + ["driver", "system_gain"]].merge(
        right[keys + ["system_gain"]], on=keys, suffixes=("_left", "_right"), validate="one_to_one"
    )
    z["difference"] = z.system_gain_left - z.system_gain_right
    codes, drivers = pd.factorize(z.driver)
    counts = np.bincount(codes)
    totals = np.bincount(codes, weights=z.difference)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(drivers), np.ones(len(drivers)) / len(drivers), size=reps)
    draws = (multiplicities @ totals) / (multiplicities @ counts)
    return {
        "difference": z.difference.mean(),
        "ci_low": np.quantile(draws, 0.025),
        "ci_high": np.quantile(draws, 0.975),
        "routes": len(z),
        "drivers": len(drivers),
    }


def paired_comparisons(route_rows):
    rows = []
    for (condition, k, method, budget), group in route_rows.groupby(
        ["condition", "k", "method", "budget"]
    ):
        uniform = group[group.allocation.eq("uniform_route")]
        if uniform.empty:
            continue
        for allocation in ["fleet_raw_cap10", "fleet_objective_cap10"]:
            alternative = group[group.allocation.eq(allocation)]
            if alternative.empty:
                continue
            stats = driver_bootstrap_difference(alternative, uniform)
            rows.append(
                {
                    "condition": condition,
                    "k": k,
                    "method": method,
                    "budget": budget,
                    "comparison": f"{allocation}_minus_uniform_route",
                    **stats,
                }
            )
    for (condition, k), group in route_rows[
        route_rows.method.eq("DG") & route_rows.budget.eq("positive")
    ].groupby(["condition", "k"]):
        stop = group[group.allocation.eq("DG_positive_stop_cap10")]
        balanced = group[group.allocation.eq("balanced_matched_DG")]
        stats = driver_bootstrap_difference(stop, balanced)
        rows.append(
            {
                "condition": condition,
                "k": k,
                "method": "DG",
                "budget": "positive",
                "comparison": "DG_positive_stop_cap10_minus_balanced_matched_DG",
                **stats,
            }
        )
    return pd.DataFrame(rows)


def adaptive_comparisons(route_rows):
    choices = pd.read_csv(BUDGET / "tables/past_only_budget_choices.csv")
    selected_rows = []
    for choice in choices.itertuples(index=False):
        hit = route_rows[
            route_rows.window.eq(choice.window)
            & route_rows.k.eq(choice.k)
            & route_rows.condition.eq(choice.condition)
            & route_rows.method.eq(choice.method)
            & route_rows.budget.eq(str(choice.chosen_budget))
        ].copy()
        hit["rule"] = choice.rule
        hit["chosen_budget"] = str(choice.chosen_budget)
        selected_rows.append(hit)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected.to_parquet(OUT / "tables/past_only_allocation_route_results.parquet", index=False)

    rows = []
    for key, group in selected.groupby(["condition", "k", "method", "rule"]):
        condition, k, method, rule = key
        uniform = group[group.allocation.eq("uniform_route")]
        for allocation in ["fleet_raw_cap10", "fleet_objective_cap10"]:
            alternative = group[group.allocation.eq(allocation)]
            stats = driver_bootstrap_difference(alternative, uniform)
            rows.append(
                {
                    "condition": condition,
                    "k": k,
                    "method": method,
                    "rule": rule,
                    "allocation": allocation,
                    "baseline_mae": alternative.baseline_mae.mean(),
                    "uniform_mae": uniform.mae.mean(),
                    "allocation_mae": alternative.mae.mean(),
                    "uniform_gain": uniform.system_gain.mean(),
                    "allocation_gain": alternative.system_gain.mean(),
                    "queries_per_route": alternative.attempts.mean(),
                    **stats,
                }
            )
    return selected, pd.DataFrame(rows)


def main():
    for folder in ["tables", "figures", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    freeze_protocol()
    route_parts = []
    task_parts = []
    timing_rows = []

    for origin in [13, 19, 25]:
        for k in [3, 5]:
            q = load_window(origin, k)
            for method, score_column in METHODS.items():
                for budget in BUDGETS:
                    start = time.perf_counter()
                    uniform = qb.deterministic_selection(q, score_column, budget)
                    uniform_ms = 1000 * (time.perf_counter() - start)
                    target = int(uniform.sum())

                    start = time.perf_counter()
                    raw = global_capacity_selection(q, q[score_column], target)
                    raw_ms = 1000 * (time.perf_counter() - start)
                    start = time.perf_counter()
                    objective = global_capacity_selection(q, q[score_column] / q.route_size, target)
                    objective_ms = 1000 * (time.perf_counter() - start)
                    selections = {
                        "uniform_route": uniform,
                        "fleet_raw_cap10": raw,
                        "fleet_objective_cap10": objective,
                    }
                    timing_rows.append(
                        {
                            "window": origin,
                            "k": k,
                            "method": method,
                            "budget": budget,
                            "tasks": len(q),
                            "routes": q["Route ID"].nunique(),
                            "uniform_ms": uniform_ms,
                            "fleet_raw_ms": raw_ms,
                            "fleet_objective_ms": objective_ms,
                        }
                    )
                    for condition in CONDITIONS:
                        for allocation, selected in selections.items():
                            route_parts.append(
                                make_route_rows(q, selected, origin, k, method, budget, allocation, condition)
                            )
                            task = q[["_rowkey", "clean_gain", "response_gain"]].copy()
                            task["selected"] = selected
                            task["window"] = origin
                            task["k"] = k
                            task["method"] = method
                            task["budget"] = str(budget)
                            task["allocation"] = allocation
                            task["condition"] = condition
                            task_parts.append(task)

            positive = positive_stop_selection(q)
            balanced = balanced_quota_selection(q, "DG_score", int(positive.sum()))
            assert positive.sum() == balanced.sum()
            for condition in CONDITIONS:
                for allocation, selected in {
                    "DG_positive_stop_cap10": positive,
                    "balanced_matched_DG": balanced,
                }.items():
                    route_parts.append(
                        make_route_rows(q, selected, origin, k, "DG", "positive", allocation, condition)
                    )
                    task = q[["_rowkey", "clean_gain", "response_gain"]].copy()
                    task["selected"] = selected
                    task["window"] = origin
                    task["k"] = k
                    task["method"] = "DG"
                    task["budget"] = "positive"
                    task["allocation"] = allocation
                    task["condition"] = condition
                    task_parts.append(task)

            print(time.strftime("%H:%M:%S"), f"w{origin}_k{k}", "completed", flush=True)

    route_rows = pd.concat(route_parts, ignore_index=True)
    task_rows = pd.concat(task_parts, ignore_index=True)
    route_rows.to_parquet(OUT / "tables/route_results.parquet", index=False)
    pd.DataFrame(timing_rows).to_csv(OUT / "tables/allocation_runtime.csv", index=False)
    summarize(route_rows).to_csv(OUT / "tables/pooled_allocation_summary.csv", index=False)
    task_summary(task_rows).to_csv(OUT / "tables/selected_task_summary.csv", index=False)
    paired_comparisons(route_rows).to_csv(OUT / "tables/paired_driver_cluster_comparisons.csv", index=False)
    _, adaptive = adaptive_comparisons(route_rows)
    adaptive.to_csv(OUT / "tables/past_only_allocation_comparisons.csv", index=False)

    completion = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "windows": 3,
        "answer_categories": [3, 5],
        "routes": int(route_rows["Route ID"].nunique()),
        "route_rows": len(route_rows),
    }
    (OUT / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
