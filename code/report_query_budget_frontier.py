import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/TRE_query_budget_validation_20260913"
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
NEAREST = ROOT / "outputs/TRE_nearest_controls_20260913"
SCRIPT = ROOT / "code/query_budget_frontier.py"

spec = importlib.util.spec_from_file_location("budget_frontier", SCRIPT)
budget_frontier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget_frontier)

shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def driver_bootstrap(group, value_column, reps=2000, seed=20260913):
    ids = group[["Route ID", "driver"]].drop_duplicates().sort_values("Route ID")
    values = group.set_index("Route ID").reindex(ids["Route ID"])[value_column].to_numpy()
    codes, drivers = pd.factorize(ids.driver)
    counts = np.bincount(codes)
    totals = np.bincount(codes, weights=values)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(drivers), np.ones(len(drivers)) / len(drivers), size=reps
    )
    draws = (multiplicities @ totals) / (multiplicities @ counts)
    return values.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975), len(drivers)


def pooled_driver_intervals(route_results):
    rows = []
    keys = ["split", "condition", "k", "method", "budget"]
    for key, group in route_results.groupby(keys):
        mean, low, high, drivers = driver_bootstrap(group, "system_gain")
        rows.append(
            {
                **dict(zip(keys, key)),
                "system_gain": mean,
                "ci_low": low,
                "ci_high": high,
                "routes": group["Route ID"].nunique(),
                "drivers": drivers,
            }
        )
    return pd.DataFrame(rows)


def selected_task_rows():
    rows = []
    response_rows = []
    for origin in [13, 19, 25]:
        for k in [3, 5]:
            name = f"w{origin}_k{k}"
            q = pd.read_parquet(SCENARIO / f"cache/{name}_evaluation.parquet")
            q = budget_frontier.add_tie(q)
            nearest = pd.read_parquet(NEAREST / f"tables/{name}_scores.parquet")
            q = q.merge(nearest[["_rowkey", "error_only"]], on="_rowkey", how="left", sort=False)
            q["clean_gain"] = q.base_error - q.proxy_error
            freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
            q["responds"] = q.entropy.le(freeze["refusal_thresholds"]["0.5"])
            q["severe_gain"] = q.clean_gain * q.responds
            q["DG_score"] = q.gain_score
            q["ER_score"] = q.error_only
            q["uncertainty_score"] = q.entropy
            for method, score in [
                ("DG", "DG_score"),
                ("ER", "ER_score"),
                ("uncertainty", "uncertainty_score"),
            ]:
                for budget in [1, 2, 3, 5, 10, "all"]:
                    selected = budget_frontier.deterministic_selection(q, score, budget)
                    z = q.loc[selected, ["_rowkey", "Route ID", "Driver ID", "base_error", "clean_gain", "severe_gain", "responds"]].copy()
                    z["window"] = origin
                    z["k"] = k
                    z["method"] = method
                    z["budget"] = str(budget)
                    rows.append(z)
                    response_rows.append(
                        {
                            "window": origin,
                            "k": k,
                            "method": method,
                            "budget": str(budget),
                            "attempts": len(z),
                            "responses": int(z.responds.sum()),
                            "response_rate": z.responds.mean(),
                        }
                    )
            # Exact expected response rate under uniform random sampling.
            by_route = q.groupby("Route ID").agg(n=("responds", "size"), rate=("responds", "mean"))
            for budget in [1, 2, 3, 5, 10, "all"]:
                attempts = by_route.n if budget == "all" else np.minimum(int(budget), by_route.n)
                response_rows.append(
                    {
                        "window": origin,
                        "k": k,
                        "method": "random",
                        "budget": str(budget),
                        "attempts": attempts.sum(),
                        "responses": (attempts * by_route.rate).sum(),
                        "response_rate": np.average(by_route.rate, weights=attempts),
                    }
                )
    selected = pd.concat(rows, ignore_index=True)
    selected.to_parquet(OUT / "tables/selected_task_gains.parquet", index=False)
    responses = pd.DataFrame(response_rows)
    responses.to_csv(OUT / "tables/response_rates_by_window.csv", index=False)
    return selected, responses


def exact_distributions(selected):
    rows = []
    for keys, group in selected.groupby(["k", "method", "budget"]):
        k, method, budget = keys
        for condition, gain_col in [("proxy_clean", "clean_gain"), ("selective_nonresponse_severe", "severe_gain")]:
            gain = group[gain_col]
            base = group.base_error
            rows.append(
                {
                    "condition": condition,
                    "k": k,
                    "method": method,
                    "budget": budget,
                    "queries": len(group),
                    "baseline_error_mean": base.mean(),
                    "gain_mean": gain.mean(),
                    "gain_median": gain.median(),
                    "relative_gain_percent": 100 * gain.mean() / base.mean(),
                    "positive_percent": 100 * gain.gt(0).mean(),
                    "harm_percent": 100 * gain.lt(0).mean(),
                    "over_30_percent": 100 * gain.gt(30).mean(),
                    "gain_q10": gain.quantile(0.1),
                    "gain_q90": gain.quantile(0.9),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tables/query_gain_distributions_exact.csv", index=False)
    return result


def adaptive_routes(route_results):
    choices = pd.read_csv(OUT / "tables/past_only_budget_evaluation.csv")
    pieces = []
    for choice in choices.itertuples(index=False):
        hit = route_results[
            route_results.window.eq(choice.window)
            & route_results.split.eq("evaluation")
            & route_results.condition.eq(choice.condition)
            & route_results.k.eq(choice.k)
            & route_results.method.eq(choice.method)
            & route_results.budget.astype(str).eq(str(choice.chosen_budget))
        ].copy()
        assert len(hit) > 0
        hit["rule"] = choice.rule
        hit["chosen_budget"] = str(choice.chosen_budget)
        pieces.append(hit)
    result = pd.concat(pieces, ignore_index=True)
    result.to_parquet(OUT / "tables/past_only_adaptive_route_results.parquet", index=False)
    rows = []
    for key, group in result.groupby(["condition", "k", "method", "rule"]):
        mean, low, high, drivers = driver_bootstrap(group, "system_gain")
        rows.append(
            {
                **dict(zip(["condition", "k", "method", "rule"], key)),
                "baseline_mae": group.baseline_mae.mean(),
                "mae": group.mae.mean(),
                "system_gain": mean,
                "ci_low": low,
                "ci_high": high,
                "queries_per_route": group.attempts.mean(),
                "routes": group["Route ID"].nunique(),
                "drivers": drivers,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "tables/past_only_adaptive_summary.csv", index=False)
    return result, summary


def compare_full_plan(adaptive):
    full = pd.concat(
        [pd.read_csv(SCENARIO / f"tables/w{o}_privileged_full_plan.csv") for o in [13, 19, 25]],
        ignore_index=True,
    ).rename(columns={"origin": "window", "mae": "full_plan_mae"})
    rows = []
    for key, group in adaptive[
        adaptive.condition.eq("proxy_clean") & adaptive.k.eq(5)
    ].groupby(["method", "rule"]):
        method, rule = key
        z = group.merge(full[["window", "Route ID", "full_plan_mae"]], on=["window", "Route ID"], validate="one_to_one")
        z["adaptive_minus_full_plan"] = z.mae - z.full_plan_mae
        z["full_plan_gain"] = z.baseline_mae - z.full_plan_mae
        diff, low, high, drivers = driver_bootstrap(z, "adaptive_minus_full_plan")
        rows.append(
            {
                "method": method,
                "rule": rule,
                "adaptive_mae": z.mae.mean(),
                "full_plan_mae": z.full_plan_mae.mean(),
                "adaptive_minus_full_plan": diff,
                "difference_ci_low": low,
                "difference_ci_high": high,
                "adaptive_gain": z.system_gain.mean(),
                "full_plan_gain": z.full_plan_gain.mean(),
                "fraction_of_full_plan_gap_percent": 100 * z.system_gain.mean() / z.full_plan_gain.mean(),
                "queries_per_route": z.attempts.mean(),
                "routes": len(z),
                "drivers": drivers,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tables/adaptive_vs_full_plan.csv", index=False)
    return result


def marginal_table(pooled):
    order = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 5, "10": 10, "all": 99}
    rows = []
    subset = pooled[pooled.split.eq("evaluation")].copy()
    for key, group in subset.groupby(["condition", "k", "method"]):
        group = group.copy()
        group["_order"] = group.budget.astype(str).map(order)
        group = group.sort_values("_order")
        previous_gain = previous_queries = None
        for row in group.itertuples(index=False):
            marginal = np.nan
            if previous_gain is not None and row.queries_per_route > previous_queries:
                marginal = (row.system_gain - previous_gain) / (row.queries_per_route - previous_queries)
            rows.append(
                {
                    "condition": key[0],
                    "k": key[1],
                    "method": key[2],
                    "budget": row.budget,
                    "queries_per_route": row.queries_per_route,
                    "system_gain": row.system_gain,
                    "marginal_system_gain_per_added_query": marginal,
                }
            )
            previous_gain, previous_queries = row.system_gain, row.queries_per_route
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tables/marginal_query_value.csv", index=False)
    return result


def make_figures(pooled):
    plot_data = pooled[
        pooled.split.eq("evaluation") & pooled.condition.eq("proxy_clean")
    ].copy()
    colors = {"DG": "#1769aa", "ER": "#ef6c00", "uncertainty": "#6a1b9a", "random": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, k in zip(axes, [3, 5]):
        for method in ["DG", "ER", "uncertainty", "random"]:
            x = plot_data[(plot_data.k.eq(k)) & (plot_data.method.eq(method))].sort_values("queries_per_route")
            ax.plot(x.queries_per_route, x.mae, marker="o", linewidth=2, label=method, color=colors[method])
        ax.axhline(73.349342, color="black", linestyle="--", linewidth=1, label="No query" if k == 3 else None)
        ax.set_title(f"{k} answer categories")
        ax.set_xlabel("Attempted confirmations per route")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Route-balanced MAE (minutes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "figures/query_budget_frontier.png", dpi=240)
    fig.savefig(OUT / "figures/query_budget_frontier.pdf")
    plt.close(fig)

    data = pooled[
        pooled.split.eq("evaluation")
        & pooled.k.eq(5)
        & pooled.method.isin(["DG", "ER"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, condition, title in zip(
        axes,
        ["proxy_clean", "selective_nonresponse_severe"],
        ["Proxy answers available", "Severe entropy-linked nonresponse"],
    ):
        for method in ["DG", "ER"]:
            x = data[data.condition.eq(condition) & data.method.eq(method)].sort_values("queries_per_route")
            ax.plot(x.queries_per_route, x.mae, marker="o", linewidth=2, label=method, color=colors[method])
        ax.axhline(73.349342, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Attempted confirmations per route")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Route-balanced MAE (minutes)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "figures/response_budget_frontier.png", dpi=240)
    fig.savefig(OUT / "figures/response_budget_frontier.pdf")
    plt.close(fig)


def main():
    route_results = pd.read_parquet(OUT / "tables/route_results.parquet")
    pooled = pd.read_csv(OUT / "tables/pooled_budget_curves.csv")
    intervals = pooled_driver_intervals(route_results[route_results.split.eq("evaluation")])
    intervals.to_csv(OUT / "tables/evaluation_driver_cluster_intervals.csv", index=False)
    selected, responses = selected_task_rows()
    exact = exact_distributions(selected)
    adaptive, adaptive_summary = adaptive_routes(route_results)
    full_comparison = compare_full_plan(adaptive)
    marginal_table(pooled)
    make_figures(pooled)
    amendment = {
        "date": "2026-09-13",
        "change": "Added driver-cluster bootstrap as the primary reporting interval; retained the route bootstrap output as a sensitivity check.",
        "reason": "Routes repeat within drivers, so driver-cluster resampling preserves the dependence structure.",
        "effect_estimates_changed": False,
    }
    (OUT / "protocol_amendment.json").write_text(
        json.dumps(amendment, ensure_ascii=False, indent=2), encoding="utf8"
    )
    completion = json.loads((OUT / "completion.json").read_text(encoding="utf8"))
    completion.update({"reporting_outputs_complete": True, "driver_cluster_intervals": True})
    (OUT / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
