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


def write_report(pooled, exact, adaptive_summary, full_comparison, intervals, responses):
    def get(frame, **filters):
        x = frame
        for column, value in filters.items():
            x = x[x[column].astype(str).eq(str(value))]
        assert len(x) == 1, (filters, len(x))
        return x.iloc[0]

    clean = pooled[pooled.split.eq("evaluation") & pooled.condition.eq("proxy_clean")]
    severe = pooled[
        pooled.split.eq("evaluation") & pooled.condition.eq("selective_nonresponse_severe")
    ]
    dg1 = get(clean, k=5, method="DG", budget=1)
    dg5 = get(clean, k=5, method="DG", budget=5)
    dgall = get(clean, k=5, method="DG", budget="all")
    er1 = get(clean, k=5, method="ER", budget=1)
    dist = get(exact, condition="proxy_clean", k=5, method="DG", budget=1)
    adaptive = get(adaptive_summary, condition="proxy_clean", k=5, method="ER", rule="minimum_mae")
    comparison = get(full_comparison, method="ER", rule="minimum_mae")
    severe_adaptive = get(
        adaptive_summary,
        condition="selective_nonresponse_severe",
        k=5,
        method="ER",
        rule="minimum_mae",
    )
    dg5_ci = get(intervals, split="evaluation", condition="proxy_clean", k=5, method="DG", budget=5)
    response = responses[
        responses.k.eq(5) & responses.method.eq("DG") & responses.budget.astype(str).eq("5")
    ]
    response_rate = np.average(response.response_rate, weights=response.attempts)

    text = f"""# 查询预算前沿验证结论

## 结论

现有数据足以支持重构，但新的主线应从“每条路线固定询问一次”改为“在不完全可靠的计划信息下，选择询问对象并控制询问深度”。查询价值随预算明显非单调；过去数据选择的预算可以转移到下一时间窗口，但响应可用性仍是决定性边界。

## 主要证据

- 五阶段 DG 从每路线 1 次增加到 5 次时，路线均衡 MAE 从 {dg1.mae:.3f} 降到 {dg5.mae:.3f} 分钟，相对不询问的改善由 {dg1.system_gain:.3f} 增至 {dg5.system_gain:.3f} 分钟。五次预算改善的司机簇 bootstrap 95% 区间为 [{dg5_ci.ci_low:.3f}, {dg5_ci.ci_high:.3f}]。
- 继续询问到 10 次后 MAE回升；询问全部任务时 MAE为 {dgall.mae:.3f}，只改善 {dgall.system_gain:.3f} 分钟。这表明更多代理信息并不保证更好的已拟合预测器表现。
- 五阶段 ER 在单次预算下达到 {er1.mae:.3f}，与 DG 接近。用每个窗口之前的两周选择预算后，ER 平均尝试 {adaptive.queries_per_route:.2f} 次，未来窗口 pooled MAE 为 {adaptive.mae:.3f}，改善 {adaptive.system_gain:.3f} 分钟，95%区间 [{adaptive.ci_low:.3f}, {adaptive.ci_high:.3f}]。
- 该过去数据选择策略与特权完整计划参考的 pooled MAE 差为 {comparison.adaptive_minus_full_plan:.3f} 分钟，区间 [{comparison.difference_ci_low:.3f}, {comparison.difference_ci_high:.3f}]；数值上取得其误差差距的 {comparison.fraction_of_full_plan_gap_percent:.1f}%。这是参考差距比较，不是成本等价或因果替代关系。
- 单次 DG 所选任务平均误差降低 {dist.gain_mean:.2f} 分钟，相对这些任务自身的基线为 {dist.relative_gain_percent:.1f}%；中位数 {dist.gain_median:.2f} 分钟，{dist.positive_percent:.1f}% 改善，{dist.harm_percent:.1f}% 恶化，{dist.over_30_percent:.1f}% 改善超过30分钟。
- 在严重熵关联缺答情景中，五阶段 DG、每路线5次尝试的有效响应率只有 {100*response_rate:.1f}%。过去数据选择的 ER 预算只改善 {severe_adaptive.system_gain:.3f} 分钟。响应结论仍是情景敏感性，不能写成司机行为发现。

## 可以写出的建设性主张

当计划阶段信号与历史训练信息存在来源差异时，预测收益取决于信息被使用的范围。按可观测风险排序并用近期已完成路线校准查询预算，可以让有限确认获得大部分可观测预测增量；无差别扩大信息覆盖会纳入大量负向更新并降低总体收益。因此，部署规则应同时决定询问对象、询问数量和停止位置，并在时间上重新校准。

## 不能写出的主张

- 不能把预测误差下降直接称为客户体验、迟到率或经济收益改善。
- 不能声称五次询问具有普遍最优性；三个时间窗口选择的预算不同。
- 不能声称已经学习真实响应概率；数据中没有人类响应标签。
- 不能声称局部确认替代完整路线数字化；完整计划模型使用了不同的信息和训练条件。

## 重构决定

可以开始重构。正文应以查询预算前沿、过去数据选预算的未来验证、任务级收益分布为三项主结果；单次询问结果降为前沿上的低成本端点。响应情景作为边界条件，ER作为简单实施策略，DG作为信息价值排序机制。完整计划和未来实际阶段继续作为参考与上界，不组成虚假的单调“信息阶梯”。
"""
    (OUT / "查询预算前沿验证结论.md").write_text(text, encoding="utf8")


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
    write_report(pooled, exact, adaptive_summary, full_comparison, intervals, responses)
    amendment = {
        "date": "2026-09-13",
        "change": "Added driver-cluster bootstrap as the reporting interval to match the existing paper; retained the route bootstrap output as a sensitivity check.",
        "reason": "Routes repeat within drivers, and the reconstructed manuscript already uses driver-cluster resampling.",
        "effect_estimates_changed": False,
    }
    (OUT / "protocol_amendment.json").write_text(
        json.dumps(amendment, ensure_ascii=False, indent=2), encoding="utf8"
    )
    completion = json.loads((OUT / "completion.json").read_text(encoding="utf8"))
    completion.update({"report_complete": True, "driver_cluster_intervals": True})
    (OUT / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
