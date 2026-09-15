import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/TRE_operational_batch_validation_20260913"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def csv(name):
    return pd.read_csv(TABLES / name)


def integrity_checks():
    checks = []

    def check(name, condition, detail):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    final_cmp = csv("final_audit_driver_cluster_comparisons.csv")
    primary = final_cmp[(final_cmp.scope == "pooled") & final_cmp.comparison.eq("daily_DG_minus_uniform_DG")].iloc[0]
    total = final_cmp[(final_cmp.scope == "pooled") & final_cmp.comparison.eq("daily_DG_minus_no_query")].iloc[0]
    check("primary matched-workload increment", primary.ci_low > 0, f"CI [{primary.ci_low:.3f}, {primary.ci_high:.3f}]")
    check("primary total gain", total.ci_low > 0, f"CI [{total.ci_low:.3f}, {total.ci_high:.3f}]")

    placebo = csv("placebo_driver_cluster_comparisons.csv")
    identifying_placebos = placebo[~placebo.comparison.str.contains("daily_raw_DG")]
    check(
        "identifying placebo comparisons",
        identifying_placebos.ci_low.gt(0).all(),
        f"minimum lower CI={identifying_placebos.ci_low.min():.3f}",
    )

    alt = csv("alternative_metric_driver_cluster_comparisons.csv")
    check("alternative metrics", alt.ci_high.lt(0).all(), f"maximum upper CI={alt.ci_high.max():.3f}")

    stress = csv("operational_stress_driver_cluster_comparisons.csv")
    stress5 = stress[stress.k.eq(5)]
    check("K5 stress: matched-workload increment", stress5[stress5.comparison.str.contains("minus_uniform")].ci_low.gt(0).all(), "all lower CIs > 0")
    check("K5 stress: total gain", stress5[stress5.comparison.str.contains("gain_vs_zero")].ci_low.gt(0).all(), "all lower CIs > 0")

    waves = csv("dispatch_wave_driver_cluster_comparisons.csv")
    check("dispatch-wave granularity", waves.ci_low.gt(0).all(), f"minimum lower CI={waves.ci_low.min():.3f}")

    anchors = csv("batch_anchor_driver_cluster_comparisons.csv")
    check("batch-anchor reconstruction", anchors.ci_low.gt(0).all(), f"minimum lower CI={anchors.ci_low.min():.3f}")

    direct = csv("direct_response_gain_driver_cluster_comparisons.csv")
    direct5 = direct[direct.k.eq(5)]
    check("direct response-gain target", direct5.ci_low.gt(0).all(), f"minimum lower CI={direct5.ci_low.min():.3f}")

    capacity = csv("final_audit_batch_capacity.csv")
    check("exact capacity by dispatch batch", capacity.attempt_difference.eq(0).all(), f"max absolute difference={capacity.attempt_difference.abs().max()}")

    invariance = csv("final_audit_selection_invariance.csv")
    check("outcome-field exclusion", invariance.outcome_field_invariant.all(), "all three windows invariant")
    check("row-order invariance", invariance.row_order_invariant.all(), "all three windows invariant")

    runtime = csv("final_audit_runtime.csv")
    check("dispatch-time runtime", runtime.combined_p95_ms.lt(1000).all(), f"maximum p95={runtime.combined_p95_ms.max():.1f} ms")

    selected = csv("final_audit_selected_gain_summary.csv")
    check("attempt-count match", selected.attempts.nunique() == 1 and selected.attempts.iloc[0] == 17866, "17,866 attempts per policy")

    er = csv("dg_er_robustness.csv")
    check(
        "DG versus independently tuned ER",
        er[er.comparison.eq("separate_past_only_minimum_mae")].ci_low.iloc[0] > 0,
        "minimum-MAE contrast excludes zero",
    )
    cohort = csv("reconstructed_cohort_size_summary.csv").iloc[0]
    check(
        "reconstructed cohort size audit",
        cohort.cohorts == len(capacity) and cohort.routes_median > 1,
        f"{int(cohort.cohorts)} cohorts; median {cohort.routes_median:.1f} routes",
    )
    pd.DataFrame(checks).to_csv(OUT / "integrity_checks.csv", index=False)
    return checks


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.7)


def save_figure(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    blue, orange, gray, green = "#2563EB", "#F97316", "#64748B", "#16A34A"

    fixed = csv("fixed_grid_summary.csv")
    f = fixed[
        fixed.split.eq("evaluation")
        & fixed.condition.eq("proxy_clean")
        & fixed.k.eq(5)
        & fixed.method.eq("DG")
        & fixed.allocation.isin(["uniform_route", "daily_objective_cap10_min1"])
    ].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for label, color, display in [
        ("uniform_route", gray, "Fixed quota per route"),
        ("daily_objective_cap10_min1", blue, "Reconstructed-cohort allocation"),
    ]:
        z = f[f.allocation.eq(label)].sort_values("queries_per_route")
        ax.plot(z.queries_per_route, z.system_gain, marker="o", linewidth=2, color=color, label=display)
    ax.axhline(2.719110, color=green, linestyle="--", linewidth=1.5, label="Privileged full-plan gain")
    ax.set(xlabel="Queries per route", ylabel="Route-equal MAE reduction (min)", title="Information-cost frontier (K=5, forward evaluation)")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, "figure_1_information_cost_frontier")

    placebo = csv("placebo_summary.csv").set_index("allocation")
    order = ["daily_DG", "daily_raw_DG", "uniform_DG", "daily_ER", "route_size_only", "permuted_DG", "random_capacity"]
    labels = ["Cohort DG/size", "Cohort raw DG", "Fixed quota", "Cohort error rank", "Route size only", "Permuted DG", "Random"]
    colors = [blue, "#60A5FA", gray, orange, "#A78BFA", "#C4B5FD", "#CBD5E1"]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.4, 4.4), layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    y = np.arange(len(order))
    axes[0].barh(y, placebo.loc[order].system_gain, color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Route-equal MAE reduction (min)")
    axes[0].set_title("Matched-workload controls")
    style_axis(axes[0])
    axes[0].grid(axis="y", visible=False)
    axes[0].grid(axis="x", alpha=0.2, linewidth=0.7)
    audit_tasks = pd.read_parquet(TABLES / "final_audit_selected_tasks.parquet")
    for allocation, color, label in [("daily_DG", blue, "Cohort DG/size"), ("uniform_DG", gray, "Fixed quota")]:
        values = np.sort(audit_tasks.loc[audit_tasks.allocation.eq(allocation), "clean_gain"].clip(-150, 150))
        axes[1].plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=2, label=label)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Queried-task error reduction (min, clipped)", ylabel="Empirical CDF", title="Selected-task gain distribution")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    save_figure(fig, "figure_2_controls_and_selected_gain")

    stress = csv("operational_stress_summary.csv")
    stress = stress[stress.k.eq(5)]
    stress_order = ["proxy_clean", "adjacent_error10", "adjacent_error20", "independent_refusal25", "independent_refusal50", "entropy_refusal25", "entropy_refusal50"]
    stress_labels = ["Clean", "10% adj.\nerror", "20% adj.\nerror", "25% random\nrefusal", "50% random\nrefusal", "25% selective\nrefusal", "50% selective\nrefusal"]
    x = np.arange(len(stress_order))
    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.35), layout="constrained",
        gridspec_kw={"width_ratios": [1.28, 1]},
    )
    for shift, policy, color, label in [(-0.18, "uniform_DG", gray, "Fixed quota"), (0.18, "daily_DG", blue, "Cohort DG/size")]:
        vals = stress.set_index(["condition", "policy"]).loc[[(c, policy) for c in stress_order], "system_gain"].to_numpy()
        axes[0].bar(x + shift, vals, width=0.36, color=color, label=label)
    axes[0].set_xticks(x, stress_labels)
    axes[0].tick_params(axis="x", labelsize=7.7)
    axes[0].set_ylabel("Route-equal MAE reduction (min)")
    axes[0].set_title("Answer noise and refusal stress tests")
    axes[0].legend(frameon=False)
    style_axis(axes[0])
    waves = csv("dispatch_wave_driver_cluster_comparisons.csv")
    waves = waves[waves.wave_size.astype(str).ne("daily")].copy()
    waves["wave_numeric"] = pd.to_numeric(waves.wave_size)
    waves = waves.sort_values("wave_numeric")
    axes[1].errorbar(waves.wave_numeric, waves.difference,
                     yerr=[waves.difference - waves.ci_low, waves.ci_high - waves.difference],
                     marker="o", color=blue, linewidth=2, capsize=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Routes coordinated per dispatch wave", ylabel="Increment over fixed quota (min)", title="Coordination scope sensitivity")
    axes[1].set_xscale("log")
    axes[1].set_xticks(waves.wave_numeric, waves.wave_numeric.astype(int).astype(str))
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style_axis(axes[1])
    save_figure(fig, "figure_3_stress_and_coordination")

    anchors = csv("batch_anchor_driver_cluster_comparisons.csv")
    anchor_order = ["earliest_executed", "latest_executed", "modal_event_date", "strict_single_date_routes"]
    anchor_labels = ["First executed stop", "Last executed stop", "Modal event date", "Single-date routes only"]
    a = anchors.set_index("anchor_definition").loc[anchor_order]
    direct = csv("direct_response_gain_summary.csv")
    d = direct[(direct.k.eq(5)) & direct.rule.eq("minimum_mae")].set_index("allocation")
    d_order = ["direct_response_gain", "plain_DG_same_config", "uniform_DG_same_budget"]
    d_labels = ["Direct E[R x gain | X]", "Plain DG, same config", "Fixed quota, same budget"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    axes[0].errorbar(np.arange(4), a.difference,
                     yerr=[a.difference - a.ci_low, a.ci_high - a.difference],
                     fmt="o", color=blue, capsize=4)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(np.arange(4), anchor_labels, rotation=25, ha="right")
    axes[0].set_ylabel("Gain beyond fixed quota (min)")
    axes[0].set_title("Dispatch-date reconstruction")
    style_axis(axes[0])
    axes[1].bar(np.arange(3), d.loc[d_order].system_gain, color=[green, orange, gray])
    axes[1].set_xticks(np.arange(3), d_labels, rotation=25, ha="right")
    axes[1].set_ylabel("Route-equal MAE reduction (min)")
    axes[1].set_title("Synthetic selective nonresponse")
    style_axis(axes[1])
    save_figure(fig, "figure_4_anchor_and_response_target")


def write_report():
    cmp = csv("final_audit_driver_cluster_comparisons.csv")
    primary = cmp[(cmp.scope == "pooled") & cmp.comparison.eq("daily_DG_minus_uniform_DG")].iloc[0]
    total = cmp[(cmp.scope == "pooled") & cmp.comparison.eq("daily_DG_minus_no_query")].iloc[0]
    window_effects = cmp[cmp.scope.str.startswith("window_")].sort_values("scope")
    task = csv("final_audit_selected_gain_summary.csv").set_index("allocation")
    workload = csv("final_audit_workload_summary.csv").set_index("allocation")
    alt_tasks = csv("alternative_task_metrics.csv").set_index("policy")
    full = csv("final_audit_vs_privileged_full_plan.csv").set_index("policy")
    place = csv("placebo_summary.csv").set_index("allocation")
    place_ci = csv("placebo_driver_cluster_comparisons.csv").set_index("comparison")
    identifying_place_ci = place_ci[~place_ci.index.to_series().str.contains("daily_raw_DG")]
    stress = csv("operational_stress_driver_cluster_comparisons.csv")
    wave = csv("dispatch_wave_driver_cluster_comparisons.csv")
    anchors = csv("batch_anchor_driver_cluster_comparisons.csv")
    direct = csv("direct_response_gain_summary.csv")
    direct_ci = csv("direct_response_gain_driver_cluster_comparisons.csv")
    calibration = csv("DG_forward_calibration_summary.csv").iloc[0]
    runtime = csv("final_audit_runtime.csv")
    cohort = csv("reconstructed_cohort_size_summary.csv").iloc[0]
    cohort_slices = csv("batch_size_stability.csv")
    er_robustness = csv("dg_er_robustness.csv").set_index("comparison")

    p = full.loc["daily_DG_fixed_past_budget"]
    d5 = direct[(direct.k.eq(5)) & direct.rule.eq("minimum_mae")].set_index("allocation")
    d5ci = direct_ci[(direct_ci.k.eq(5)) & direct_ci.rule.eq("minimum_mae")].set_index("comparison")
    severe = stress[(stress.k.eq(5)) & stress.condition.eq("entropy_refusal50")]
    severe_vs_uniform = severe[severe.comparison.str.contains("minus_uniform")].iloc[0]
    severe_vs_zero = severe[severe.comparison.str.contains("gain_vs_zero")].iloc[0]
    wave2 = wave[wave.wave_size.astype(str).eq("2")].iloc[0]

    report = f"""# 写论文前的最终实验冻结报告

**冻结日期：2026-09-14**  
**结论：离线验证足以支持一篇以“有限询问资源的跨路线分配”为核心的建设性论文。公开数据中的共享批次是运营上合理的重建组，而非真实记录的共同派发关系；真实批次和司机响应均须由下一阶段外部验证。**

## 1. 最终要主张什么

真实业务问题不是“哪个模型的平均 MAE 最低”，而是：在配送计划只部分可见、运营方每天只能向司机发出有限次确认时，应该把这些确认机会给哪些路线、哪些任务。

我们验证出的办法是：先估计每个任务在获得粗粒度意图阶段后可减少多少 ETA 误差（DG），再把可共享容量的路线作为一个整体分配。每条路线至少获得 1 次询问，单条路线最多 10 次；其余机会按 `预测 DG / 路线任务数` 跨路线排序。预算、上限和模型均只用过去窗口确定，随后在三个未来窗口评估。离线数据没有真实批次字段，因此使用国家与日期信息重建共享组。

这构成一条完整的论文逻辑：

1. **发现问题：** 固定“每条路线问同样多次”忽略了不同路线的信息价值差异。
2. **解释原因：** DG 虽然不是精确的收益预测，但在未来重建组中具有稳定的相对排序信号；{calibration.positive_batch_correlation_percent:.1f}% 的组呈正相关。
3. **提出办法：** 在同组、同总询问量下，把容量从低预期贡献路线转给高预期贡献路线，并用最低覆盖和单路线封顶约束集中程度。
4. **验证办法：** 严格前向窗口、组内等预算、司机聚类自助法、安慰剂、替代指标、噪声/拒答压力、组规模与日期锚点敏感性、选择字段审计和运行时间测试。

## 2. 主结果

三个前向评估窗口共包含 **4,112 条路线、261 名司机、59,237 个配送任务**。过去数据选出的固定预算合计产生 **17,866 次询问，平均每条路线 4.345 次**。

| 策略 | 路线等权 MAE | 相对不询问的改善 | 每次被问任务的平均改善 |
|---|---:|---:|---:|
| 每路线固定配额 | {place.loc['uniform_DG','mae']:.3f} min | {place.loc['uniform_DG','system_gain']:.3f} min | {task.loc['uniform_DG','selected_gain_mean']:.2f} min |
| 同日跨路线 DG/规模分配 | {place.loc['daily_DG','mae']:.3f} min | {place.loc['daily_DG','system_gain']:.3f} min | {task.loc['daily_DG','selected_gain_mean']:.2f} min |

新策略相对不询问改善 **{total.difference:.3f} 分钟**，司机聚类 95% CI **[{total.ci_low:.3f}, {total.ci_high:.3f}]**；在完全相同询问量下，相对固定配额再改善 **{primary.difference:.3f} 分钟**，95% CI **[{primary.ci_low:.3f}, {primary.ci_high:.3f}]**。这就是可观测的工程增量，不再把所有收益摊薄成一个难以解释的 1.24%。

三个窗口的点估计分别为 **{window_effects.iloc[0].difference:.3f}、{window_effects.iloc[1].difference:.3f}、{window_effects.iloc[2].difference:.3f} 分钟**。第一个窗口单独的区间包含 0，后两个窗口排除 0；因此论文应把合并的预先定义前向结果作为主检验，并完整报告窗口异质性，不能声称每个窗口都单独显著。

按配送任务等权，MAE 从 **{alt_tasks.loc['baseline','mae']:.3f}** 降至 **{alt_tasks.loc['daily_DG','mae']:.3f}**；相对固定配额，路线内 RMSE 额外降低 1.345 分钟、90 分位绝对误差降低 2.737 分钟、超过 60 分钟的任务比例下降 1.031 个百分点，四个司机聚类区间均排除 0。

## 3. 为什么不是“集中询问碰巧有效”

在相同 17,866 次询问、相同每日批次约束、相同每路线最低覆盖和上限下：

| 分配依据 | 相对不询问的改善 |
|---|---:|
| DG/路线规模 | {place.loc['daily_DG','system_gain']:.3f} min |
| 原始 DG | {place.loc['daily_raw_DG','system_gain']:.3f} min |
| 只按当前误差 ER | {place.loc['daily_ER','system_gain']:.3f} min |
| 只按路线规模 | {place.loc['route_size_only','system_gain']:.3f} min |
| 批次内打乱 DG | {place.loc['permuted_DG','system_gain']:.3f} min |
| 随机分配 | {place.loc['random_capacity','system_gain']:.3f} min |

DG/规模相对 ER、路线规模、打乱 DG 和随机分配的额外收益均显著；这些识别性对照中最小的司机聚类下界仍为 **{identifying_place_ci.ci_low.min():.3f} 分钟**。DG/规模与原始 DG 的差异不显著，说明主要贡献来自可迁移的 DG 排序，而除以路线规模是用于解释和控制机会成本的稳健归一化，不应包装成单独的算法突破。即使允许 DG 和 ER 分别用过去数据选择预算和上限，DG 仍多改善 **{er_robustness.loc['separate_past_only_minimum_mae','dg_minus_er']:.3f} 分钟**，95% CI **[{er_robustness.loc['separate_past_only_minimum_mae','ci_low']:.3f}, {er_robustness.loc['separate_past_only_minimum_mae','ci_high']:.3f}]**。

## 4. 工程约束与收益分布

跨路线分配后，所有路线仍至少被询问一次；25.0% 路线只问 1 次，10.55% 路线达到 10 次上限，中位数为 4 次，工作量 Gini 为 {workload.loc['daily_DG','attempt_gini']:.3f}。每个发车批次的询问总数与固定配额逐批完全相等。

被询问任务中，平均改善 **{task.loc['daily_DG','selected_gain_mean']:.2f} 分钟**，中位数 **{task.loc['daily_DG','selected_gain_median']:.2f} 分钟**，有 **{task.loc['daily_DG','over_30_percent']:.1f}%** 的任务改善超过 30 分钟。仍有 **{task.loc['daily_DG','harm_percent']:.1f}%** 的任务出现负收益，这说明 DG 是群体层面的资源分配信号，并不能保证每次询问都获益；论文应诚实报告这一点。

特征预测加分配对每个约 1.9–2.1 万任务的窗口，中位耗时为 **{runtime.combined_median_ms.min():.0f}–{runtime.combined_median_ms.max():.0f} ms**，最慢窗口的 95 分位为 **{runtime.combined_p95_ms.max():.0f} ms**，具备发车前批处理的计算可行性。

## 5. 稳健性边界

- **较少响应类别：** K=5 明显强于 K=3。K=3 的干净场景仍有正增量，但在 20% 阶段误报或 50% 选择性拒答时总收益区间会触及 0；正文应把 K=5 作为推荐配置。
- **回答误差：** K=5 在 10% 和 20% 相邻阶段误报下，相对固定配额仍分别多改善 0.991 和 0.938 分钟，区间均排除 0。
- **拒答：** 即使 50% 独立拒答，仍多改善 0.522 分钟；最严的熵相关选择性拒答下，仍多改善 **{severe_vs_uniform.difference:.3f}** 分钟，95% CI **[{severe_vs_uniform.ci_low:.3f}, {severe_vs_uniform.ci_high:.3f}]**，相对不询问的总改善为 **{severe_vs_zero.difference:.3f}** 分钟，95% CI **[{severe_vs_zero.ci_low:.3f}, {severe_vs_zero.ci_high:.3f}]**。
- **不要求全车队集中调度：** 每波只协调 2 条路线就比固定配额多改善 **{wave2.difference:.3f}** 分钟，95% CI **[{wave2.ci_low:.3f}, {wave2.ci_high:.3f}]**；10–20 条路线后收益接近饱和。
- **重建组规模：** 共 {int(cohort.cohorts)} 个组，路线数中位数 {cohort.routes_median:.0f}，四分位区间 {cohort.routes_p25:.1f}--{cohort.routes_p75:.0f}，范围 {cohort.routes_min:.0f}--{cohort.routes_max:.0f}。由小到大的四个规模组增量为 {', '.join(f'{v:.3f}' for v in cohort_slices.improvement)} 分钟，最大组的点估计反而最小。
- **日期重构：** 起始日、结束日、众数日期及剔除所有跨日路线时，额外改善介于 **{anchors.difference.min():.3f}–{anchors.difference.max():.3f} 分钟**，所有区间排除 0。
- **选择审计：** 打乱全部结果字段后选择集合不变；改变输入行顺序后选择集合也不变。这证明算法没有偷看结果，但不能证明重建组是真实共同派发组。

## 6. 与完整计划信息的关系

公开数据中的特权“完整计划”基线 MAE 为 **{p.full_plan_mae:.3f}**。固定过去预算的跨路线策略为 **{p.policy_mae:.3f}**，点估计好 {p.policy_minus_full_plan_gain:.3f} 分钟，但差异区间 **[{p.ci_low:.3f}, {p.ci_high:.3f}]** 包含 0。因此可写成：**有限、选择性的粗粒度计划确认在该离线数据上达到与完整计划特征基线统计上不可区分的精度。** 不能写成“显著击败完整计划”。

这一结果的业务意义是，全面路线数字化不是获得大部分预测收益的唯一工程路径。当前数据没有采集系统建设成本，因此“成本低一个数量级”只能作为待测假设，不能作为实证结论。

## 7. 响应感知方法的证据等级

简单的 `响应概率 × DG` 没有改善最终收益，不能作为推荐方法。直接学习 `E[R × 实际降误差 | X]` 则在冻结的选择性拒答场景中成立：K=5、过去数据选择的最小 MAE配置下，收益为 **{d5.loc['direct_response_gain','system_gain']:.3f} 分钟**，而同配置普通 DG 为 **{d5.loc['plain_DG_same_config','system_gain']:.3f} 分钟**；差值 **{d5ci.loc['direct_response_gain_minus_plain_DG_same_config','difference']:.3f} 分钟**，95% CI **[{d5ci.loc['direct_response_gain_minus_plain_DG_same_config','ci_low']:.3f}, {d5ci.loc['direct_response_gain_minus_plain_DG_same_config','ci_high']:.3f}]**。

这只支持一个待外部验证的方法假设：当历史数据包含真实“是否回答”标签时，可以比较直接学习可实现收益与分别预测响应概率和信息价值。当前响应机制是确定性合成规则，因此该实验降为附录方法检查，不作为主要贡献或司机行为证据。

## 8. 论文可安全使用的贡献表述

1. 将部分计划可见性从单任务特征获取问题重新表述为**可共享容量路线组内、有限人工确认容量的跨路线分配问题**。
2. 给出一个可部署的受约束分配规则：过去数据选择预算，每路线最低覆盖，单路线封顶，同批次总工作量严格守恒。
3. 用三个严格前向窗口证明，相同询问量下的跨路线分配对 MAE、RMSE、尾部误差和严重迟差比例均产生可观测增量，并通过信息打乱、随机集中和路线规模安慰剂排除替代解释。

合成拒答实验仅作为附录中的方法检查，不列为第四项主要贡献。

## 9. 不能越界的表述

- 不能声称已经验证真实司机愿意、能够按要求报告意图阶段；代理答案来自软件计划阶段。
- 不能声称每次询问必然改善 ETA；约 41.8% 的被问任务离线反事实收益为负。
- 不能声称显著击败完整计划基线；只能说统计上不可区分且点估计更好。
- 不能声称 DG/路线规模是普适最优算法；实证支持的是这个场景中的受约束资源分配机制。
- 不能把合成选择性拒答结果写成真实人的响应规律。

## 10. 实验是否已经跑完

针对当前公开数据能够验证的核心证据链，实验已经跑完：预算扫描、K=3/5、固定配额、跨路线分配、过去窗口调参、安慰剂、ER 同预算与独立调参对照、替代指标、国家和重建组切片、阶段误报、独立与选择性拒答、调度波次、日期锚点、选择字段审计、完整计划对照和运行时间均已完成。

论文已据此重构。后续若获得记录的真实派发关系和司机回答日志，应作为外部验证新增，而不应继续用当前数据制造更多相似的离线表格。
"""
    (OUT / "写论文前的最终实验冻结报告.md").write_text(report, encoding="utf8")


def write_manifest(checks):
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    manifest_rows = []
    generated_metadata = {"SHA256SUMS.csv", "VALIDATION_COMPLETE.json"}
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in generated_metadata):
        data = path.read_bytes()
        manifest_rows.append({
            "path": path.relative_to(OUT).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    pd.DataFrame(manifest_rows).to_csv(OUT / "SHA256SUMS.csv", index=False)
    complete = {
        "status": "complete",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [
            "operational daily-batch allocation",
            "past-only budget and cap selection",
            "matched-workload placebos",
            "post-review DG-versus-ER and reconstructed-cohort-size audit",
            "alternative metrics and slices",
            "DG forward ranking calibration",
            "answer noise and refusal stress tests",
            "direct response-gain learning",
            "dispatch-wave sensitivity",
            "dispatch-date anchor sensitivity",
            "selection invariance and exact-capacity audit",
            "privileged full-plan comparison",
            "runtime benchmark",
        ],
        "integrity_checks_passed": sum(x["passed"] for x in checks),
        "integrity_checks_total": len(checks),
        "files_in_manifest": len(manifest_rows),
    }
    (OUT / "VALIDATION_COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf8"
    )

    return OUT / "SHA256SUMS.csv"


def main():
    checks = integrity_checks()
    make_figures()
    write_report()
    manifest_path = write_manifest(checks)
    print(json.dumps({"complete": True, "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
