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
            "DG-versus-ER and reconstructed-cohort-size robustness audit",
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
    manifest_path = write_manifest(checks)
    print(json.dumps({"complete": True, "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
