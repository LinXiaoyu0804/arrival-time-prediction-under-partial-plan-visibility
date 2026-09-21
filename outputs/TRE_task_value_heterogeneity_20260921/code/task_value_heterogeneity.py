"""Exploratory heterogeneity audit for task-level confirmation value.

This analysis is post-primary and descriptive. It uses only information that
is observable before a confirmation is requested to define operating groups,
then evaluates the proxy-stage error reduction for every eligible task.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
BUDGET = ROOT / "outputs/TRE_query_budget_validation_20260913"
PRIMARY = ROOT / "outputs/TRE_operational_batch_validation_20260913"
OUT = ROOT / "outputs/TRE_task_value_heterogeneity_20260921"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
for folder in (TABLES, FIGURES, OUT / "code"):
    folder.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
import operational_batch_validation as op
import query_budget_frontier as qb


SEED = 20260921
REPS = 2000
PROTOCOL = {
    "date": "2026-09-21",
    "status": (
        "Post-primary exploratory analysis prompted after the primary allocation results and after "
        "descriptive inspection. It does not alter the primary endpoint, models, budgets, or policy."
    ),
    "question": (
        "Which pre-query task and route contexts have higher conditional proxy-confirmation value, "
        "and how often do individual routes improve or worsen when capacity is reallocated?"
    ),
    "estimand": "base absolute error minus proxy-informed absolute error for every eligible task",
    "weights": "inverse route size, normalized within each reported group",
    "inference": "2,000 paired driver-cluster bootstrap resamples",
    "primary_descriptive_groups": {
        "route_size": ["10-12 tasks", "13-17 tasks", "18 or more tasks"],
        "time_window_width": ["0-240 minutes", "241-420 minutes", "more than 420 minutes"],
    },
    "interpretation_limit": (
        "The estimand is the conditional value of disclosing a recorded software-plan stage. It is "
        "not a causal estimate of a live driver's response or of customer-level business value."
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def bootstrap_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    codes, drivers = pd.factorize(frame["Driver ID"], sort=True)
    rng = np.random.default_rng(SEED)
    multiplicities = rng.multinomial(
        len(drivers), np.ones(len(drivers), dtype=float) / len(drivers), size=REPS
    )
    return codes, multiplicities


def group_draws(
    frame: pd.DataFrame,
    codes: np.ndarray,
    multiplicities: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    n_drivers = multiplicities.shape[1]
    weight = frame["route_weight"].to_numpy(dtype=float)
    gain = frame["clean_gain"].to_numpy(dtype=float)
    numer = np.bincount(codes[mask], weights=(weight * gain)[mask], minlength=n_drivers)
    denom = np.bincount(codes[mask], weights=weight[mask], minlength=n_drivers)
    return (multiplicities @ numer) / (multiplicities @ denom)


def load_tasks() -> pd.DataFrame:
    choices = pd.read_csv(BUDGET / "tables/past_only_budget_choices.csv")
    choices = choices[
        choices.k.eq(5)
        & choices.method.eq("DG")
        & choices.condition.eq("proxy_clean")
        & choices.rule.eq("minimum_mae")
    ]
    parts = []
    for origin in (13, 19, 25):
        q = pd.read_parquet(SCENARIO / "cache" / f"w{origin}_k5_evaluation.parquet")
        threshold = json.loads(
            (SCENARIO / f"w{origin}_k5_freeze.json").read_text(encoding="utf8")
        )["refusal_thresholds"]["0.5"]
        q = op.add_fields(q, threshold).reset_index(drop=True)
        q["DG_score"] = q["gain_score"]
        budget = int(choices.loc[choices.window.eq(origin), "chosen_budget"].iloc[0])
        q["fixed_query"] = qb.deterministic_selection(q, "DG_score", budget)
        q["allocation_query"] = op.daily_batch_selection(
            q, q.DG_score / q.route_size, budget, cap=10, minimum=1
        )
        q["window"] = origin
        q["route_weight"] = 1.0 / q["route_size"]
        parts.append(q)
    tasks = pd.concat(parts, ignore_index=True)
    tasks["route_group"] = pd.cut(
        tasks.route_size,
        [-np.inf, 12, 17, np.inf],
        labels=["10--12", "13--17", "$\\geq$18"],
    )
    tasks["window_width_group"] = pd.cut(
        tasks.time_window_width,
        [-np.inf, 240, 420, np.inf],
        labels=["$\\leq$240", "241--420", "$>$420"],
    )
    tasks["position_group"] = pd.cut(
        tasks.seq_rank_norm,
        [-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
    )
    tasks["driver_history_group"] = pd.cut(
        np.expm1(tasks.driver_hist_count_log),
        [-np.inf, 20, 100, np.inf],
        labels=["limited", "moderate", "experienced"],
    )
    tasks["address_history_group"] = pd.cut(
        np.expm1(tasks.address_hist_count_log),
        [-np.inf, 2, 10, np.inf],
        labels=["limited", "moderate", "familiar"],
    )
    tasks["entropy_group"] = pd.cut(
        tasks.entropy,
        [-np.inf, 0.75, 0.90, np.inf],
        labels=["low", "medium", "high"],
    )
    return tasks


def task_profiles(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = [
        ("route_size", "Route deliveries", "route_group"),
        ("time_window_width", "Time-window width (min)", "window_width_group"),
        ("predicted_position", "Predicted route position", "position_group"),
        ("driver_history", "Driver history", "driver_history_group"),
        ("address_history", "Address history", "address_history_group"),
        ("stage_entropy", "Stage uncertainty", "entropy_group"),
    ]
    codes, multiplicities = bootstrap_matrix(tasks)
    rows = []
    draw_lookup: dict[tuple[str, str], np.ndarray] = {}
    for dimension, label, column in definitions:
        categories = list(tasks[column].cat.categories)
        for order, group in enumerate(categories, start=1):
            mask = tasks[column].astype(str).eq(str(group)).to_numpy()
            z = tasks.loc[mask]
            draws = group_draws(tasks, codes, multiplicities, mask)
            draw_lookup[(dimension, str(group))] = draws
            row = {
                "dimension": dimension,
                "dimension_label": label,
                "group_order": order,
                "group": str(group),
                "tasks": len(z),
                "routes": z["Route ID"].nunique(),
                "drivers": z["Driver ID"].nunique(),
                "route_balanced_mean_gain": weighted_mean(z.clean_gain, z.route_weight),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "harm_percent": 100 * weighted_mean(z.clean_gain.lt(0), z.route_weight),
                "severe_harm_percent": 100 * weighted_mean(z.clean_gain.lt(-30), z.route_weight),
                "fixed_query_percent": 100 * weighted_mean(z.fixed_query, z.route_weight),
                "allocation_query_percent": 100 * weighted_mean(z.allocation_query, z.route_weight),
            }
            for window in (13, 19, 25):
                zw = z[z.window.eq(window)]
                row[f"window_{window}_gain"] = weighted_mean(zw.clean_gain, zw.route_weight)
            rows.append(row)
    profiles = pd.DataFrame(rows)

    contrast_specs = [
        ("short_minus_long_routes", "route_size", "10--12", "$\\geq$18"),
        ("moderate_minus_narrow_windows", "time_window_width", "241--420", "$\\leq$240"),
        ("late_minus_early_position", "predicted_position", "late", "early"),
        ("high_minus_low_stage_uncertainty", "stage_entropy", "high", "low"),
    ]
    contrast_rows = []
    for name, dimension, favorable, reference in contrast_specs:
        draws = draw_lookup[(dimension, favorable)] - draw_lookup[(dimension, reference)]
        point = (
            profiles.loc[(profiles.dimension.eq(dimension)) & (profiles.group.eq(favorable)), "route_balanced_mean_gain"].iloc[0]
            - profiles.loc[(profiles.dimension.eq(dimension)) & (profiles.group.eq(reference)), "route_balanced_mean_gain"].iloc[0]
        )
        row = {
            "contrast": name,
            "favorable_group": favorable,
            "reference_group": reference,
            "difference": point,
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
        }
        for window in (13, 19, 25):
            fav = profiles.loc[(profiles.dimension.eq(dimension)) & (profiles.group.eq(favorable)), f"window_{window}_gain"].iloc[0]
            ref = profiles.loc[(profiles.dimension.eq(dimension)) & (profiles.group.eq(reference)), f"window_{window}_gain"].iloc[0]
            row[f"window_{window}_difference"] = fav - ref
        contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)
    return profiles, contrasts


def route_reallocation() -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.read_parquet(PRIMARY / "tables/final_audit_route_results.parquet")
    work = pd.read_parquet(PRIMARY / "tables/final_audit_route_workloads.parquet")
    keys = ["window", "Route ID"]
    daily = routes[routes.allocation.eq("daily_DG")].set_index(keys)
    fixed = routes[routes.allocation.eq("uniform_DG")].set_index(keys)
    wd = work[work.allocation.eq("daily_DG")].set_index(keys)
    wf = work[work.allocation.eq("uniform_DG")].set_index(keys)
    detail = pd.DataFrame(
        {
            "driver": daily.driver,
            "route_size": daily.route_size,
            "allocation_attempts": wd.attempts,
            "fixed_attempts": wf.attempts,
            "delta_attempts": wd.attempts - wf.attempts,
            "allocation_gain": daily.system_gain,
            "fixed_gain": fixed.system_gain,
            "incremental_gain": daily.system_gain - fixed.system_gain,
        }
    ).reset_index()
    detail["attempt_change"] = np.select(
        [detail.delta_attempts.lt(0), detail.delta_attempts.eq(0)],
        ["fewer", "same"],
        default="more",
    )
    rows = []
    for group, z in detail.groupby("attempt_change", sort=False):
        rows.append(
            {
                "attempt_change": group,
                "routes": len(z),
                "route_percent": 100 * len(z) / len(detail),
                "mean_delta_attempts": z.delta_attempts.mean(),
                "mean_incremental_gain": z.incremental_gain.mean(),
                "median_incremental_gain": z.incremental_gain.median(),
                "improved_percent": 100 * z.incremental_gain.gt(0).mean(),
                "harmed_percent": 100 * z.incremental_gain.lt(0).mean(),
                "over_5_minute_harm_percent": 100 * z.incremental_gain.lt(-5).mean(),
            }
        )
    overall = {
        "attempt_change": "overall",
        "routes": len(detail),
        "route_percent": 100.0,
        "mean_delta_attempts": 0.0,
        "mean_incremental_gain": detail.incremental_gain.mean(),
        "median_incremental_gain": detail.incremental_gain.median(),
        "improved_percent": 100 * detail.incremental_gain.gt(0).mean(),
        "harmed_percent": 100 * detail.incremental_gain.lt(0).mean(),
        "over_5_minute_harm_percent": 100 * detail.incremental_gain.lt(-5).mean(),
        "p10_incremental_gain": detail.incremental_gain.quantile(0.10),
        "p90_incremental_gain": detail.incremental_gain.quantile(0.90),
        "unchanged_percent": 100 * detail.incremental_gain.eq(0).mean(),
    }
    summary = pd.concat([pd.DataFrame(rows), pd.DataFrame([overall])], ignore_index=True)
    return detail, summary


def make_main_table(profiles: pd.DataFrame, destination: Path) -> None:
    main = profiles[
        profiles.dimension.isin(["time_window_width", "predicted_position", "stage_entropy"])
    ].copy()
    lines = [
        r"\begin{table}[tb]\centering\small",
        r"\caption{Exploratory value profiles from pre-query operating context.}\label{tab:valueprofiles}",
        r"\begin{tabular}{llrrrr}\toprule",
        r"Context & Group & Tasks & Fixed (\%) & Alloc. (\%) & Gain [95\% CI] \\\midrule",
    ]
    for dimension, z in main.groupby("dimension", sort=False):
        z = z.sort_values("group_order")
        for j, row in enumerate(z.itertuples(index=False)):
            context = row.dimension_label if j == 0 else ""
            lines.append(
                f"{context} & {row.group} & {row.tasks:,} & {row.fixed_query_percent:.1f} & "
                f"{row.allocation_query_percent:.1f} & {row.route_balanced_mean_gain:.2f} "
                f"[{row.ci_low:.2f}, {row.ci_high:.2f}] \\\\"
            )
        if dimension != "stage_entropy":
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\begin{minipage}{0.97\linewidth}\footnotesize Gain is the inverse-route-size-weighted reduction in absolute error if the recorded five-stage proxy answer is disclosed to every task in the group. Fixed and Alloc. are route-balanced query rates under fixed quotas and cohort allocation. Groups use only information observable before querying. The analysis is post-primary and descriptive.\end{minipage}",
            r"\end{table}",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf8")


def make_figure(profiles: pd.DataFrame, route_summary: pd.DataFrame, destination: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.35), constrained_layout=True)
    axes = axes.ravel()
    blue, orange, gray = "#2F6690", "#D9822B", "#6B7280"
    panels = [
        (axes[0], "time_window_width", "Time-window width (min)"),
        (axes[1], "predicted_position", "Predicted route position"),
        (axes[2], "stage_entropy", "Stage uncertainty"),
    ]
    for ax, dimension, title in panels:
        z = profiles[profiles.dimension.eq(dimension)].sort_values("group_order")
        x = np.arange(len(z))
        y = z.route_balanced_mean_gain.to_numpy()
        err = np.vstack([y - z.ci_low.to_numpy(), z.ci_high.to_numpy() - y])
        ax.errorbar(x, y, yerr=err, fmt="o", color=blue, capsize=3, lw=1.4)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x, z.group)
        ax.set_title(title)
        ax.set_ylabel("Gain per confirmation (min)" if ax in (axes[0], axes[2]) else "")
        ax.grid(axis="y", alpha=0.2)

    ax = axes[3]
    overall = route_summary[route_summary.attempt_change.eq("overall")].iloc[0]
    values = [overall.improved_percent, overall.harmed_percent, overall.unchanged_percent]
    bars = ax.bar([0, 1, 2], values, color=[blue, orange, gray], width=0.68)
    ax.set_xticks([0, 1, 2], ["Improved", "Worsened", "Unchanged"], rotation=15)
    ax.set_ylabel("Routes (\%)")
    ax.set_title("Allocation versus fixed quota")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (OUT / "protocol.json").write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    tasks = load_tasks()
    profiles, contrasts = task_profiles(tasks)
    route_detail, route_summary = route_reallocation()
    profiles.to_csv(TABLES / "task_value_profiles.csv", index=False)
    contrasts.to_csv(TABLES / "task_value_contrasts.csv", index=False)
    route_detail.to_parquet(TABLES / "route_reallocation_detail.parquet", index=False)
    route_summary.to_csv(TABLES / "route_reallocation_summary.csv", index=False)
    make_main_table(profiles, TABLES / "task_value_profiles.tex")
    make_figure(profiles, route_summary, FIGURES / "task_value_heterogeneity.pdf")
    completion = {
        "rows": len(tasks),
        "routes": int(tasks["Route ID"].nunique()),
        "drivers": int(tasks["Driver ID"].nunique()),
        "outputs": {},
    }
    for path in sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "completion.json"]):
        completion["outputs"][path.relative_to(OUT).as_posix()] = sha256(path)
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf8")
    print(profiles[profiles.dimension.isin(["route_size", "time_window_width"])].to_string(index=False))
    print(contrasts.to_string(index=False))
    print(route_summary.to_string(index=False))


if __name__ == "__main__":
    main()
