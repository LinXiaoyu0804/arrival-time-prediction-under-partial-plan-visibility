"""Additional robustness audit of cohort size and DG-versus-ER comparisons.

This analysis recombines archived route-level outputs without refitting a
model or altering the primary specification.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/TRE_operational_batch_validation_20260913"
TABLES = OUT / "tables"


def driver_bootstrap_difference(left, right, reps=2000, seed=20260913):
    keys = ["window", "Route ID"]
    paired = left[keys + ["driver", "system_gain"]].merge(
        right[keys + ["system_gain"]],
        on=keys,
        suffixes=("_dg", "_er"),
        validate="one_to_one",
    )
    paired["difference"] = paired.system_gain_dg - paired.system_gain_er
    codes, drivers = pd.factorize(paired.driver)
    counts = np.bincount(codes)
    totals = np.bincount(codes, weights=paired.difference)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(drivers), np.ones(len(drivers)) / len(drivers), size=reps
    )
    draws = (multiplicities @ totals) / (multiplicities @ counts)
    return {
        "dg_minus_er": paired.difference.mean(),
        "ci_low": np.quantile(draws, 0.025),
        "ci_high": np.quantile(draws, 0.975),
        "routes": len(paired),
        "drivers": len(drivers),
    }


def comparison_row(label, dg, er):
    stats = driver_bootstrap_difference(dg, er)
    return {
        "comparison": label,
        "dg_queries_per_route": dg.attempts.mean(),
        "er_queries_per_route": er.attempts.mean(),
        "dg_mae": dg.mae.mean(),
        "er_mae": er.mae.mean(),
        **stats,
    }


def main():
    capacity = pd.read_csv(TABLES / "final_audit_batch_capacity.csv")
    routes = capacity.routes.astype(float)
    cohort_summary = pd.DataFrame(
        [
            {
                "cohorts": len(capacity),
                "routes_min": routes.min(),
                "routes_p25": routes.quantile(0.25),
                "routes_median": routes.median(),
                "routes_p75": routes.quantile(0.75),
                "routes_max": routes.max(),
            }
        ]
    )
    cohort_summary.to_csv(TABLES / "reconstructed_cohort_size_summary.csv", index=False)

    fixed = pd.read_parquet(TABLES / "fixed_grid_route_results.parquet")
    fixed = fixed[
        fixed.split.eq("evaluation")
        & fixed.condition.eq("proxy_clean")
        & fixed.k.eq(5)
        & fixed.allocation.eq("daily_objective_cap10_min1")
    ]

    same_budget_rows = []
    for budget in [1, 2, 3, 5, 10]:
        group = fixed[fixed.budget.eq(str(budget))]
        same_budget_rows.append(
            {
                "budget": budget,
                **comparison_row(
                    f"same_budget_{budget}",
                    group[group.method.eq("DG")],
                    group[group.method.eq("ER")],
                ),
            }
        )
    pd.DataFrame(same_budget_rows).to_csv(
        TABLES / "dg_er_same_budget_comparisons.csv", index=False
    )

    primary_parts = []
    for window, budget in [(13, "3"), (19, "5"), (25, "5")]:
        primary_parts.append(fixed[fixed.window.eq(window) & fixed.budget.eq(budget)])
    primary = pd.concat(primary_parts, ignore_index=True)
    robustness_rows = [
        comparison_row(
            "primary_same_allocator_and_workload",
            primary[primary.method.eq("DG")],
            primary[primary.method.eq("ER")],
        )
    ]

    joint = pd.read_parquet(TABLES / "joint_past_only_route_results.parquet")
    joint = joint[joint.condition.eq("proxy_clean") & joint.k.eq(5)]
    for rule in ["minimum_mae", "knee90"]:
        group = joint[joint.rule.eq(rule)]
        robustness_rows.append(
            comparison_row(
                f"separate_past_only_{rule}",
                group[group.method.eq("DG")],
                group[group.method.eq("ER")],
            )
        )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(TABLES / "dg_er_robustness.csv", index=False)

    checks = {
        "status": "complete",
        "audit_type": "additional_robustness",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_refits": 0,
        "primary_specification_changed": False,
        "cohort_count": int(cohort_summary.iloc[0].cohorts),
        "all_intermediate_budget_dg_er_intervals_positive": bool(
            pd.DataFrame(same_budget_rows)
            .query("budget in [2, 3, 5]")
            .ci_low.gt(0)
            .all()
        ),
        "separately_tuned_minimum_dg_er_interval_positive": bool(
            robustness.loc[
                robustness.comparison.eq("separate_past_only_minimum_mae"), "ci_low"
            ].iloc[0]
            > 0
        ),
    }
    (OUT / "ADDITIONAL_ROBUSTNESS_AUDIT_COMPLETE.json").write_text(
        json.dumps(checks, indent=2), encoding="utf8"
    )
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
