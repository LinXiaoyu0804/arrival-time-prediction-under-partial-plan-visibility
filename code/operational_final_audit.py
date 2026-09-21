import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
BUDGET_OUT = ROOT / "outputs/TRE_query_budget_validation_20260913"
OUT = ROOT / "outputs/TRE_operational_batch_validation_20260913"
for folder in ["tables", "code"]:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
import query_budget_frontier as qb
import route_adaptive_allocation as ra
import operational_batch_validation as op


PROTOCOL = {
    "date": "2026-09-14",
    "status": "Specified before inspecting the final-audit outputs.",
    "policy": (
        "K=5, clean proxy answers, DG score divided by route size, minimum one and maximum ten "
        "queries per route, with the query budget selected from prior data only."
    ),
    "questions": [
        "Where is the gain concentrated and what is the workload distribution?",
        "Does each dispatch batch use exactly the same number of attempts as the uniform comparator?",
        "Is task selection invariant to outcome-field perturbation and row order?",
        "How does the operational policy compare with privileged full-plan visibility?",
        "Is the allocator fast enough to run at dispatch time?",
    ],
    "statistics": "Paired driver-cluster bootstrap, 2,000 resamples, seed 20260914.",
    "runtime_scope": "Already-computed feature matrix to DG prediction, plus constrained allocation.",
    "selection_inputs": [
        "dispatch batch identifier",
        "route identifier",
        "predicted DG score",
        "route size",
        "deterministic tie breaker",
        "past-only budget",
        "minimum one query and maximum ten queries per route",
    ],
    "outcome_fields_excluded": [
        "target_minutes",
        "planned_phase",
        "actual_phase",
        "base_error",
        "proxy_error",
        "oracle_error",
        "clean_gain",
        "responds",
        "response_gain",
    ],
    "batch_reconstruction_note": (
        "The public data omit an explicit dispatch date. Country-week-weekday batch labels were "
        "reconstructed from the first executed stop of each route. A deployment observes dispatch "
        "batch membership directly; no arrival or outcome value enters the allocation after the "
        "batch label has been assigned."
    ),
}


def freeze_protocol():
    path = OUT / "operational_final_audit_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def load_eval(origin):
    freeze = json.loads((SCENARIO / f"w{origin}_k5_freeze.json").read_text(encoding="utf8"))
    threshold = freeze["refusal_thresholds"]["0.5"]
    return op.add_fields(ra.load_window(origin, 5), threshold).reset_index(drop=True)


def budget_choices():
    choices = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    return choices[
        choices.k.eq(5)
        & choices.method.eq("DG")
        & choices.condition.eq("proxy_clean")
        & choices.rule.eq("minimum_mae")
    ].sort_values("window")


def bootstrap_gain_interval(route_rows, reps=2000, seed=20260914):
    codes, drivers = pd.factorize(route_rows.driver, sort=True)
    counts = np.bincount(codes)
    totals = np.bincount(codes, weights=route_rows.system_gain)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(drivers), np.ones(len(drivers)) / len(drivers), size=reps)
    draws = (multiplicities @ totals) / (multiplicities @ counts)
    return {
        "system_gain": route_rows.system_gain.mean(),
        "ci_low": np.quantile(draws, 0.025),
        "ci_high": np.quantile(draws, 0.975),
        "routes": len(route_rows),
        "drivers": len(drivers),
    }


def selected_task_and_workload_audit():
    task_rows = []
    workload_rows = []
    batch_rows = []
    invariant_rows = []
    runtime_rows = []
    route_rows = []

    for choice in budget_choices().itertuples(index=False):
        origin = int(choice.window)
        budget = int(choice.chosen_budget)
        q = load_eval(origin)
        uniform = qb.deterministic_selection(q, "DG_score", budget)
        priority = q.DG_score.to_numpy() / q.route_size.to_numpy()
        daily = op.daily_batch_selection(q, priority, budget, cap=10, minimum=1)
        assert int(uniform.sum()) == int(daily.sum())

        for label, selected in [("uniform_DG", uniform), ("daily_DG", daily)]:
            chosen = q.loc[selected, [
                "Route ID", "Driver ID", "_rowkey", "clean_gain", "DG_score", "route_size"
            ]].copy()
            chosen["window"] = origin
            chosen["allocation"] = label
            chosen["priority"] = chosen.DG_score / chosen.route_size
            chosen["within_route_priority_rank"] = (
                chosen.groupby("Route ID")["priority"].rank(method="first", ascending=False).astype(int)
            )
            task_rows.append(chosen)

            wr = q[["Route ID", "Driver ID", "route_size"]].drop_duplicates("Route ID").copy()
            counts = q.assign(_selected=selected).groupby("Route ID")._selected.sum()
            wr["attempts"] = wr["Route ID"].map(counts).astype(int)
            wr["window"] = origin
            wr["allocation"] = label
            workload_rows.append(wr)

        b = q.assign(uniform=uniform, daily=daily).groupby(op.BATCH_COLUMNS).agg(
            routes=("Route ID", "nunique"),
            tasks=("Route ID", "size"),
            uniform_attempts=("uniform", "sum"),
            daily_attempts=("daily", "sum"),
        ).reset_index()
        b["window"] = origin
        b["attempt_difference"] = b.daily_attempts - b.uniform_attempts
        batch_rows.append(b)
        assert b.attempt_difference.eq(0).all()

        # Holding operational batch membership and scores fixed, perturb every outcome-only field.
        changed = q.copy()
        rng = np.random.default_rng(20260914 + origin)
        for column in PROTOCOL["outcome_fields_excluded"]:
            if column in changed:
                changed[column] = rng.permutation(changed[column].to_numpy())
        daily_after_outcome_change = op.daily_batch_selection(
            changed, changed.DG_score.to_numpy() / changed.route_size.to_numpy(), budget, cap=10, minimum=1
        )
        outcome_invariant = np.array_equal(daily, daily_after_outcome_change)

        shuffled = q.sample(frac=1, random_state=20260914 + origin).reset_index(drop=True)
        shuffled_selected = op.daily_batch_selection(
            shuffled,
            shuffled.DG_score.to_numpy() / shuffled.route_size.to_numpy(),
            budget,
            cap=10,
            minimum=1,
        )
        original_keys = set(q.loc[daily, "_rowkey"].astype(str))
        shuffled_keys = set(shuffled.loc[shuffled_selected, "_rowkey"].astype(str))
        row_order_invariant = original_keys == shuffled_keys
        invariant_rows.append(
            {
                "window": origin,
                "tasks": len(q),
                "selected": int(daily.sum()),
                "outcome_field_invariant": outcome_invariant,
                "row_order_invariant": row_order_invariant,
                "selected_set_sha256": hashlib.sha256(
                    "\n".join(sorted(original_keys)).encode("utf8")
                ).hexdigest(),
            }
        )
        assert outcome_invariant and row_order_invariant

        model_bundle = joblib.load(BUDGET_OUT / f"models/w{origin}_k5_temporary_scores.joblib")
        features = model_bundle["features"]
        prediction_ms = []
        allocation_ms = []
        combined_ms = []
        for _ in range(20):
            start = time.perf_counter()
            scores = model_bundle["DG"].predict(q[features])
            middle = time.perf_counter()
            _ = op.daily_batch_selection(q, scores / q.route_size.to_numpy(), budget, cap=10, minimum=1)
            end = time.perf_counter()
            prediction_ms.append(1000 * (middle - start))
            allocation_ms.append(1000 * (end - middle))
            combined_ms.append(1000 * (end - start))
        runtime_rows.append(
            {
                "window": origin,
                "tasks": len(q),
                "routes": q["Route ID"].nunique(),
                "batches": q[op.BATCH_COLUMNS].drop_duplicates().shape[0],
                "prediction_median_ms": np.median(prediction_ms),
                "allocation_median_ms": np.median(allocation_ms),
                "combined_median_ms": np.median(combined_ms),
                "combined_p95_ms": np.quantile(combined_ms, 0.95),
                "tasks_per_second_median": len(q) / (np.median(combined_ms) / 1000),
            }
        )

        for label, selected in [("uniform_DG", uniform), ("daily_DG", daily)]:
            route_rows.append(
                ra.make_route_rows(q, selected, origin, 5, "DG", budget, label, "proxy_clean")
            )
        print(time.strftime("%H:%M:%S"), "final audit", origin, flush=True)

    tasks = pd.concat(task_rows, ignore_index=True)
    workloads = pd.concat(workload_rows, ignore_index=True)
    batches = pd.concat(batch_rows, ignore_index=True)
    invariants = pd.DataFrame(invariant_rows)
    runtimes = pd.DataFrame(runtime_rows)
    routes = pd.concat(route_rows, ignore_index=True)

    task_summary = tasks.groupby("allocation").agg(
        attempts=("clean_gain", "size"),
        selected_gain_mean=("clean_gain", "mean"),
        selected_gain_median=("clean_gain", "median"),
        selected_gain_p10=("clean_gain", lambda x: np.quantile(x, 0.10)),
        selected_gain_p90=("clean_gain", lambda x: np.quantile(x, 0.90)),
        positive_percent=("clean_gain", lambda x: 100 * x.gt(0).mean()),
        harm_percent=("clean_gain", lambda x: 100 * x.lt(0).mean()),
        over_30_percent=("clean_gain", lambda x: 100 * x.gt(30).mean()),
        under_minus_30_percent=("clean_gain", lambda x: 100 * x.lt(-30).mean()),
    ).reset_index()
    workload_summary = workloads.groupby("allocation").agg(
        routes=("Route ID", "size"),
        mean_attempts=("attempts", "mean"),
        min_attempts=("attempts", "min"),
        p10_attempts=("attempts", lambda x: np.quantile(x, 0.10)),
        median_attempts=("attempts", "median"),
        p90_attempts=("attempts", lambda x: np.quantile(x, 0.90)),
        p95_attempts=("attempts", lambda x: np.quantile(x, 0.95)),
        max_attempts=("attempts", "max"),
        at_cap_percent=("attempts", lambda x: 100 * x.ge(10).mean()),
        one_query_percent=("attempts", lambda x: 100 * x.eq(1).mean()),
        attempt_gini=("attempts", ra.gini),
    ).reset_index()

    tasks.to_parquet(OUT / "tables/final_audit_selected_tasks.parquet", index=False)
    workloads.to_parquet(OUT / "tables/final_audit_route_workloads.parquet", index=False)
    routes.to_parquet(OUT / "tables/final_audit_route_results.parquet", index=False)
    task_summary.to_csv(OUT / "tables/final_audit_selected_gain_summary.csv", index=False)
    workload_summary.to_csv(OUT / "tables/final_audit_workload_summary.csv", index=False)
    tasks.groupby(["allocation", "within_route_priority_rank"]).agg(
        attempts=("clean_gain", "size"),
        mean_gain=("clean_gain", "mean"),
        median_gain=("clean_gain", "median"),
        positive_percent=("clean_gain", lambda x: 100 * x.gt(0).mean()),
        harm_percent=("clean_gain", lambda x: 100 * x.lt(0).mean()),
        over_30_percent=("clean_gain", lambda x: 100 * x.gt(30).mean()),
    ).reset_index().to_csv(OUT / "tables/final_audit_marginal_query_ordinal.csv", index=False)
    batches.to_csv(OUT / "tables/final_audit_batch_capacity.csv", index=False)
    invariants.to_csv(OUT / "tables/final_audit_selection_invariance.csv", index=False)
    runtimes.to_csv(OUT / "tables/final_audit_runtime.csv", index=False)

    comparison_rows = []
    for window, group in routes.groupby("window"):
        daily = group[group.allocation.eq("daily_DG")]
        uniform = group[group.allocation.eq("uniform_DG")]
        comparison_rows.append({
            "scope": f"window_{window}",
            "comparison": "daily_DG_minus_uniform_DG",
            **ra.driver_bootstrap_difference(daily, uniform, seed=20260914 + int(window)),
        })
    daily = routes[routes.allocation.eq("daily_DG")]
    uniform = routes[routes.allocation.eq("uniform_DG")]
    comparison_rows.append({
        "scope": "pooled",
        "comparison": "daily_DG_minus_uniform_DG",
        **ra.driver_bootstrap_difference(daily, uniform, seed=20260914),
    })
    gain_ci = bootstrap_gain_interval(daily)
    comparison_rows.append({
        "scope": "pooled",
        "comparison": "daily_DG_minus_no_query",
        "difference": gain_ci.pop("system_gain"),
        **gain_ci,
    })
    pd.DataFrame(comparison_rows).to_csv(
        OUT / "tables/final_audit_driver_cluster_comparisons.csv", index=False
    )
    return routes


def full_plan_comparison(primary_routes):
    full_rows = []
    for origin in [13, 19, 25]:
        full = pd.read_csv(SCENARIO / f"tables/w{origin}_privileged_full_plan.csv").rename(
            columns={"origin": "window"}
        )
        full_rows.append(full)
    full = pd.concat(full_rows, ignore_index=True)
    baseline = primary_routes[primary_routes.allocation.eq("daily_DG")][
        ["window", "Route ID", "driver", "baseline_mae", "route_size"]
    ]
    full = full.merge(baseline, on=["window", "Route ID", "driver"], validate="one_to_one")
    full["system_gain"] = full.baseline_mae - full.mae
    full["allocation"] = "privileged_full_plan"

    comparisons = []
    policy_sets = {
        "daily_DG_fixed_past_budget": primary_routes[primary_routes.allocation.eq("daily_DG")],
    }
    joint = pd.read_parquet(OUT / "tables/joint_past_only_route_results.parquet")
    policy_sets["daily_DG_joint_past_only_minimum"] = joint[
        joint.k.eq(5)
        & joint.condition.eq("proxy_clean")
        & joint.method.eq("DG")
        & joint.rule.eq("minimum_mae")
    ].copy()
    policy_sets["daily_DG_joint_past_only_knee"] = joint[
        joint.k.eq(5)
        & joint.condition.eq("proxy_clean")
        & joint.method.eq("DG")
        & joint.rule.eq("knee90")
    ].copy()

    for label, policy in policy_sets.items():
        stats = ra.driver_bootstrap_difference(policy, full, seed=20260914)
        comparisons.append(
            {
                "policy": label,
                "policy_mae": policy.mae.mean(),
                "full_plan_mae": full.mae.mean(),
                "policy_gain": policy.system_gain.mean(),
                "full_plan_gain": full.system_gain.mean(),
                "policy_minus_full_plan_gain": stats["difference"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "queries_per_route": policy.attempts.mean(),
                "routes": stats["routes"],
                "drivers": stats["drivers"],
            }
        )
    pd.DataFrame(comparisons).to_csv(
        OUT / "tables/final_audit_vs_privileged_full_plan.csv", index=False
    )


def main():
    freeze_protocol()
    routes = selected_task_and_workload_audit()
    full_plan_comparison(routes)
    (OUT / "final_audit_completion.json").write_text(
        json.dumps({"completed": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf8",
    )
    print("FINAL AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
