import json
import shutil
import sys
import time
from pathlib import Path

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
    "status": "Specified after the final audit and before inspecting anchor-sensitivity outputs.",
    "question": (
        "Does the same-batch capacity-allocation conclusion depend on reconstructing a route's "
        "dispatch day from its first executed stop?"
    ),
    "population": "K=5 clean proxy-answer evaluation windows; past-only minimum-MAE DG budgets.",
    "policy": "DG score divided by route size, minimum one and maximum ten queries per route.",
    "anchors": {
        "earliest_executed": "Country-week-weekday of minimum IndexA; primary definition.",
        "latest_executed": "Country-week-weekday of maximum IndexA.",
        "modal_event_date": "Most frequent country-week-weekday among route tasks; deterministic ties.",
        "strict_single_date_routes": "Exclude routes whose tasks span more than one country-week-weekday.",
    },
    "matched_capacity": "Exactly the uniform-route attempt count within every reconstructed batch.",
    "statistics": "Paired driver-cluster bootstrap, 2,000 resamples, seed 20260914.",
}


def freeze_protocol():
    path = OUT / "batch_anchor_sensitivity_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def load_eval(origin):
    freeze = json.loads((SCENARIO / f"w{origin}_k5_freeze.json").read_text(encoding="utf8"))
    q = op.add_fields(ra.load_window(origin, 5), freeze["refusal_thresholds"]["0.5"])
    return q.reset_index(drop=True)


def choices():
    z = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    return z[
        z.k.eq(5)
        & z.method.eq("DG")
        & z.condition.eq("proxy_clean")
        & z.rule.eq("minimum_mae")
    ].sort_values("window")


def set_route_batch(q, definition):
    q = q.copy()
    source = ["Country", "Week ID", "Day of Week"]
    if definition == "earliest_executed":
        anchors = q.sort_values(["Route ID", "IndexA", "_tie"]).groupby("Route ID").first()
    elif definition == "latest_executed":
        anchors = q.sort_values(["Route ID", "IndexA", "_tie"]).groupby("Route ID").last()
    elif definition == "modal_event_date":
        counts = q.groupby(["Route ID"] + source).size().rename("n").reset_index()
        anchors = (
            counts.sort_values(["Route ID", "n", "Country", "Week ID", "Day of Week"],
                               ascending=[True, False, True, True, True])
            .groupby("Route ID").first()
        )
    elif definition == "strict_single_date_routes":
        unique_dates = q.groupby("Route ID")[source].apply(lambda x: len(x.drop_duplicates()))
        q = q[q["Route ID"].isin(unique_dates.index[unique_dates.eq(1)])].reset_index(drop=True)
        q["route_size"] = q.groupby("Route ID")["Route ID"].transform("size")
        anchors = q.groupby("Route ID").first()
    else:
        raise ValueError(definition)
    rename = dict(zip(source, op.BATCH_COLUMNS))
    anchors = anchors[source].rename(columns=rename)
    for column in op.BATCH_COLUMNS:
        q[column] = q["Route ID"].map(anchors[column])
    assert q.groupby("Route ID")[op.BATCH_COLUMNS].nunique().to_numpy().max() == 1
    return q


def main():
    freeze_protocol()
    route_rows = []
    audit_rows = []
    for choice in choices().itertuples(index=False):
        origin = int(choice.window)
        budget = int(choice.chosen_budget)
        base = load_eval(origin)
        source = ["Country", "Week ID", "Day of Week"]
        cross = base.groupby("Route ID")[source].apply(lambda x: len(x.drop_duplicates())).gt(1)
        for definition in PROTOCOL["anchors"]:
            q = set_route_batch(base, definition)
            uniform = qb.deterministic_selection(q, "DG_score", budget)
            daily = op.daily_batch_selection(
                q, q.DG_score.to_numpy() / q.route_size.to_numpy(), budget, cap=10, minimum=1
            )
            assert int(uniform.sum()) == int(daily.sum())
            capacity = q.assign(uniform=uniform, daily=daily).groupby(op.BATCH_COLUMNS).agg(
                uniform=("uniform", "sum"), daily=("daily", "sum")
            )
            assert capacity.uniform.eq(capacity.daily).all()
            for label, selected in [("uniform_DG", uniform), ("daily_DG", daily)]:
                z = ra.make_route_rows(
                    q, selected, origin, 5, "DG", budget, label, "proxy_clean"
                )
                z["anchor_definition"] = definition
                route_rows.append(z)
            audit_rows.append({
                "window": origin,
                "anchor_definition": definition,
                "routes": q["Route ID"].nunique(),
                "tasks": len(q),
                "batches": q[op.BATCH_COLUMNS].drop_duplicates().shape[0],
                "cross_date_routes_in_full_window": int(cross.sum()),
                "attempts": int(daily.sum()),
                "all_batch_capacities_equal": True,
            })
        print(time.strftime("%H:%M:%S"), "anchor sensitivity", origin, flush=True)

    routes = pd.concat(route_rows, ignore_index=True)
    routes.to_parquet(OUT / "tables/batch_anchor_route_results.parquet", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "tables/batch_anchor_audit.csv", index=False)
    summary = routes.groupby(["anchor_definition", "allocation"]).agg(
        routes=("Route ID", "size"),
        drivers=("driver", "nunique"),
        baseline_mae=("baseline_mae", "mean"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "tables/batch_anchor_summary.csv", index=False)

    comparisons = []
    for definition, group in routes.groupby("anchor_definition"):
        daily = group[group.allocation.eq("daily_DG")]
        uniform = group[group.allocation.eq("uniform_DG")]
        comparisons.append({
            "anchor_definition": definition,
            **ra.driver_bootstrap_difference(daily, uniform, seed=20260914),
        })
    pd.DataFrame(comparisons).to_csv(
        OUT / "tables/batch_anchor_driver_cluster_comparisons.csv", index=False
    )
    (OUT / "batch_anchor_sensitivity_completion.json").write_text(
        json.dumps({"completed": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf8",
    )
    print("BATCH ANCHOR SENSITIVITY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
