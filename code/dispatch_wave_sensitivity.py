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
BUDGET_OUT = ROOT / "outputs/TRE_query_budget_validation_20260913"
OUT = ROOT / "outputs/TRE_operational_batch_validation_20260913"

sys.path.insert(0, str(ROOT / "code"))
import query_budget_frontier as qb
import route_adaptive_allocation as ra
import operational_batch_validation as op


WAVE_SIZES = [1, 2, 5, 10, 20, 50, "daily"]
REPS = 50
PROTOCOL = {
    "date": "2026-09-14",
    "status": "Specified before inspecting dispatch-wave sensitivity results.",
    "question": (
        "How many simultaneously coordinatable routes are needed for cross-route DG allocation to "
        "improve on equal per-route quotas?"
    ),
    "design": (
        "Within each route-start country/day batch, assign routes by outcome-independent seeded hashes "
        "to waves of at most 1, 2, 5, 10, 20, or 50 routes. Repeat 50 partitions and average route-level "
        "outcomes. The daily condition leaves the full day batch intact."
    ),
    "policy": (
        "K=5, previously frozen clean-answer DG minimum-MAE budget per window, at least one query per "
        "route, at most ten, score divided by route size, exact same total attempts within every wave."
    ),
    "statistics": "Paired driver-cluster bootstrap on the Monte Carlo mean route outcomes, 2,000 resamples.",
    "limits": "Synthetic wave partitions measure coordination-scale sensitivity, not observed depot waves.",
}


def freeze_protocol():
    path = OUT / "dispatch_wave_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def route_hash(route, seed):
    return int(hashlib.sha256(f"{seed}|{route}".encode()).hexdigest()[:16], 16)


def wave_selection(q, priority, budget, wave_size, seed):
    if wave_size == "daily":
        return op.daily_batch_selection(q, priority, budget, cap=10, minimum=1)
    selected = np.zeros(len(q), dtype=bool)
    priority = np.asarray(priority, dtype=float)
    routes = q[["Route ID"] + op.BATCH_COLUMNS].drop_duplicates("Route ID").copy()
    wave_map = {}
    for batch, group in routes.groupby(op.BATCH_COLUMNS, sort=True):
        ordered = sorted(group["Route ID"].tolist(), key=lambda r: route_hash(r, seed))
        for pos, route in enumerate(ordered):
            wave_map[route] = (*batch, pos // int(wave_size))
    q_wave = q["Route ID"].map(wave_map)
    groups = {}
    for i, wave in enumerate(q_wave):
        groups.setdefault(wave, []).append(i)
    for ids_list in groups.values():
        ids = np.asarray(ids_list, dtype=int)
        sub = q.iloc[ids].reset_index(drop=True)
        target = int(np.minimum(budget, sub.groupby("Route ID").size()).sum())
        local = op.constrained_batch_selection(sub, priority[ids], target, cap=10, minimum=1)
        selected[ids] = local
    return selected


def expected_routes(q, origin, budget, wave_size):
    if wave_size == "daily":
        selected = wave_selection(q, q.DG_score.to_numpy() / q.route_size.to_numpy(), budget, wave_size, 0)
        route = ra.make_route_rows(q, selected, origin, 5, "DG", budget, "daily", "proxy_clean")
        return route

    route_ids = np.sort(q["Route ID"].unique())
    index = {route: i for i, route in enumerate(route_ids)}
    sizes = q.groupby("Route ID").size().reindex(route_ids).to_numpy()
    drivers = q.groupby("Route ID")["Driver ID"].first().reindex(route_ids).to_numpy()
    baseline = q.groupby("Route ID").base_error.mean().reindex(route_ids).to_numpy()
    gains = np.zeros((REPS, len(route_ids)))
    attempts = np.zeros((REPS, len(route_ids)))
    priority = q.DG_score.to_numpy() / q.route_size.to_numpy()
    for rep in range(REPS):
        selected = wave_selection(q, priority, budget, wave_size, 20260914 + rep)
        z = q.assign(_selected=selected)
        gain = z.loc[z._selected].groupby("Route ID").clean_gain.sum()
        att = z.groupby("Route ID")._selected.sum()
        for route, value in gain.items():
            gains[rep, index[route]] = value / sizes[index[route]]
        attempts[rep] = att.reindex(route_ids, fill_value=0).to_numpy()
    mean_gain = gains.mean(axis=0)
    return pd.DataFrame(
        {
            "Route ID": route_ids,
            "driver": drivers,
            "baseline_mae": baseline,
            "route_size": sizes,
            "attempts": attempts.mean(axis=0),
            "system_gain": mean_gain,
            "mae": baseline - mean_gain,
            "window": origin,
            "allocation": str(wave_size),
        }
    )


def main():
    freeze_protocol()
    choices = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    choices = choices[
        choices.k.eq(5)
        & choices.method.eq("DG")
        & choices.condition.eq("proxy_clean")
        & choices.rule.eq("minimum_mae")
    ]
    rows = []
    for choice in choices.itertuples(index=False):
        name = f"w{choice.window}_k5"
        freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
        q = op.add_fields(ra.load_window(choice.window, 5), freeze["refusal_thresholds"]["0.5"]).reset_index(drop=True)
        budget = int(choice.chosen_budget)
        for wave_size in WAVE_SIZES:
            route = expected_routes(q, choice.window, budget, wave_size)
            route["wave_size"] = str(wave_size)
            rows.append(route)
        print(time.strftime("%H:%M:%S"), choice.window, flush=True)

    routes = pd.concat(rows, ignore_index=True)
    routes.to_parquet(OUT / "tables/dispatch_wave_route_results.parquet", index=False)
    summary = routes.groupby("wave_size").agg(
        routes=("Route ID", "size"),
        drivers=("driver", "nunique"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "tables/dispatch_wave_summary.csv", index=False)

    uniform = routes[routes.wave_size.eq("1")]
    comparisons = []
    for wave_size in WAVE_SIZES[1:]:
        alternative = routes[routes.wave_size.eq(str(wave_size))]
        stats = ra.driver_bootstrap_difference(alternative, uniform)
        comparisons.append({"wave_size": str(wave_size), **stats})
    pd.DataFrame(comparisons).to_csv(OUT / "tables/dispatch_wave_driver_cluster_comparisons.csv", index=False)

    completion_path = OUT / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf8"))
    completion.update({"dispatch_wave_complete": True, "dispatch_wave_repetitions": REPS})
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
