import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
BUDGET_OUT = ROOT / "outputs/TRE_query_budget_validation_20260913"
OUT = ROOT / "outputs/TRE_operational_batch_validation_20260913"
for folder in ["models", "code"]:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
import query_budget_frontier as qb
import route_adaptive_allocation as ra
import operational_batch_validation as op


BUDGETS = [0, 1, 2, 3, 5, 10]
CAPS = [5, 7, 10, 15, "unlimited"]
PROTOCOL = {
    "date": "2026-09-13",
    "status": (
        "Specified after observing that a separately estimated response probability multiplied by DG "
        "raised response rate but did not improve severe-scenario error."
    ),
    "question": (
        "In the frozen severe entropy-linked nonresponse scenario, can direct estimation of realized "
        "response-weighted gain E[R*d|X] outperform the product of separately estimated response "
        "probability and clean-answer DG?"
    ),
    "training": (
        "For each window and stage granularity, train a temporary LightGBM regressor on the first four "
        "calibration weeks and choose budget/cap on the final two weeks. Refit on all six calibration "
        "weeks, then apply the frozen configuration to the forward evaluation window."
    ),
    "target": (
        "Synthetic response indicator times realized proxy-plan-stage error reduction. This target is "
        "available only because the sensitivity scenario defines responses for every retrospective task."
    ),
    "allocation": (
        "Daily country batch, minimum one query per route for positive budgets, score divided by route "
        "size, candidate budgets 0/1/2/3/5/10 and caps 5/7/10/15/unlimited."
    ),
    "comparators": [
        "plain DG under the same selected budget and cap",
        "uniform per-route DG under the same selected budget",
        "the previously tested separate p(response) times DG policy",
    ],
    "statistics": "Paired driver-cluster bootstrap, 2,000 resamples, seed 20260913.",
    "limit": (
        "This tests estimator structure inside a synthetic response scenario. It cannot establish human "
        "response probabilities or field effectiveness."
    ),
}


def freeze_protocol():
    path = OUT / "direct_response_gain_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def add_response_fields(q, threshold):
    q = op.add_fields(q, threshold)
    return q.reset_index(drop=True)


def fit_direct(q, features):
    weights = qb.route_weights(q)
    return LGBMRegressor(**qb.MODEL_PARAMETERS).fit(
        q[features], q.response_gain, sample_weight=weights
    )


def cap_candidates(budget):
    if budget == 0:
        return ["none"]
    return [cap for cap in CAPS if cap == "unlimited" or int(cap) >= budget]


def select_tasks(q, score, budget, cap):
    if budget == 0:
        return np.zeros(len(q), dtype=bool)
    return op.daily_batch_selection(q, score / q.route_size, budget, cap=cap, minimum=1)


def evaluate_candidates(q, score):
    rows = []
    for budget in BUDGETS:
        for cap in cap_candidates(budget):
            selected = select_tasks(q, score, budget, cap)
            route = ra.make_route_rows(
                q,
                selected,
                origin=-1,
                k=-1,
                method="direct_response_gain",
                budget=budget,
                allocation="candidate",
                condition="selective_nonresponse_severe",
            )
            rows.append(
                {
                    "budget": budget,
                    "cap": str(cap),
                    "system_gain": route.system_gain.mean(),
                    "mae": route.mae.mean(),
                    "queries_per_route": route.attempts.mean(),
                    "response_rate": route.responses.sum() / max(route.attempts.sum(), 1),
                }
            )
    return pd.DataFrame(rows)


def choose(table, rule):
    table = table.copy()
    table["cap_order"] = table.cap.map(
        lambda x: 0 if x == "none" else (999 if x == "unlimited" else int(x))
    )
    best_gain = table.system_gain.max()
    if rule == "minimum_mae":
        eligible = table[table.system_gain.eq(best_gain)]
    else:
        eligible = table[table.system_gain.ge(0.9 * best_gain)]
    return eligible.sort_values(["budget", "cap_order"]).iloc[0], best_gain


def main():
    freeze_protocol()
    raw = qb.load_raw()
    weeks = raw.groupby("Route ID")["Week ID"].agg(["min", "max"])
    all_routes = []
    choice_rows = []

    for origin in [13, 19, 25]:
        calibration = raw[
            raw["Route ID"].isin(weeks.index[(weeks["min"] >= origin - 6) & (weeks["max"] < origin)])
        ].copy()
        selector_train = calibration[
            calibration["Route ID"].isin(weeks.index[weeks["max"] < origin - 2])
        ].copy()
        budget_tune = calibration[
            calibration["Route ID"].isin(weeks.index[weeks["min"] >= origin - 2])
        ].copy()
        for k in [3, 5]:
            name = f"w{origin}_k{k}"
            eta_model = joblib.load(SCENARIO / f"models/{name}.joblib")
            selector = joblib.load(SCENARIO / f"models/{name}_selector.joblib")
            features = selector["features"]
            freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
            threshold = freeze["refusal_thresholds"]["0.5"]

            train_q = add_response_fields(qb.answer_losses(selector_train, eta_model, k), threshold)
            tune_q = add_response_fields(qb.answer_losses(budget_tune, eta_model, k), threshold)
            full_q = add_response_fields(qb.answer_losses(calibration, eta_model, k), threshold)
            evaluation = add_response_fields(ra.load_window(origin, k), threshold)

            temporary = fit_direct(train_q, features)
            full = fit_direct(full_q, features)
            joblib.dump(
                {"model": full, "features": features, "synthetic_response_threshold": threshold},
                OUT / f"models/{name}_direct_response_gain.joblib",
            )
            tune_score = temporary.predict(tune_q[features])
            eval_score = full.predict(evaluation[features])
            candidates = evaluate_candidates(tune_q, tune_score)
            candidates["window"] = origin
            candidates["k"] = k
            candidates.to_csv(OUT / f"tables/w{origin}_k{k}_direct_response_candidates.csv", index=False)

            for rule in ["minimum_mae", "knee90"]:
                selected_config, best_gain = choose(candidates, rule)
                budget = int(selected_config.budget)
                cap = selected_config.cap
                direct_selected = select_tasks(evaluation, eval_score, budget, cap)
                plain_selected = select_tasks(evaluation, evaluation.DG_score.to_numpy(), budget, cap)
                uniform_selected = (
                    np.zeros(len(evaluation), dtype=bool)
                    if budget == 0
                    else qb.deterministic_selection(evaluation, "DG_score", budget)
                )
                for label, selected in {
                    "direct_response_gain": direct_selected,
                    "plain_DG_same_config": plain_selected,
                    "uniform_DG_same_budget": uniform_selected,
                }.items():
                    route = ra.make_route_rows(
                        evaluation,
                        selected,
                        origin,
                        k,
                        "direct_response_gain",
                        budget,
                        label,
                        "selective_nonresponse_severe",
                    )
                    route["rule"] = rule
                    route["chosen_cap"] = cap
                    all_routes.append(route)
                direct_route = all_routes[-3]
                choice_rows.append(
                    {
                        "window": origin,
                        "k": k,
                        "rule": rule,
                        "chosen_budget": budget,
                        "chosen_cap": cap,
                        "tuning_gain": selected_config.system_gain,
                        "best_tuning_gain": best_gain,
                        "evaluation_gain": direct_route.system_gain.mean(),
                        "evaluation_mae": direct_route.mae.mean(),
                        "evaluation_queries_per_route": direct_route.attempts.mean(),
                        "evaluation_response_rate": direct_route.responses.sum() / max(direct_route.attempts.sum(), 1),
                    }
                )
            print(time.strftime("%H:%M:%S"), name, flush=True)

    routes = pd.concat(all_routes, ignore_index=True)
    routes.to_parquet(OUT / "tables/direct_response_gain_route_results.parquet", index=False)
    choices = pd.DataFrame(choice_rows)
    choices.to_csv(OUT / "tables/direct_response_gain_choices.csv", index=False)

    summary_rows = []
    comparison_rows = []
    for key, group in routes.groupby(["k", "rule"]):
        k, rule = key
        for allocation, z in group.groupby("allocation"):
            summary_rows.append(
                {
                    "k": k,
                    "rule": rule,
                    "allocation": allocation,
                    "baseline_mae": z.baseline_mae.mean(),
                    "mae": z.mae.mean(),
                    "system_gain": z.system_gain.mean(),
                    "queries_per_route": z.attempts.mean(),
                    "response_rate": z.responses.sum() / max(z.attempts.sum(), 1),
                }
            )
        direct = group[group.allocation.eq("direct_response_gain")]
        for comparator in ["plain_DG_same_config", "uniform_DG_same_budget"]:
            stats = ra.driver_bootstrap_difference(direct, group[group.allocation.eq(comparator)])
            comparison_rows.append(
                {
                    "k": k,
                    "rule": rule,
                    "comparison": f"direct_response_gain_minus_{comparator}",
                    **stats,
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUT / "tables/direct_response_gain_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(
        OUT / "tables/direct_response_gain_driver_cluster_comparisons.csv", index=False
    )

    completion_path = OUT / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf8"))
    completion.update(
        {
            "direct_response_gain_complete": True,
            "direct_response_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
