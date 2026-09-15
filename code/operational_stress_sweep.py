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

sys.path.insert(0, str(ROOT / "code"))
import query_budget_frontier as qb
import route_adaptive_allocation as ra
import operational_batch_validation as op


PROTOCOL = {
    "date": "2026-09-13",
    "status": "Specified before inspecting multi-query daily-batch stress results.",
    "question": (
        "Does the clean-answer, past-only daily-batch policy remain useful under refusal and adjacent-stage "
        "answer errors when it queries multiple tasks per route?"
    ),
    "policy": (
        "Reuse the clean-answer K=3/K=5 DG minimum-MAE budget chosen from past data. Compare equal route "
        "quotas with daily country-batch DG/route-size allocation, minimum one and cap ten."
    ),
    "conditions": [
        "clean proxy answer",
        "independent 25% and 50% refusal (exact expectation)",
        "entropy-linked refusal at calibration 75th and 50th percentile thresholds",
        "10% and 20% adjacent-stage answer error (exact expectation)",
    ],
    "statistics": "Paired driver-cluster bootstrap, 2,000 resamples, seed 20260913.",
    "limits": "All response and answer-error processes are sensitivity scenarios, not human observations.",
}


def freeze_protocol():
    path = OUT / "operational_stress_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == PROTOCOL
    else:
        path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def condition_gains(q, model, k, thresholds):
    predictions = []
    for answer in range(k):
        x = q[qb.FF].copy()
        x["answer_phase"] = answer
        predictions.append(model["aware"].predict(x))
    predictions = np.stack(predictions, axis=1)
    y = q.target_minutes.to_numpy()
    ix = np.arange(len(q))
    planned = q.planned_phase.to_numpy()
    base = np.abs(q.base_eta.to_numpy() - y)
    clean_error = np.abs(predictions[ix, planned] - y)
    left = np.maximum(planned - 1, 0)
    right = np.minimum(planned + 1, k - 1)
    adjacent_error = (
        np.abs(predictions[ix, left] - y) + np.abs(predictions[ix, right] - y)
    ) / 2
    adjacent_error = np.where(
        planned == 0,
        np.abs(predictions[:, 1] - y),
        np.where(planned == k - 1, np.abs(predictions[:, k - 2] - y), adjacent_error),
    )
    clean_gain = base - clean_error
    return {
        "proxy_clean": (clean_gain, np.ones(len(q))),
        "independent_refusal25": (0.75 * clean_gain, np.full(len(q), 0.75)),
        "independent_refusal50": (0.50 * clean_gain, np.full(len(q), 0.50)),
        "entropy_refusal25": (
            clean_gain * q.entropy.le(thresholds["0.75"]).to_numpy(),
            q.entropy.le(thresholds["0.75"]).to_numpy(float),
        ),
        "entropy_refusal50": (
            clean_gain * q.entropy.le(thresholds["0.5"]).to_numpy(),
            q.entropy.le(thresholds["0.5"]).to_numpy(float),
        ),
        "adjacent_error10": (base - (0.9 * clean_error + 0.1 * adjacent_error), np.ones(len(q))),
        "adjacent_error20": (base - (0.8 * clean_error + 0.2 * adjacent_error), np.ones(len(q))),
    }


def make_rows(q, selected, gain, response, origin, k, budget, policy, condition):
    z = q.assign(_selected=selected, _gain=gain, _response=response)
    route = z.groupby("Route ID").agg(
        driver=("Driver ID", "first"),
        baseline_mae=("base_error", "mean"),
        route_size=("base_error", "size"),
        attempts=("_selected", "sum"),
    )
    selected_gain = z.loc[z._selected].groupby("Route ID")._gain.sum()
    selected_response = z.loc[z._selected].groupby("Route ID")._response.sum()
    route["selected_gain_sum"] = selected_gain.reindex(route.index, fill_value=0.0)
    route["responses"] = selected_response.reindex(route.index, fill_value=0.0)
    route["system_gain"] = route.selected_gain_sum / route.route_size
    route["mae"] = route.baseline_mae - route.system_gain
    route["window"] = origin
    route["k"] = k
    route["budget"] = str(budget)
    route["policy"] = policy
    route["condition"] = condition
    return route.reset_index()


def main():
    freeze_protocol()
    choices = pd.read_csv(BUDGET_OUT / "tables/past_only_budget_choices.csv")
    choices = choices[
        choices.method.eq("DG") & choices.condition.eq("proxy_clean") & choices.rule.eq("minimum_mae")
    ]
    rows = []
    for choice in choices.itertuples(index=False):
        origin, k, budget = int(choice.window), int(choice.k), int(choice.chosen_budget)
        name = f"w{origin}_k{k}"
        freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
        q = op.add_fields(ra.load_window(origin, k), freeze["refusal_thresholds"]["0.5"]).reset_index(drop=True)
        model = joblib.load(SCENARIO / f"models/{name}.joblib")
        conditions = condition_gains(q, model, k, freeze["refusal_thresholds"])
        uniform = qb.deterministic_selection(q, "DG_score", budget)
        daily = op.daily_batch_selection(q, q.DG_score / q.route_size, budget, cap=10, minimum=1)
        assert uniform.sum() == daily.sum()
        for condition, (gain, response) in conditions.items():
            for policy, selected in {"uniform_DG": uniform, "daily_DG": daily}.items():
                rows.append(make_rows(q, selected, gain, response, origin, k, budget, policy, condition))
        print(time.strftime("%H:%M:%S"), name, flush=True)

    routes = pd.concat(rows, ignore_index=True)
    routes.to_parquet(OUT / "tables/operational_stress_route_results.parquet", index=False)
    summary = routes.groupby(["condition", "k", "policy"]).agg(
        routes=("Route ID", "size"),
        baseline_mae=("baseline_mae", "mean"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
        responses=("responses", "sum"),
        attempts=("attempts", "sum"),
    ).reset_index()
    summary["response_rate"] = summary.responses / summary.attempts
    summary.to_csv(OUT / "tables/operational_stress_summary.csv", index=False)

    comparisons = []
    for (condition, k), group in routes.groupby(["condition", "k"]):
        daily = group[group.policy.eq("daily_DG")]
        uniform = group[group.policy.eq("uniform_DG")]
        versus_uniform = ra.driver_bootstrap_difference(daily, uniform)
        versus_baseline = ra.driver_bootstrap_difference(daily, daily.assign(system_gain=0.0))
        comparisons.append(
            {
                "condition": condition,
                "k": k,
                "comparison": "daily_DG_minus_uniform_DG",
                **versus_uniform,
            }
        )
        comparisons.append(
            {
                "condition": condition,
                "k": k,
                "comparison": "daily_DG_gain_vs_zero",
                **versus_baseline,
            }
        )
    pd.DataFrame(comparisons).to_csv(OUT / "tables/operational_stress_driver_cluster_comparisons.csv", index=False)

    completion_path = OUT / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf8"))
    completion.update({"operational_stress_complete": True, "stress_time": time.strftime("%Y-%m-%d %H:%M:%S")})
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
