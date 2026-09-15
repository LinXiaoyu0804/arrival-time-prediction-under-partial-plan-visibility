import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "6"
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "outputs/TRE_scenario_alignment_20260913"
NEAREST = ROOT / "outputs/TRE_nearest_controls_20260913"
OUT = ROOT / "outputs/TRE_query_budget_validation_20260913"
DATA = ROOT / "data/source"

for folder in ["tables", "figures", "models", "code"]:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
import planned_actual_core as core


# Reuse the exact final-scenario feature construction without executing its driver.
scenario_source = (ROOT / "code/scenario_alignment.py").read_text(encoding="utf8")
exec(scenario_source[scenario_source.index("rename=") : scenario_source.index("\nraw=[]")])


BUDGETS = [0, 1, 2, 3, 5, 10, "all"]
METHODS = ["DG", "ER", "uncertainty", "random"]
CONDITIONS = ["proxy_clean", "selective_nonresponse_severe"]
MODEL_PARAMETERS = dict(
    objective="regression",
    n_estimators=200,
    learning_rate=0.04,
    num_leaves=15,
    min_child_samples=100,
    reg_lambda=10,
    random_state=20260913,
    n_jobs=6,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)


PROTOCOL = {
    "date": "2026-09-13",
    "status": (
        "Retrospective validation. The K=5 pooled clean-answer curve at the listed budgets "
        "was inspected before this script was frozen. All K=3 results, window-specific curves, "
        "nested budget choices, response curves, distributions, and intervals remain fully reported."
    ),
    "question": (
        "Does selective use of a mismatched plan-stage signal yield a useful information-error "
        "frontier, and can a budget chosen only from past data transfer to the next time window?"
    ),
    "reuse": (
        "Reuse final actual-history-only base and stage-aware ETA models, task eligibility, feature "
        "allowlist, three forward windows, software-plan proxy answers, and deterministic tie break."
    ),
    "answer_categories": [3, 5],
    "query_budgets_per_route": BUDGETS,
    "policies": {
        "DG": "Predicted supplied-answer loss reduction.",
        "ER": "Predicted baseline absolute error risk.",
        "uncertainty": "Normalized service-stage entropy.",
        "random": "Exact expectation for uniform sampling without replacement.",
    },
    "budget_selection": (
        "Split each six-week calibration block temporally: first four weeks fit temporary DG and ER "
        "scores; final two weeks choose a budget. Refit/reuse scores trained on all six weeks and apply "
        "the frozen budget to the next evaluation window. Report minimum-MAE and 90%-of-best-gain knee "
        "rules. Ties favor the smaller budget."
    ),
    "metrics": (
        "Primary route-equal MAE. Also report system gain, attempted queries, gain per selected query, "
        "selected-task gain distribution, and the fraction of the observed gap to the privileged "
        "full-plan reference."
    ),
    "response": (
        "Clean proxy answers and the previously frozen severe entropy-linked nonresponse scenario. "
        "The latter is a sensitivity model, not observed response behavior."
    ),
    "statistics": "Paired route bootstrap with 2,000 resamples, seed 20260913.",
    "limits": (
        "No observed human responses, query costs, customer outcomes, or untouched dataset. Budget "
        "curves characterize predictive value; they do not estimate economic value."
    ),
}


def freeze_protocol():
    target = OUT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text(encoding="utf8")) == PROTOCOL
    else:
        target.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def load_raw():
    pieces = []
    for part in ["training", "validation", "confirmatory_evaluation"]:
        z = pd.read_parquet(DATA / f"work_experiments/planned_actual_v1/partitions/{part}.parquet")
        z = z[z["_strict_primary"].astype(str).str.lower().eq("true")].copy()
        pieces.append(z)
    z = pd.concat(pieces, ignore_index=True)
    z["_rowkey"] = z["Route ID"].astype(str) + "|" + z.IndexP.astype(str)
    assert z._rowkey.is_unique
    return z


def answer_losses(raw_part, model, k):
    q = current_features(raw_part, model)
    truth = core.add_route_fields(raw_part).copy()
    truth["planned_phase"] = stage(truth, "IndexP", k)
    truth["actual_phase"] = stage(truth, "IndexA", k)
    truth = truth.set_index("_rowkey")
    q["planned_phase"] = q._rowkey.map(truth.planned_phase)
    q["actual_phase"] = q._rowkey.map(truth.actual_phase)
    predictions = []
    for answer in range(k):
        x = q[FF].copy()
        x["answer_phase"] = answer
        predictions.append(model["aware"].predict(x))
    predictions = np.stack(predictions, axis=1)
    y = q.target_minutes.to_numpy()
    ix = np.arange(len(q))
    q["base_error"] = np.abs(q.base_eta.to_numpy() - y)
    q["proxy_error"] = np.abs(predictions[ix, q.planned_phase.to_numpy()] - y)
    q["actual_stage_error"] = np.abs(predictions[ix, q.actual_phase.to_numpy()] - y)
    q["clean_gain"] = q.base_error - q.proxy_error
    return q


def route_weights(q):
    w = 1 / q.groupby("Route ID")["Route ID"].transform("size")
    return (w / w.mean()).to_numpy()


def fit_temporary_scores(q, features):
    weights = route_weights(q)
    dg = LGBMRegressor(**MODEL_PARAMETERS).fit(
        q[features], q.base_error - q.actual_stage_error, sample_weight=weights
    )
    er = LGBMRegressor(**MODEL_PARAMETERS).fit(q[features], q.base_error, sample_weight=weights)
    return dg, er


def add_tie(q):
    q = q.copy()
    q["_tie"] = [
        int(hashlib.sha256(f"{a}|{b}|{c}".encode()).hexdigest()[:12], 16)
        for a, b, c in zip(q["Route ID"], q["Stop ID"], q["Address ID"])
    ]
    return q


def budget_number(budget, mean_route_size=None):
    if budget == "all":
        return float("inf") if mean_route_size is None else float(mean_route_size)
    return float(budget)


def deterministic_selection(q, score_column, budget):
    if budget == 0:
        return np.zeros(len(q), dtype=bool)
    if budget == "all":
        return np.ones(len(q), dtype=bool)
    selected = np.zeros(len(q), dtype=bool)
    values = q[score_column].to_numpy()
    route_codes, _ = pd.factorize(q["Route ID"], sort=True)
    for route_code in range(route_codes.max() + 1):
        ids = np.flatnonzero(route_codes == route_code)
        order = ids[np.lexsort((q._tie.to_numpy()[ids], -values[ids]))]
        selected[order[: min(int(budget), len(order))]] = True
    return selected


def exact_random_route_rows(q, budget, gain_column):
    route = q.groupby("Route ID").agg(
        driver=("Driver ID", "first"),
        baseline_mae=("base_error", "mean"),
        route_size=("base_error", "size"),
        mean_gain=(gain_column, "mean"),
    )
    if budget == "all":
        route["attempts"] = route.route_size
    else:
        route["attempts"] = np.minimum(int(budget), route.route_size)
    route["system_gain"] = route.attempts * route.mean_gain / route.route_size
    route["mae"] = route.baseline_mae - route.system_gain
    route["selected_gain_sum"] = route.attempts * route.mean_gain
    return route.reset_index()


def deterministic_route_rows(q, score_column, budget, gain_column):
    selected = deterministic_selection(q, score_column, budget)
    z = q.assign(_selected=selected)
    route = z.groupby("Route ID").agg(
        driver=("Driver ID", "first"),
        baseline_mae=("base_error", "mean"),
        route_size=("base_error", "size"),
        attempts=("_selected", "sum"),
    )
    selected_sum = z.loc[z._selected].groupby("Route ID")[gain_column].sum()
    route["selected_gain_sum"] = selected_sum.reindex(route.index, fill_value=0)
    route["system_gain"] = route.selected_gain_sum / route.route_size
    route["mae"] = route.baseline_mae - route.system_gain
    return route.reset_index(), selected


def evaluate_curve(q, window, k, split, condition, score_columns):
    gain_column = "clean_gain" if condition == "proxy_clean" else "response_gain"
    route_rows = []
    distribution_rows = []
    for method in METHODS:
        for budget in BUDGETS:
            if method == "random":
                route = exact_random_route_rows(q, budget, gain_column)
                selected = None
            else:
                route, selected = deterministic_route_rows(q, score_columns[method], budget, gain_column)
            route["window"] = window
            route["k"] = k
            route["split"] = split
            route["condition"] = condition
            route["method"] = method
            route["budget"] = str(budget)
            route_rows.append(route)
            if selected is not None and selected.any():
                g = q.loc[selected, gain_column]
                base = q.loc[selected, "base_error"]
                distribution_rows.append(
                    {
                        "window": window,
                        "k": k,
                        "split": split,
                        "condition": condition,
                        "method": method,
                        "budget": str(budget),
                        "queries": int(selected.sum()),
                        "baseline_error_mean": base.mean(),
                        "gain_mean": g.mean(),
                        "gain_median": g.median(),
                        "relative_gain_percent": 100 * g.mean() / base.mean(),
                        "positive_percent": 100 * (g > 0).mean(),
                        "harm_percent": 100 * (g < 0).mean(),
                        "over_30_percent": 100 * (g > 30).mean(),
                        "gain_q10": g.quantile(0.1),
                        "gain_q90": g.quantile(0.9),
                    }
                )
    return pd.concat(route_rows, ignore_index=True), pd.DataFrame(distribution_rows)


def summarize_curve(route_rows):
    keys = ["split", "condition", "k", "method", "budget"]
    out = route_rows.groupby(keys).agg(
        routes=("Route ID", "size"),
        baseline_mae=("baseline_mae", "mean"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
        selected_gain_sum=("selected_gain_sum", "sum"),
        total_attempts=("attempts", "sum"),
    ).reset_index()
    out["gain_per_query"] = out.selected_gain_sum / out.total_attempts.replace(0, np.nan)
    out["relative_system_gain_percent"] = 100 * out.system_gain / out.baseline_mae
    return out


def choose_budgets(summary):
    rows = []
    for keys, group in summary.groupby(["window", "k", "condition", "method"]):
        window, k, condition, method = keys
        if method not in ["DG", "ER"]:
            continue
        group = group.copy()
        mean_n = group.queries_per_route.max()
        group["budget_order"] = group.budget.map(
            lambda x: budget_number(x, mean_n)
        )
        best_gain = group.system_gain.max()
        best = group[group.system_gain.eq(best_gain)].sort_values("budget_order").iloc[0]
        if best_gain <= 0:
            knee = group[group.budget.eq("0")].iloc[0]
        else:
            knee = group[group.system_gain.ge(0.9 * best_gain)].sort_values("budget_order").iloc[0]
        for rule, chosen in [("minimum_mae", best), ("knee90", knee)]:
            rows.append(
                {
                    "window": window,
                    "k": k,
                    "condition": condition,
                    "method": method,
                    "rule": rule,
                    "chosen_budget": chosen.budget,
                    "tuning_gain": chosen.system_gain,
                    "tuning_mae": chosen.mae,
                    "tuning_queries_per_route": chosen.queries_per_route,
                    "best_tuning_gain": best_gain,
                }
            )
    return pd.DataFrame(rows)


def apply_choices(eval_summary, choices):
    rows = []
    for choice in choices.itertuples(index=False):
        hit = eval_summary[
            (eval_summary.window.eq(choice.window))
            & (eval_summary.k.eq(choice.k))
            & (eval_summary.condition.eq(choice.condition))
            & (eval_summary.method.eq(choice.method))
            & (eval_summary.budget.eq(str(choice.chosen_budget)))
        ]
        assert len(hit) == 1
        result = hit.iloc[0]
        oracle_group = eval_summary[
            (eval_summary.window.eq(choice.window))
            & (eval_summary.k.eq(choice.k))
            & (eval_summary.condition.eq(choice.condition))
            & (eval_summary.method.eq(choice.method))
        ]
        oracle_gain = oracle_group.system_gain.max()
        rows.append(
            {
                **choice._asdict(),
                "evaluation_mae": result.mae,
                "evaluation_gain": result.system_gain,
                "evaluation_queries_per_route": result.queries_per_route,
                "evaluation_oracle_best_gain": oracle_gain,
                "evaluation_regret": oracle_gain - result.system_gain,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_intervals(route_rows, reps=2000):
    rng = np.random.default_rng(20260913)
    rows = []
    keys = ["split", "condition", "k", "method", "budget"]
    for key, group in route_rows.groupby(keys):
        values = group.system_gain.to_numpy()
        n = len(values)
        means = np.empty(reps)
        for start in range(0, reps, 100):
            count = min(100, reps - start)
            ids = rng.integers(0, n, size=(count, n))
            means[start : start + count] = values[ids].mean(axis=1)
        rows.append(
            {
                **dict(zip(keys, key)),
                "system_gain": values.mean(),
                "ci_low": np.quantile(means, 0.025),
                "ci_high": np.quantile(means, 0.975),
            }
        )
    return pd.DataFrame(rows)


def main():
    freeze_protocol()
    raw = load_raw()
    weeks = raw.groupby("Route ID")["Week ID"].agg(["min", "max"])
    route_results = []
    distributions = []
    tuning_summaries = []

    for origin, end in [(13, 19), (19, 25), (25, 32)]:
        calibration = raw[
            raw["Route ID"].isin(
                weeks.index[(weeks["min"] >= origin - 6) & (weeks["max"] < origin)]
            )
        ].copy()
        selector_train = calibration[
            calibration["Route ID"].isin(weeks.index[weeks["max"] < origin - 2])
        ].copy()
        budget_tune = calibration[
            calibration["Route ID"].isin(weeks.index[weeks["min"] >= origin - 2])
        ].copy()
        assert len(selector_train) and len(budget_tune)
        assert not set(selector_train["Route ID"]) & set(budget_tune["Route ID"])

        for k in [3, 5]:
            name = f"w{origin}_k{k}"
            eta_model = joblib.load(SCENARIO / f"models/{name}.joblib")
            full_selector = joblib.load(SCENARIO / f"models/{name}_selector.joblib")
            features = full_selector["features"]

            train_q = answer_losses(selector_train, eta_model, k)
            temporary_dg, temporary_er = fit_temporary_scores(train_q, features)
            joblib.dump(
                {"DG": temporary_dg, "ER": temporary_er, "features": features},
                OUT / f"models/{name}_temporary_scores.joblib",
            )

            tune_q = add_tie(answer_losses(budget_tune, eta_model, k))
            tune_q["DG_score"] = temporary_dg.predict(tune_q[features])
            tune_q["ER_score"] = temporary_er.predict(tune_q[features])
            tune_q["uncertainty_score"] = tune_q.entropy

            freeze = json.loads((SCENARIO / f"{name}_freeze.json").read_text(encoding="utf8"))
            severe_threshold = freeze["refusal_thresholds"]["0.5"]
            tune_q["response_gain"] = tune_q.clean_gain * tune_q.entropy.le(severe_threshold)
            score_columns = {
                "DG": "DG_score",
                "ER": "ER_score",
                "uncertainty": "uncertainty_score",
            }
            for condition in CONDITIONS:
                routes, dist = evaluate_curve(
                    tune_q, origin, k, "budget_tuning", condition, score_columns
                )
                route_results.append(routes)
                distributions.append(dist)

            eval_q = pd.read_parquet(SCENARIO / f"cache/{name}_evaluation.parquet")
            eval_q = add_tie(eval_q)
            nearest_scores = pd.read_parquet(NEAREST / f"tables/{name}_scores.parquet")
            eval_q = eval_q.merge(
                nearest_scores[["_rowkey", "error_only"]], on="_rowkey", how="left", sort=False
            )
            eval_q["clean_gain"] = eval_q.base_error - eval_q.proxy_error
            eval_q["response_gain"] = eval_q.clean_gain * eval_q.entropy.le(severe_threshold)
            eval_q["DG_score"] = eval_q.gain_score
            eval_q["ER_score"] = eval_q.error_only
            eval_q["uncertainty_score"] = eval_q.entropy
            for condition in CONDITIONS:
                routes, dist = evaluate_curve(
                    eval_q, origin, k, "evaluation", condition, score_columns
                )
                route_results.append(routes)
                distributions.append(dist)

            print(time.strftime("%H:%M:%S"), name, "completed", flush=True)

    route_results = pd.concat(route_results, ignore_index=True)
    distributions = pd.concat(distributions, ignore_index=True)
    route_results.to_parquet(OUT / "tables/route_results.parquet", index=False)
    distributions.to_csv(OUT / "tables/query_gain_distributions_by_window.csv", index=False)

    window_summary = summarize_curve(route_results)
    # Restore window because summarize_curve intentionally pools by default keys.
    window_summary = route_results.groupby(
        ["window", "split", "condition", "k", "method", "budget"]
    ).agg(
        routes=("Route ID", "size"),
        baseline_mae=("baseline_mae", "mean"),
        mae=("mae", "mean"),
        system_gain=("system_gain", "mean"),
        queries_per_route=("attempts", "mean"),
        selected_gain_sum=("selected_gain_sum", "sum"),
        total_attempts=("attempts", "sum"),
    ).reset_index()
    window_summary["gain_per_query"] = (
        window_summary.selected_gain_sum / window_summary.total_attempts.replace(0, np.nan)
    )
    window_summary["relative_system_gain_percent"] = (
        100 * window_summary.system_gain / window_summary.baseline_mae
    )
    window_summary.to_csv(OUT / "tables/window_budget_curves.csv", index=False)

    pooled = summarize_curve(route_results)
    pooled.to_csv(OUT / "tables/pooled_budget_curves.csv", index=False)
    intervals = bootstrap_intervals(route_results[route_results.split.eq("evaluation")])
    intervals.to_csv(OUT / "tables/evaluation_bootstrap_intervals.csv", index=False)

    tuning = window_summary[window_summary.split.eq("budget_tuning")]
    choices = choose_budgets(tuning)
    evaluation = window_summary[window_summary.split.eq("evaluation")]
    selected = apply_choices(evaluation, choices)
    choices.to_csv(OUT / "tables/past_only_budget_choices.csv", index=False)
    selected.to_csv(OUT / "tables/past_only_budget_evaluation.csv", index=False)

    # Pool selected-query distributions across windows for direct reporting.
    eval_routes = route_results[route_results.split.eq("evaluation")]
    pooled_dist = distributions[distributions.split.eq("evaluation")].groupby(
        ["condition", "k", "method", "budget"], as_index=False
    ).agg(
        queries=("queries", "sum"),
        baseline_error_mean=("baseline_error_mean", "mean"),
        gain_mean=("gain_mean", "mean"),
        gain_median=("gain_median", "mean"),
        relative_gain_percent=("relative_gain_percent", "mean"),
        positive_percent=("positive_percent", "mean"),
        harm_percent=("harm_percent", "mean"),
        over_30_percent=("over_30_percent", "mean"),
        gain_q10=("gain_q10", "mean"),
        gain_q90=("gain_q90", "mean"),
    )
    pooled_dist.to_csv(OUT / "tables/query_gain_distributions_pooled.csv", index=False)

    completion = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "windows": 3,
        "answer_categories": [3, 5],
        "budgets": BUDGETS,
        "route_rows": len(route_results),
        "status": "complete",
    }
    (OUT / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
