# Arrival-time prediction under partial plan visibility

This repository contains the code, public source data, and frozen experimental evidence for a study of selective predeparture confirmation in last-mile delivery.

The operational question is: when a dispatcher can request only a limited number of route-stage confirmations, how should those requests be allocated across routes? The retained experiments compare equal per-route quotas with a predicted-gain allocator under exactly matched total communication capacity. In the primary reconstructed dispatch-cohort analysis, predicted-gain allocation reduces route-equal mean absolute error by 1.044 minutes relative to the equal-quota policy (driver-clustered 95% CI: 0.541 to 1.646 minutes).

## Repository contents

- `code/`: analysis, robustness, reporting, integrity-check, and pipeline scripts.
- `data/source/`: the public Mendeley workbook, deterministic analysis partitions, and provenance records.
- `outputs/TRE_scenario_alignment_20260913/`: scenario bridge and forward predictions.
- `outputs/TRE_nearest_controls_20260913/`: six frozen error-only score files used by the retained comparisons.
- `outputs/TRE_query_budget_validation_20260913/`: query-budget frontier and past-only budget choices.
- `outputs/TRE_operational_batch_validation_20260913/`: final allocation results, robustness analyses, fitted artifacts, protocols, and checksums.
- `outputs/TRE_task_value_heterogeneity_20260921/`: post-primary task-context profiles and route-level allocation downside audit.

Manuscript files, PDF builds, abandoned methods, and unretained exploratory outputs are intentionally excluded.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Verify the frozen release

```bash
python code/verify_release.py
```

This checks the identity of the public source workbook, the frozen evidence hashes, matched query workloads, integrity flags, the primary estimates, and the absence of manuscript source/build files.

## Reproduce analyses

```bash
# Recompute operational analyses from retained intermediate predictions
python code/run_pipeline.py operational

# Refit the scenario bridge, rebuild the budget frontier, and run all analyses
python code/run_pipeline.py full
```

The `full` stage is computationally expensive. The six nearest-control score files are retained as frozen inputs because the upstream exploratory model search is outside the final contribution. All allocation, robustness, and statistical comparisons used by the study remain executable.

The heterogeneity audit is explicitly exploratory. It uses all 59,237 eligible tasks, inverse-route-size weights, and driver-cluster bootstrap intervals to examine pre-query time-window width, predicted route position, and stage uncertainty. It also reports the full route-level distribution of allocation gains and losses. These results motivate monitoring and fallback rules; they are not causal estimates of driver response.

## Data provenance and licenses

The source workbook is the public *Last-mile delivery route deviations dataset: planned vs. actual routes*, version 1, DOI [10.17632/kkwgfvmtxn.1](https://doi.org/10.17632/kkwgfvmtxn.1), distributed under CC BY 4.0. Its byte size and SHA-256 hash are recorded in `data/source/work_experiments/planned_actual_v1/audit/source_manifest.json`.

Repository code is released under the MIT License in `LICENSE`. The third-party source dataset remains subject to its CC BY 4.0 terms.

## Empirical scope

The public records contain software plans and executed routes. Software-plan stages are used as offline proxy answers, and dispatch cohorts are reconstructed from the available calendar fields. The evidence therefore supports retrospective, matched-workload allocation claims; it does not constitute observed driver-response behavior or a field causal trial.
